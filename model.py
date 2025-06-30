import logging
import torch.nn.functional as F
import torch.nn as nn
import inspect
import torch
import re
import os
import time
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers import normalizers, pre_tokenizers
import math
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ───── Config Class ────────────────────────────────────
class GPTConfig:
    def __init__(self, vocab_size=50257, block_size=256, embed_dim=256,
                 n_heads=4, n_layers=4, dropout=0.1,
                 learning_rate=1e-4, weight_decay=0.1, betas=(0.9, 0.95), training=False,
                 dtype=torch.float32, device='cpu', use_rope=True, norm_type='layernorm'):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.training = training
        self.dtype = dtype
        self.device = device
        self.use_rope = use_rope
        self.norm_type = norm_type

# ───── Rotary Positional Embedding ─────────────────────
def precompute_rope_freqs(seq_len, head_dim, base=10000, device='cuda'):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # shape (seq_len, head_dim//2, 2)

def apply_rope(x, freqs_cis):
    B, T, H, D = x.shape
    x = x.view(B, T, H, D // 2, 2)  # [..., D//2, 2]
    cos = freqs_cis[:, None, :, 0]  # [T, 1, D//2]
    sin = freqs_cis[:, None, :, 1]  # [T, 1, D//2]
    x1 = x[..., 0] * cos - x[..., 1] * sin
    x2 = x[..., 0] * sin + x[..., 1] * cos
    x_out = torch.stack([x1, x2], dim=-1).flatten(-2)
    return x_out

def get_dtype_for_device(device):
    if device == 'cuda':
        return torch.float32  # can make it torch.bfloat16 if gpu is bf16 compatible
    return torch.float32

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / (x.shape[-1] ** 0.5)
        return self.weight * (x / (rms + self.eps))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Register buffers with correct dtype, will be moved with .to()
        self.register_buffer('k_cache', torch.zeros(max_batch_size, n_heads, max_seq_len, head_dim, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros_like(self.k_cache))
        self.register_buffer('current_pos', torch.zeros(max_batch_size, dtype=torch.long), persistent=False)

    def reset(self):
        """Reset position counters for all batches."""
        self.current_pos.zero_()

    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        B, H, S, D = k_val.shape
        device = self.k_cache.device

        k_val = k_val.to(device=device, dtype=self.k_cache.dtype)
        v_val = v_val.to(device=device, dtype=self.v_cache.dtype)

        for b in range(B):
            cur = int(self.current_pos[b].item())  # ✅ ensure int
            end = int(cur + S)                     # ✅ ensure int

            if end > self.k_cache.size(2):
                raise ValueError(f"KVCache overflow: tried to write until {end}, but max is {self.k_cache.size(2)}")
            self.k_cache[b, :, cur:end, :].copy_(k_val[b])
            self.v_cache[b, :, cur:end, :].copy_(v_val[b])
            self.current_pos[b] = end

        max_len = int(self.current_pos.max().item())  # ✅ safer
        return (
            self.k_cache[:, :, :max_len, :],
            self.v_cache[:, :, :max_len, :]
        )

    def to(self, *args, **kwargs):
        """Move all internal buffers to the specified device/dtype."""
        super().to(*args, **kwargs)
        self.k_cache = self.k_cache.to(*args, **kwargs)
        self.v_cache = self.v_cache.to(*args, **kwargs)
        self.current_pos = self.current_pos.to(*args, **kwargs)
        return self


class MHSA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.n_heads == 0
        self.heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.qkv_proj = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=False)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = config.dropout
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.training_flag = config.training
        self.use_rope = config.use_rope

    def forward(self, x, input_pos=None, kv_cache: KVCache = None, freqs_cis=None):
        x = x.to(DTYPE)
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        if self.use_rope and freqs_cis is not None:
            rope = freqs_cis[:T]  # [T, D//2, 2]
            q = apply_rope(q.transpose(1, 2), rope).transpose(1, 2)
            k = apply_rope(k.transpose(1, 2), rope).transpose(1, 2)

        use_kv = kv_cache is not None and input_pos is not None and not self.training
        if use_kv:
            # ✅ Ensure q, k, v all match dtype of kv cache
            target_dtype = kv_cache.k_cache.dtype
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
            k, v = kv_cache.update(input_pos, k, v)
        else:
            # ✅ Otherwise ensure all 3 are in the same dtype (typically AMP controlled)
            target_dtype = q.dtype  # use q's dtype to enforce consistency
            k = k.to(target_dtype)
            v = v.to(target_dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = out.to(self.out_proj.weight.dtype)  # Fix: match Linear weight dtype
        return self.resid_drop(self.out_proj(out))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_class = nn.LayerNorm if config.norm_type == 'layernorm' else RMSNorm
        self.sa = MHSA(config)
        self.ff = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim)
        )
        self.ln1 = norm_class(config.embed_dim)
        self.ln2 = norm_class(config.embed_dim) 
        self.drop = nn.Dropout(config.dropout)
        self.kv_cache = None

    def init_cache(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype):
        self.kv_cache = KVCache(max_batch_size, max_seq_len, n_heads, head_dim, dtype=dtype)
        device = next(self.parameters()).device
        self.kv_cache = self.kv_cache.to(device=device, dtype=dtype)

    def forward(self, x, input_pos=None, freqs_cis=None):
        x = x.to(DTYPE)
        if self.training:
            x = self.ln1(x + self.drop(self.sa(x)))
        else:
            x = self.ln1(x + self.drop(self.sa(x, input_pos=input_pos, kv_cache=self.kv_cache, freqs_cis=freqs_cis)))
        return self.ln2(x + self.drop(self.ff(x))).to(x.dtype)
    


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        norm_class = nn.LayerNorm if config.norm_type == 'layernorm' else RMSNorm
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.embed_dim),
            'wpe': nn.Embedding(config.block_size * 2, config.embed_dim),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            'ln_f': norm_class(config.embed_dim)
        })
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
#        self.transformer['wte'].weight = self.lm_head.weight
        self.dtype = config.dtype
        self.device = config.device
        self.apply(self._init_weights)

        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('ff.2.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        for name, param in self.named_parameters():
            try:
                if param.is_floating_point():
                    param.data = param.data.to(device=config.device, dtype=config.dtype)
                else:
                    param.data = param.data.to(device=config.device)              
            except Exception as e:
                print(f"❌ Failed to move param: {name}")
                print(f"    shape: {param.shape}, dtype: {param.dtype}")
                raise e

        # ✅ Carefully move only initialized weights to correct device/dtype
        for param in self.parameters():
            if param.is_floating_point():
                param.data = param.data.to(device=config.device, dtype=config.dtype)
            else:
                param.data = param.data.to(device=config.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, input_pos=None):
        idx = idx.to(self.device)
        B, T = idx.shape
        pos_ids = torch.arange(T, device=idx.device) if input_pos is None else input_pos
        tok_emb = self.transformer['wte'](idx)

        if self.config.use_rope:
            pos_emb = 0  # Skip learned pos emb if using RoPE
        else:
            pos_emb = self.transformer['wpe'](pos_ids)

        x = self.transformer['drop'](tok_emb + pos_emb).to(DTYPE)

        if self.config.use_rope:
            self.freqs_cis = precompute_rope_freqs(self.config.block_size, self.config.embed_dim // self.config.n_heads, device=idx.device)
        else:
            self.freqs_cis = None

        for block in self.transformer['h']:
            x = block(x, input_pos=pos_ids, freqs_cis=self.freqs_cis if self.config.use_rope else None)
        x = self.transformer['ln_f'](x)
        return self.lm_head(x)
    


    def setup_kv_cache(self, batch_size, total_len):
        for block in self.transformer['h']:
            block.init_cache(batch_size, total_len, self.config.n_heads, self.config.embed_dim // self.config.n_heads, dtype=DTYPE)

    def clear_kv_cache(self):
        for block in self.transformer['h']:
            block.kv_cache = None

    def configure_optimizers(self, device_type):
        weight_decay = self.config.weight_decay
        learning_rate = self.config.learning_rate
        betas = self.config.betas
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        fused_ok = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        extra_args = dict(fused=True) if fused_ok else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=1.0, top_k=None):
        B = token_ids.shape[0]
        prompt_len = token_ids.size(1)
        total_len = prompt_len + max_new_tokens
        input_pos = torch.arange(prompt_len, device=token_ids.device)
        self.setup_kv_cache(B, total_len)

        for step in range(max_new_tokens):
            idx_cond = token_ids[:, -1:]
            pos_cond = torch.full((token_ids.size(0),), prompt_len + step, device=token_ids.device, dtype=torch.long)

            logits = self(idx_cond, input_pos=pos_cond)
            logits = logits[:, -1, :] / temperature

            # 🚨 Safety checks
            if torch.isnan(logits).any():
                raise ValueError("❌ NaNs found in logits!")
            if logits.shape[-1] != self.config.vocab_size:
                raise ValueError(f"❌ Logits vocab size mismatch: expected {self.config.vocab_size}, got {logits.shape[-1]}")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            if torch.isnan(probs).any() or (probs < 0).any():
                raise ValueError("❌ Invalid probabilities in sampling")

            idx_next = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, idx_next), dim=1)

        return token_ids

# ────── Tokenizer Wrapper ────────────────────────────────────
class TokenizerWrapper:
    def __init__(self, path="tokenizer.json"):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

# ────── Dataset Loader ───────────────────────────────────────
import pickle
import os

import os
import pickle
import torch
import numpy as np

class BinDataset:
    def __init__(self, train_path, val_path, meta_path, block_size=256, device='cpu',
                 max_train_len=None, max_val_len=None):
        self.block_size = block_size
        self.device = device

        self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
        self.val_data   = np.memmap(val_path,   dtype=np.uint16, mode='r')

        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.bos_id = meta["bos_id"]
        self.train_bos_pos = torch.tensor(meta["bos_positions"]["train"], dtype=torch.long)
        self.val_bos_pos = torch.tensor(meta["bos_positions"]["val"], dtype=torch.long)

        if max_train_len is not None:
            self.train_bos_pos = self.train_bos_pos[:max_train_len]
        if max_val_len is not None:
            self.val_bos_pos = self.val_bos_pos[:max_val_len]

    def get_stream_length(self, stream_type='train'):
        if stream_type == 'train':
            return len(self.train_bos_pos)
        else:
            return len(self.val_bos_pos)

    def get_batch(self, stream_type='train', batch_size=64):
        
        data = self.train_data if stream_type == 'train' else self.val_data
        bos_pos = self.train_bos_pos if stream_type == 'train' else self.val_bos_pos
        idx = torch.randint(0, len(bos_pos), (batch_size,))
        starts = bos_pos[idx]

        x, y = [], []
        for s in starts:
            if s + self.block_size + 1 < len(data):
                x.append(torch.from_numpy(data[s: s + self.block_size].astype(np.int64)))
                y.append(torch.from_numpy(data[s + 1: s + 1 + self.block_size].astype(np.int64)))

        x = torch.stack(x).to(self.device)
        y = torch.stack(y).to(self.device)
        return x, y

# ────── Trainer Class ────────────────────────────────────────
class Trainer:
    def __init__(self, model, dataset, tokenizer, device='cpu',
                 epochs=10, batch_size=32, print_every=1, grad_accum_steps=1,
                 grad_clip=1.0, checkpoint_dir="checkpoints", warmup_iters=0.1, resume_mode="scratch", use_amp=False):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.optimizer = model.configure_optimizers(device)
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_every = print_every
        self.grad_accum_steps = grad_accum_steps
        self.bos_id = self.tokenizer.encode("<bos>")[0]
        self.checkpoint_dir = checkpoint_dir
        self.resume_mode = resume_mode

        self.decay_lr = True
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.lr_decay_iters = epochs * dataset.get_stream_length('train') // batch_size
        self.warmup_iters = int(warmup_iters * self.lr_decay_iters)
        self.min_lr = self.base_lr / 10
        self.max_iters = self.lr_decay_iters

        self.iter_count = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.DTYPE = get_dtype_for_device(self.device)
        self.USE_AMP = use_amp
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=self.USE_AMP)
        self.grad_clip = grad_clip
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if resume_mode == "resume":
            self._load_checkpoint(os.path.join(self.checkpoint_dir, "ckpt_latest.pt"))

        if torch.__version__ >= "2.0":
            print("📦 Compiling model...")
            self.model = torch.compile(self.model)

    def _save_checkpoint(self, name="ckpt_latest.pt"):
        # ✅ Unwrap compiled model if needed
        model_to_save = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model

        ckpt = {
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "trainer_state": {
                "iter_count": self.iter_count,
                "epoch": self.current_epoch,
                "best_val_loss": self.best_val_loss
            },
            "config": model_to_save.config.__dict__
        }
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(ckpt, path)
        print(f"📦 Saved checkpoint to {path}")


    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"⚠️ No checkpoint found at {path}, starting from scratch.")
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        state = ckpt["trainer_state"]
        self.iter_count = state.get("iter_count", 0)
        self.current_epoch = state.get("epoch", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"✅ Loaded checkpoint from {path}")

    def _get_lr(self):
        if not self.decay_lr:
            return self.base_lr
        it = self.iter_count
        if it < self.warmup_iters:
            return self.base_lr * (it + 1) / self.warmup_iters
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.base_lr - self.min_lr)

    def train(self):
        print("🚀 Starting training...")
        start_time = time.time()
        ep_steps = self.dataset.get_stream_length('train') // self.batch_size
        print(f" batch_size: {self.batch_size} , * block_size: {self.dataset.block_size}, // train_len: {self.dataset.get_stream_length('train')}")
        print(f" current epoch: {self.current_epoch}")
        print(f" epochs: {self.epochs}")
        print(f" steps per epoch: {ep_steps}")

        for epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            losses = torch.zeros(ep_steps, device=self.device)

            for step in range(ep_steps):
                self.optimizer.zero_grad()
                total_loss = 0.0

                for _ in range(self.grad_accum_steps):
                    # 🔁 Call get_batch() once per micro-batch
                    x_batch, y_batch = self.dataset.get_batch('train', batch_size=self.batch_size)
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    with torch.amp.autocast(device_type=self.device, dtype=self.DTYPE, enabled=self.USE_AMP):
                        logits = self.model(x_batch)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                        loss = loss / self.grad_accum_steps  # Normalize for accumulation

                    if self.USE_AMP:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    total_loss += loss.item()

                # ⛓ Gradient clipping and optimizer step
                if self.USE_AMP:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if self.USE_AMP:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # 🔁 Learning rate scheduler
                lr = self._get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.iter_count += 1

                losses[step] = total_loss

            train_loss = losses.mean()

            # 🧪 Validation
            if epoch % self.print_every == 0 or epoch == self.epochs - 1:
                self.model.eval()
                self.model.clear_kv_cache()
                with torch.no_grad():
                    x_val, y_val = self.dataset.get_batch('validation', batch_size=self.batch_size)
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    with torch.amp.autocast(device_type=self.device, dtype=self.DTYPE, enabled=self.USE_AMP):
                        val_logits = self.model(x_val)
                        val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))

                    print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f} | LR: {lr:.6f}")
                    print(f"Model generated: {self._generate('Once upon a time', 50, 0.9, 30)}")
                    self.model.clear_kv_cache()

                if val_loss.item() < self.best_val_loss:
                    self.best_val_loss = val_loss.item()
                    self._save_checkpoint("ckpt_best.pt")

            self._save_checkpoint("ckpt_latest.pt")
            self.current_epoch += 1

        print(f"✅ Training finished in {time.time() - start_time:.1f}s")
        print(f"Model generated: {self._generate('Once upon a time', 50, 0.9, 30)}")

        # 🧪 Manual Evaluation
        print("\n🧪 Manual Evaluation Prompts:")
        prompts = [
            ("Grammar/Coherence", "It was raining outside. The children..."),
            ("Creativity", "Sally was reading a book about space. Suddenly..."),
            ("Knowledge", "Lily was sad. She sat alone in the garden."),
            ("Reasoning", "Tim dropped his toy in the river. What happened next?"),
            ("Consistency", "The cat saw a dog. The dog barked. Then..."),
            ("Context Tracking", "Ben went to the beach. He built a sandcastle. Later...")
        ]
        for label, prompt in prompts:
            print(f"\n[{label}] Prompt: {prompt}")
            print("Generated:", self._generate(prompt, 50, 0.9, 40))

    def _generate(self, prompt, max_new_tokens, temperature, top_k):
        self.model.eval()
        self.model.clear_kv_cache()
        with torch.no_grad():
            prompt_ids = self.tokenizer.encode(prompt)
            max_id = max(prompt_ids)
            if max_id >= self.model.config.vocab_size:
                raise ValueError(f"Token ID {max_id} exceeds model vocab size {self.model.config.vocab_size}")
            input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            tokens = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
            tokens = tokens.squeeze(0).tolist()
        return self.tokenizer.decode(tokens)

# ++++++ Main Run ++++++ ───────────────────────────────────────

if __name__ == "__main__":
    # Training Config
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    EPOCHS = 6
    BATCH_SIZE = 64
    ACCUM_STEPS = 1
    
    PRINT_EVERY = 2
    BLOCK_SIZE = 256
    TRAINING = True
    DTYPE = get_dtype_for_device(DEVICE)
    CHECKPOINT_DIR = "checkpoints"
    RESUME_MODE = "resume"  # "resume" or "scratch"  
    USE_AMP = DTYPE == torch.bfloat16   #safer to use bf16 for amp
    WARMUP_ITERS = 0.05

    # Model Config
    EMBED_DIM = 256
    N_HEADS = 4   
    N_LAYERS = 4
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.1
    BETAS = (0.9, 0.95)
    WINDOW_SIZE = BLOCK_SIZE

    TRAIN_LEN = 1000
    VAL_LEN = 100

    #tokenizer
    TOKENIZER_PATH = "tokenizer.json"

    
    # Tokenizer
    tokenizer = TokenizerWrapper(
        path=TOKENIZER_PATH,
    )

    # Dataset
    dataset = BinDataset(
        train_path="data/train.bin",
        val_path="data/val.bin",
        block_size=BLOCK_SIZE,
        device=DEVICE,
        max_train_len=TRAIN_LEN,
        max_val_len=VAL_LEN,
        meta_path="data/meta.pkl"
    )
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"device: {DEVICE}")
    print(f"DataTYPE: {DTYPE}")
    print(f"Tokenizer vocab size: {actual_vocab_size}")
    # Model & Optimizer
    config = GPTConfig(
        vocab_size=actual_vocab_size,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        block_size=WINDOW_SIZE,
        training=TRAINING,
        device=DEVICE,
        dtype=DTYPE,
        norm_type='rms',
        use_rope=True
    )
    model = GPT(config)
    print(f"Model created: {model}")
    print(f"Embedding table shape: {model.transformer['wte'].weight.shape}")
 #   model = model.to(dtype=DTYPE, device=DEVICE)
    model = model.to(dtype=DTYPE, device=DEVICE)

    test_prompt = "Once upon a time"
    print(f"tokenizer.encode(test_prompt): {tokenizer.encode(test_prompt)}")
    print(f"tokenizer.decode(tokenizer.encode(test_prompt)): {tokenizer.decode(tokenizer.encode(test_prompt))}")

    # Trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=DEVICE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
        grad_accum_steps=ACCUM_STEPS,
        grad_clip=1.0,
        checkpoint_dir=CHECKPOINT_DIR,
        resume_mode=RESUME_MODE,
        use_amp=USE_AMP,
        warmup_iters=WARMUP_ITERS
    )
    trainer.train()
