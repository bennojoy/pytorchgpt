# PyTorchGPT - Language Model Training Framework

A complete, from-scratch implementation of a GPT-style language model training and inference framework built in PyTorch. This project demonstrates modern language model architecture with efficient training and generation capabilities.

## 🎯 Overview

PyTorchGPT is a production-ready language model framework that includes:

- **Modern GPT architecture** with Rotary Positional Embeddings (RoPE)
- **Byte-level BPE tokenizer** training and inference
- **Efficient data processing** pipeline with streaming support
- **Complete training framework** with checkpointing and validation
- **Fast inference** with KV caching and text generation
- **GPU optimization** with mixed precision training

## 🏗️ Architecture

### Model Components

The model follows the standard transformer architecture with modern improvements:

- **Multi-head Self-Attention** with scaled dot-product attention
- **Rotary Positional Embeddings (RoPE)** for better position encoding
- **RMSNorm/LayerNorm** options for normalization
- **Feedforward networks** with GELU activation
- **KV caching** for efficient autoregressive generation

### Key Features

- **Configurable architecture** (embedding dim, layers, heads, etc.)
- **Memory efficient** with streaming data processing
- **Production ready** with proper error handling and logging
- **GPU optimized** with CUDA support and mixed precision
- **Modular design** for easy experimentation

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pytorchgpt
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install datasets tokenizers tqdm numpy
   ```

## 🚀 Quick Start

### 1. Prepare Data

First, prepare the TinyStories dataset and train the tokenizer:

```bash
python prepare_data.py
```

This will:
- Download the TinyStories dataset
- Train a byte-level BPE tokenizer
- Convert text to binary token files for fast training
- Save metadata for training

### 2. Train the Model

```python
from model import GPTConfig, GPT, Trainer, BinDataset, TokenizerWrapper

# Configuration
config = GPTConfig(
    vocab_size=8000,      # From tokenizer
    block_size=256,       # Context length
    embed_dim=512,        # Embedding dimension
    n_heads=8,           # Number of attention heads
    n_layers=6,          # Number of transformer layers
    dropout=0.1,         # Dropout rate
    learning_rate=1e-4,  # Learning rate
    device='cuda'        # Use GPU
)

# Initialize model, dataset, and tokenizer
model = GPT(config)
dataset = BinDataset('data/train.bin', 'data/val.bin', 'data/meta.pkl', 
                    block_size=config.block_size, device=config.device)
tokenizer = TokenizerWrapper('tokenizer.json')

# Train
trainer = Trainer(
    model=model,
    dataset=dataset,
    tokenizer=tokenizer,
    device=config.device,
    epochs=10,
    batch_size=32,
    checkpoint_dir='checkpoints'
)

trainer.train()
```

### 3. Generate Text

```python
# Load trained model
model = GPT.load_from_checkpoint('checkpoints/ckpt_best.pt')

# Generate text
prompt = "Once upon a time"
generated = model.generate(
    token_ids=tokenizer.encode(prompt),
    max_new_tokens=100,
    temperature=0.8,
    top_k=40
)

print(tokenizer.decode(generated))
```

## 📁 Project Structure

```
pytorchgpt/
├── model.py              # Main model implementation
├── prepare_data.py       # Data preparation and tokenizer training
├── tokenizer.json        # Trained tokenizer
├── data/                 # Processed dataset files
│   ├── train.bin        # Training tokens
│   ├── val.bin          # Validation tokens
│   └── meta.pkl         # Dataset metadata
├── checkpoints/          # Model checkpoints
│   ├── ckpt_latest.pt   # Latest checkpoint
│   └── ckpt_best.pt     # Best validation checkpoint
└── README.md            # This file
```

## 🔧 Configuration

### Model Configuration

The `GPTConfig` class allows you to customize the model architecture:

```python
config = GPTConfig(
    vocab_size=8000,        # Vocabulary size
    block_size=256,         # Maximum sequence length
    embed_dim=512,          # Embedding dimension
    n_heads=8,             # Number of attention heads
    n_layers=6,            # Number of transformer layers
    dropout=0.1,           # Dropout rate
    learning_rate=1e-4,    # Learning rate
    weight_decay=0.1,      # Weight decay
    use_rope=True,         # Use Rotary Positional Embeddings
    norm_type='layernorm'  # 'layernorm' or 'rmsnorm'
)
```

### Training Configuration

The `Trainer` class provides extensive training options:

```python
trainer = Trainer(
    model=model,
    dataset=dataset,
    tokenizer=tokenizer,
    device='cuda',
    epochs=10,
    batch_size=32,
    print_every=100,        # Print loss every N steps
    grad_accum_steps=1,     # Gradient accumulation steps
    grad_clip=1.0,          # Gradient clipping
    checkpoint_dir='checkpoints',
    warmup_iters=0.1,       # Warmup iterations (fraction of total)
    resume_mode='scratch',  # 'scratch', 'latest', or 'best'
    use_amp=False          # Use automatic mixed precision
)
```

## 📊 Dataset

This project uses the **TinyStories dataset**, a collection of simple, short stories designed for language model training and evaluation. The dataset is automatically downloaded and processed during setup.

### Data Processing

The data processing pipeline:

1. **Downloads** TinyStories dataset using HuggingFace datasets
2. **Trains** a byte-level BPE tokenizer on the training split
3. **Tokenizes** all text with special tokens (`<bos>`, `<eos>`)
4. **Saves** tokens as binary files for fast loading during training
5. **Tracks** story boundaries for proper training

## 🎮 Usage Examples

### Training from Scratch

```python
from model import *

# Load configuration
config = GPTConfig(
    vocab_size=8000,
    block_size=256,
    embed_dim=512,
    n_heads=8,
    n_layers=6,
    device='cuda'
)

# Initialize components
model = GPT(config)
dataset = BinDataset('data/train.bin', 'data/val.bin', 'data/meta.pkl', 
                    block_size=config.block_size, device=config.device)
tokenizer = TokenizerWrapper('tokenizer.json')

# Train
trainer = Trainer(model, dataset, tokenizer, device=config.device, epochs=10)
trainer.train()
```

### Resuming Training

```python
# Resume from latest checkpoint
trainer = Trainer(model, dataset, tokenizer, device=config.device, 
                 resume_mode='latest')
trainer.train()

# Resume from best checkpoint
trainer = Trainer(model, dataset, tokenizer, device=config.device, 
                 resume_mode='best')
trainer.train()
```

### Text Generation

```python
# Load trained model
model = GPT.load_from_checkpoint('checkpoints/ckpt_best.pt')
model.eval()

# Generate text
prompts = [
    "Once upon a time",
    "The little cat",
    "In a magical forest"
]

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    generated = model.generate(
        token_ids=tokens,
        max_new_tokens=50,
        temperature=0.8,
        top_k=40
    )
    print(f"Prompt: {prompt}")
    print(f"Generated: {tokenizer.decode(generated)}")
    print("-" * 50)
```

### Interactive Generation

```python
model = GPT.load_from_checkpoint('checkpoints/ckpt_best.pt')
model.eval()

print("Interactive text generation (type 'quit' to exit):")
while True:
    prompt = input("\nEnter prompt: ")
    if prompt.lower() == 'quit':
        break
    
    tokens = tokenizer.encode(prompt)
    generated = model.generate(
        token_ids=tokens,
        max_new_tokens=100,
        temperature=0.9,
        top_k=40
    )
    print(f"Generated: {tokenizer.decode(generated)}")
```

## 🔍 Model Architecture Details

### Transformer Block

Each transformer block consists of:

1. **Multi-head Self-Attention** with RoPE
2. **Residual connection** and normalization
3. **Feedforward network** (4x expansion)
4. **Residual connection** and normalization

### Attention Mechanism

- **Scaled dot-product attention** with causal masking
- **Rotary Positional Embeddings** for better position encoding
- **KV caching** for efficient autoregressive generation
- **Multi-head attention** with configurable head count

### Training Optimizations

- **Gradient accumulation** for larger effective batch sizes
- **Learning rate warmup** and scheduling
- **Gradient clipping** for stability
- **Mixed precision training** (optional)
- **Checkpointing** for resuming training

## 📈 Performance Tips

### Training

1. **Use GPU** for significantly faster training
2. **Adjust batch size** based on your GPU memory
3. **Use gradient accumulation** for larger effective batch sizes
4. **Enable mixed precision** if your GPU supports it
5. **Monitor validation loss** to avoid overfitting

### Generation

1. **Use KV caching** for faster generation
2. **Adjust temperature** for creativity vs. coherence
3. **Use top-k sampling** for better text quality
4. **Clear cache** between different generation runs

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Slow training**: Ensure you're using GPU and check data loading
3. **Poor generation quality**: Try different temperature and top-k values
4. **Tokenizer errors**: Re-run `prepare_data.py` to regenerate tokenizer

### Debug Mode

Enable debug logging by setting environment variables:

```bash
export CUDA_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New features
- Documentation improvements

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **TinyStories dataset** by Ronen Eldan
- **HuggingFace** for datasets and tokenizers
- **PyTorch** team for the excellent framework
- **OpenAI** for the original GPT architecture

---

**Happy training! 🚀** 