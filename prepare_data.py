import os
import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, normalizers, pre_tokenizers, Tokenizer
import pickle

# ─────────────────────────────────────────────────────────────
DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_PATH = "tokenizer.json"
VOCAB_SIZE = 8000
OUTPUT_DIR = "data/"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
# ─────────────────────────────────────────────────────────────

def train_tokenizer():
    print("📚 Loading dataset to train tokenizer...")
    dataset = load_dataset(DATASET_NAME, split=TRAIN_SPLIT, streaming=True)

    print("🔤 Training tokenizer on full dataset...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer._tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    tokenizer.train_from_iterator(
        (ex["text"] for ex in dataset),
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )

    tokenizer.save(TOKENIZER_PATH)
    print(f"✅ Tokenizer saved to {TOKENIZER_PATH}")

def load_tokenizer(path):
    return Tokenizer.from_file(path)

def encode_and_save(split_name, tokenizer, out_path):
    print(f"✍️ Tokenizing and saving {split_name} split...")
    dataset = load_dataset(DATASET_NAME, split=split_name, streaming=True)
    all_ids = []
    bos_positions = []

    bos_id = tokenizer.encode("<bos>").ids[0]
    eos_id = tokenizer.encode("<eos>").ids[0]

    for ex in dataset:
        encoded = tokenizer.encode(f"<bos> {ex['text']} <eos>").ids
        bos_positions.append(len(all_ids))  # record where this <bos> starts
        all_ids.extend(encoded)

    arr = np.array(all_ids, dtype=np.uint16)
    print(f"💾 Writing {len(arr)} tokens to {out_path}")
    arr.tofile(out_path)

    return bos_positions

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Train tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        train_tokenizer()
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # Encode and save splits
    train_bos_positions = encode_and_save(TRAIN_SPLIT, tokenizer, os.path.join(OUTPUT_DIR, "train.bin"))
    val_bos_positions   = encode_and_save(VAL_SPLIT, tokenizer, os.path.join(OUTPUT_DIR, "val.bin"))

    # Save meta
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_id": tokenizer.encode("<bos>").ids[0],
        "eos_id": tokenizer.encode("<eos>").ids[0],
        "bos_positions": {
            "train": train_bos_positions,
            "val": val_bos_positions
        }
    }

    with open(os.path.join(OUTPUT_DIR, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("✅ All done. Metadata saved to meta.pkl.")

if __name__ == "__main__":
    main()
