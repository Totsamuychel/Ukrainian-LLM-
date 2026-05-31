from tokenizers import Tokenizer
from pathlib import Path
import numpy as np
import pickle

# The paths
tokenizer_path = "data/processed/tokenizer.json"
corpus_path = "data/processed/dataset_train_clean.txt"
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(tokenizer_path)

# Read corpus
print("Reading corpus...")
with open(corpus_path, "r", encoding="utf-8") as f:
    data = f.read()

print(f"Corpus size: {len(data) / 1e6:.2f} MB")
print("Encoding text...")
encoded = tokenizer.encode(data)
ids = np.array(encoded.ids, dtype=np.uint16)

print(f"Total tokens: {len(ids):,}")

# Split into train/val (90/10)
split = int(0.9 * len(ids))
train_ids, val_ids = ids[:split], ids[split:]

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")

# Save as .bin files
train_ids.tofile(output_dir / "train.bin")
val_ids.tofile(output_dir / "val.bin")

# Save metadata
meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "tokenizer_path": tokenizer_path,
    "corpus_size_mb": len(data) / 1e6,
    "total_tokens": len(ids),
}
with open(output_dir / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"Saved train.bin, val.bin and meta.pkl in: {output_dir}")
