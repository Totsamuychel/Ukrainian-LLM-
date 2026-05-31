from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

# Load and read your merged corpus
corpus_path = Path("london_corpus_cleaned_merged.txt")

with open(corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenizer settings
vocab_size = 50000  # Increased for better Ukrainian Cyrillic coverage

output_dir = Path("tokenizer_london")
output_dir.mkdir(exist_ok=True)

# Initialize tokenizer with ByteLevel pre-tokenizer (works well with Cyrillic)
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

# Trainer with special tokens
trainer = BpeTrainer(
    vocab_size=vocab_size,
    min_frequency=2,
    special_tokens=["<|endoftext|>"]
)

# training
print("Training tokenizer for Ukrainian Cyrillic...")
tokenizer.train_from_iterator(
    [raw_text],
    trainer=trainer,
    length=1
)

# saves tokenizer files
tokenizer.save(str(output_dir / "tokenizer.json"))
tokenizer.model.save(str(output_dir))

print(f"Tokenizer saved to {output_dir}")
print(f"Vocab size: {tokenizer.get_vocab_size()}")
