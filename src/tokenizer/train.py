from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

# path to your merged and cleaned text file
corpus_path = "data/processed/dataset_train_clean.txt"

# Output directory for tokenizer files
output_dir = "data/processed"

Path(output_dir).mkdir(exist_ok=True)

# Initialize tokenizer with ByteLevel pre-tokenizer (works well with Cyrillic)
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

# Trainer with special tokens for Ukrainian
# Оптимізовано для української мови: 15K vocab замість 50K
trainer = BpeTrainer(
    vocab_size=15000,  # Оптимально для української мови
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]
)

print("Training tokenizer for Ukrainian Cyrillic...")
print("Vocab size: 15000 (optimized for Ukrainian)")
tokenizer.train(files=[corpus_path], trainer=trainer)

# save tokenizer files
tokenizer.save(f"{output_dir}/tokenizer.json")

# Also save in the old format for compatibility
tokenizer.model.save(output_dir)

print(f"Tokenizer saved to: {output_dir}")
print(f"Final vocab size: {tokenizer.get_vocab_size()}")
