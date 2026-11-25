#!/bin/bash
# End-to-end training pipeline for Ukrainian Historical LLM

set -e

echo "=========================================="
echo "TRAINING GPT FOR UKRAINIAN LANGUAGE"
echo "=========================================="

# Step 1: Check architecture optimizations
echo ""
echo "Step 1: Validating architecture optimizations..."
python scripts/check_optimization.py

# Step 2: Train custom BPE tokenizer
echo ""
echo "Step 2: Training custom BPE tokenizer (15K vocab)..."
python src/tokenizer/train.py

# Step 3: Dataset preparation and tokenization
echo ""
echo "Step 3: Encoding dataset into binary tokens..."
python scripts/prepare_dataset.py

# Step 4: Model pretraining
echo ""
echo "Step 4: Training GPT model on GPU..."
python scripts/train.py

echo ""
echo "=========================================="
echo "TRAINING PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Checkpoints and logs saved to: out/"
echo "Best model checkpoint: out/ckpt.pt"
