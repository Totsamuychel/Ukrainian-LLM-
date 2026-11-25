#!/usr/bin/env python3
"""
Hardware & Architecture Optimization Benchmark for Ukrainian Language Model
"""
import os
import sys

# Ensure 'src' is findable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPTConfig, GPT

print("=" * 70)
print("ARCHITECTURE OPTIMIZATION & VRAM BENCHMARK")
print("=" * 70)

# 1. Baseline Configuration (Standard GPT-2 Small for English)
print("\n📊 BASELINE CONFIGURATION (GPT-2 Small, English standard):")
print("-" * 70)
original_config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)
original_model = GPT(original_config)
original_params = original_model.get_num_params()
print(f"Block size: {original_config.block_size}")
print(f"Vocab size: {original_config.vocab_size}")
print(f"Layers: {original_config.n_layer}")
print(f"Heads: {original_config.n_head}")
print(f"Embedding dim: {original_config.n_embd}")
print(f"Dropout: {original_config.dropout}")
print(f"Bias: {original_config.bias}")
print(f"Total parameters: {original_params:,}")

# 2. Optimized Configuration (Targeted for historical Ukrainian corpus)
print("\n🇺🇦 OPTIMIZED CONFIGURATION (Ukrainian Custom BPE):")
print("-" * 70)
ukrainian_config = GPTConfig(
    block_size=512,
    vocab_size=15360,
    n_layer=10,
    n_head=10,
    n_embd=640,
    dropout=0.1,
    bias=False
)
ukrainian_model = GPT(ukrainian_config)
ukrainian_params = ukrainian_model.get_num_params()
print(f"Block size: {ukrainian_config.block_size}")
print(f"Vocab size: {ukrainian_config.vocab_size}")
print(f"Layers: {ukrainian_config.n_layer}")
print(f"Heads: {ukrainian_config.n_head}")
print(f"Embedding dim: {ukrainian_config.n_embd}")
print(f"Dropout: {ukrainian_config.dropout}")
print(f"Bias: {ukrainian_config.bias}")
print(f"Total parameters: {ukrainian_params:,}")

# Comparison
print("\n📈 ARCHITECTURAL COMPARISON:")
print("-" * 70)
reduction = (1 - ukrainian_params / original_params) * 100
print(f"Parameter reduction: {reduction:.1f}%")
print(f"Baseline parameters: {original_params:,}")
print(f"Optimized parameters: {ukrainian_params:,}")
print(f"Parameters saved: {original_params - ukrainian_params:,}")

# VRAM Estimation
print("\n💾 ESTIMATED VRAM CONSUMPTION (batch_size=32):")
print("-" * 70)

def estimate_vram(model, batch_size: int, block_size: int):
    """Estimates memory footprint: weights (FP32), activations, and Adam optimizer states."""
    model_vram = model.get_num_params() * 4 / 1e9  # float32
    activation_vram = batch_size * block_size * model.config.n_embd * 4 / 1e9
    optimizer_vram = model.get_num_params() * 8 / 1e9  # Adam states
    total = model_vram + activation_vram + optimizer_vram
    return model_vram, activation_vram, optimizer_vram, total

batch_size = 32
orig_model_vram, orig_act_vram, orig_opt_vram, orig_total = estimate_vram(
    original_model, batch_size, original_config.block_size
)
ukr_model_vram, ukr_act_vram, ukr_opt_vram, ukr_total = estimate_vram(
    ukrainian_model, batch_size, ukrainian_config.block_size
)

print(f"Baseline configuration:")
print(f"  Model weights: {orig_model_vram:.2f} GB")
print(f"  Activations: {orig_act_vram:.2f} GB")
print(f"  Optimizer states: {orig_opt_vram:.2f} GB")
print(f"  Total VRAM estimate: {orig_total:.2f} GB")

print(f"\nOptimized configuration:")
print(f"  Model weights: {ukr_model_vram:.2f} GB")
print(f"  Activations: {ukr_act_vram:.2f} GB")
print(f"  Optimizer states: {ukr_opt_vram:.2f} GB")
print(f"  Total VRAM estimate: {ukr_total:.2f} GB")

print(f"\nVRAM footprint reduction: {(1 - ukr_total / orig_total) * 100:.1f}%")
print(f"Hardware compatibility:")
print(f"  RTX 3090 (24GB VRAM): {'✅ Fits comfortably' if ukr_total < 24 else '❌ OOM risk'}")
print(f"  RTX 4060 (8GB VRAM):  {'✅ Fits' if ukr_total < 8 else '⚠️ Requires batch_size=16'}")

print("\n💡 KEY ARCHITECTURAL TAKEAWAYS:")
print("-" * 70)
print("1. Vocab size 15K eliminates rare-token bloat and fits morphological structure.")
print("2. Context window of 512 preserves narrative coherence while cutting activation memory by 50%.")
print("3. Dropout 0.1 prevents overfitting on small historical literature corpora.")
print("4. Removing biases (bias=False) stabilizes training and increases throughput.")
print("=" * 70)
