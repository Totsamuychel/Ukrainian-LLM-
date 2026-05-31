#!/usr/bin/env python3
"""
Перевірка оптимізацій для української мови
"""
import os
import sys
import torch

# Ensure 'src' is findable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPTConfig, GPT

print("=" * 70)
print("ПЕРЕВІРКА ОПТИМІЗАЦІЙ ДЛЯ УКРАЇНСЬКОЇ МОВИ")
print("=" * 70)

# Оригінальна конфігурація (англійська)
print("\n📊 ОРИГІНАЛЬНА КОНФІГУРАЦІЯ (англійська):")
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
print(f"Всього параметрів: {original_params:,}")

# Оптимізована конфігурація (українська)
print("\n🇺🇦 ОПТИМІЗОВАНА КОНФІГУРАЦІЯ (українська):")
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
print(f"Всього параметрів: {ukrainian_params:,}")

# Порівняння
print("\n📈 ПОРІВНЯННЯ:")
print("-" * 70)
reduction = (1 - ukrainian_params / original_params) * 100
print(f"Зменшення параметрів: {reduction:.1f}%")
print(f"Оригінальна: {original_params:,}")
print(f"Оптимізована: {ukrainian_params:,}")
print(f"Економія: {original_params - ukrainian_params:,} параметрів")

# Оцінка VRAM
print("\n💾 ОЦІНКА VRAM (batch_size=32):")
print("-" * 70)

# Приблизна оцінка VRAM
def estimate_vram(model, batch_size, block_size):
    # Model weights
    model_vram = model.get_num_params() * 4 / 1e9  # float32
    
    # Activations (приблизно)
    activation_vram = batch_size * block_size * model.config.n_embd * 4 / 1e9
    
    # Optimizer states (Adam: 2x model size)
    optimizer_vram = model.get_num_params() * 8 / 1e9
    
    total = model_vram + activation_vram + optimizer_vram
    return model_vram, activation_vram, optimizer_vram, total

batch_size = 32
orig_model_vram, orig_act_vram, orig_opt_vram, orig_total = estimate_vram(
    original_model, batch_size, original_config.block_size
)
ukr_model_vram, ukr_act_vram, ukr_opt_vram, ukr_total = estimate_vram(
    ukrainian_model, batch_size, ukrainian_config.block_size
)

print(f"Оригінальна конфігурація:")
print(f"  Model weights: {orig_model_vram:.2f} GB")
print(f"  Activations: {orig_act_vram:.2f} GB")
print(f"  Optimizer: {orig_opt_vram:.2f} GB")
print(f"  Всього: {orig_total:.2f} GB")

print(f"\nОптимізована конфігурація:")
print(f"  Model weights: {ukr_model_vram:.2f} GB")
print(f"  Activations: {ukr_act_vram:.2f} GB")
print(f"  Optimizer: {ukr_opt_vram:.2f} GB")
print(f"  Всього: {ukr_total:.2f} GB")

print(f"\nЕкономія VRAM: {(1 - ukr_total / orig_total) * 100:.1f}%")
print(f"RTX 3090 (24GB): {'✅ OK' if ukr_total < 24 else '❌ OOM'}")
print(f"RTX 4060 (8GB): {'✅ OK' if ukr_total < 8 else '❌ OOM'}")

# Рекомендації
print("\n✅ РЕКОМЕНДАЦІЇ:")
print("-" * 70)
print("1. Використовуй train_ukrainian.py для навчання")
print("2. Vocab size 15K оптимально для української мови")
print("3. Block size 512 економить пам'ять без втрати якості")
print("4. Dropout 0.1 запобігає перенавчанню на малому датасеті")
print("5. Batch size 32 оптимально для RTX 3090")

print("\n" + "=" * 70)
