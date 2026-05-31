#!/usr/bin/env python3
"""
Тестування обученої моделі на українській мові
"""
import os
import sys
import torch
import pickle
from pathlib import Path
from tokenizers import Tokenizer

# Ensure 'src' is findable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPT, GPTConfig

# Параметри
checkpoint_path = "out/ckpt.pt"
tokenizer_path = "data/processed/tokenizer.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("ТЕСТУВАННЯ МОДЕЛІ НА УКРАЇНСЬКІЙ МОВІ")
print("=" * 70)

# Завантажимо чекпоінт
print("\n📥 Завантаження моделі...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model_args = checkpoint['model_args']
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

print(f"✅ Модель завантажена")
print(f"   Параметрів: {model.get_num_params():,}")
print(f"   Ітерацій: {checkpoint['iter_num']}")
print(f"   Val loss: {checkpoint['best_val_loss']:.4f}")

# Завантажимо токенайзер
print("\n🔤 Завантаження токенайзера...")
tokenizer = Tokenizer.from_file(tokenizer_path)
print(f"✅ Токенайзер завантажений (vocab_size={tokenizer.get_vocab_size()})")

# Функція для генерації тексту
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Генерує текст на основі промпту"""
    # Кодуємо промпт
    encoded = tokenizer.encode(prompt)
    x = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Генеруємо
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Обрізаємо якщо занадто довгий
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
    
    # Декодуємо
    generated_ids = x[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

# Тестові промпти
prompts = [
    "Привіт",
    "Що буде в майбутньому ?",
    "Придумай розповідь про котів",
    "Вулиці міста",
]

print("\n" + "=" * 70)
print("ГЕНЕРАЦІЯ ТЕКСТУ")
print("=" * 70)

for i, prompt in enumerate(prompts, 1):
    print(f"\n📝 Промпт {i}: \"{prompt}\"")
    print("-" * 70)
    generated = generate(prompt, max_new_tokens=80, temperature=0.7)
    print(f"Результат:\n{generated}")
    print()

print("=" * 70)
print("✅ ТЕСТУВАННЯ ЗАВЕРШЕНО!")
print("=" * 70)
