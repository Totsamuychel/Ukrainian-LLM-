#!/usr/bin/env python3
"""
Inference & Text Generation with Trained Ukrainian Language Model
"""
import os
import sys
import torch
from tokenizers import Tokenizer

# Ensure 'src' is findable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPT, GPTConfig

# Configuration
checkpoint_path = "out/ckpt.pt"
tokenizer_path = "data/processed/tokenizer.json"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("UKRAINIAN LANGUAGE MODEL INFERENCE & EVALUATION")
print("=" * 70)

# Load model checkpoint
print("\n📥 Loading checkpoint...")
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

print(f"✅ Model successfully loaded:")
print(f"   Parameters: {model.get_num_params():,}")
print(f"   Trained Iterations: {checkpoint['iter_num']}")
print(f"   Best Val Loss: {checkpoint['best_val_loss']:.4f}")

# Load custom BPE tokenizer
print("\n🔤 Loading tokenizer...")
tokenizer = Tokenizer.from_file(tokenizer_path)
print(f"✅ Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")

def generate(prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
    """Generates text continuation from a prompt using autoregressive top-k sampling."""
    encoded = tokenizer.encode(prompt)
    x = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if sequence exceeds block size
            x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
    
    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)

# Test prompts
prompts = [
    "Привіт",
    "Що буде в майбутньому ?",
    "Придумай розповідь про котів",
    "Вулиці міста",
]

print("\n" + "=" * 70)
print("TEXT GENERATION SAMPLING")
print("=" * 70)

for i, prompt in enumerate(prompts, 1):
    print(f"\n📝 Prompt {i}: \"{prompt}\"")
    print("-" * 70)
    generated = generate(prompt, max_new_tokens=80, temperature=0.7)
    print(f"Output:\n{generated}")
    print()

print("=" * 70)
print("✅ INFERENCE COMPLETE")
print("=" * 70)
