#!/usr/bin/env python3
"""
Pretraining GPT on 18th-19th Century Ukrainian Historical Texts

Optimized for:
- Ukrainian language morphology (custom BPE vocab_size=15K)
- Consumer & workstation GPUs (e.g. NVIDIA RTX 3090, 24GB VRAM)
- Curated ~118MB historical literature dataset
- Mixed precision (bfloat16/float16) and gradient accumulation

Usage:
    python scripts/train.py
    
With overrides:
    python scripts/train.py --batch_size=32 --max_iters=30000
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import torch
torch._dynamo.config.suppress_errors = True

import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Ensure 'src' is findable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import GPTConfig, GPT

# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume'

# Logging
wandb_log = False
wandb_project = 'ukrainian_llm'
wandb_run_name = 'ukrainian_gpt_v1'

# Data
dataset = 'ukrainian'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 512

# Model architecture - optimized for Ukrainian BPE vocabulary
n_layer = 10
n_head = 10
n_embd = 640
dropout = 0.1
bias = False

# Optimizer
learning_rate = 4e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate decay schedule
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 30000
min_lr = 4e-5

# Distributed training (DDP)
backend = 'nccl'

# Compute device & precision
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# ============================================================================
# CONFIGURATION OVERRIDE
# ============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(PROJECT_ROOT, 'src', 'model', 'configurator.py')).read())
config = {k: globals()[k] for k in config_keys}

# ============================================================================
# INITIALIZATION
# ============================================================================

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ============================================================================
# DATA LOADING
# ============================================================================

data_dir = 'data/processed'

def get_batch(split: str):
    """Samples a random batch from memory-mapped binary tokens."""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"Loaded tokenizer metadata: vocab_size = {meta_vocab_size}")
else:
    print(f"ERROR: meta.pkl not found at {meta_path}. Run scripts/prepare_dataset.py first!")
    exit(1)

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

if init_from == 'scratch':
    print("Initializing model from scratch")
    if meta_vocab_size is None:
        print("Error: vocab_size could not be determined")
        exit(1)
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from checkpoint: {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("Compiling model via torch.compile...")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# ============================================================================
# LOSS ESTIMATION & LEARNING RATE SCHEDULE
# ============================================================================

@torch.no_grad()
def estimate_loss():
    """Estimates average loss across train and validation splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it: int) -> float:
    """Computes learning rate with linear warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*60)
print("TRAINING UKRAINIAN HISTORICAL GPT MODEL")
print("="*60)
print(f"Model parameters: {model_args}")
print(f"Dataset directory: {data_dir}")
print(f"Checkpoint output: {out_dir}")
print("="*60 + "\n")

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluation and checkpoint saving
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"Saving checkpoint to: {out_dir}/ckpt.pt")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    # Forward/backward pass with gradient accumulation
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"Step {iter_num}: loss {lossf:.4f}, step time {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
