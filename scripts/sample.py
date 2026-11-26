#!/usr/bin/env python3
"""Inference & Text Generation with Trained Ukrainian Language Model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# Ensure 'src' is findable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model.model import GPT, GPTConfig

DEFAULT_PROMPTS = [
    "Привіт",
    "Що буде в майбутньому ?",
    "Придумай розповідь про котів",
    "Вулиці міста",
]


def load_model(checkpoint_path: Path, device: str) -> tuple[GPT, dict]:
    """Load a trained GPT model from a checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    """Generate text continuation from a prompt using autoregressive top-k sampling."""
    encoded = tokenizer.encode(prompt)
    x = torch.tensor(encoded.ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = (
                x
                if x.size(1) <= model.config.block_size
                else x[:, -model.config.block_size:]
            )
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)

    return tokenizer.decode(x[0].tolist())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ukrainian text from a trained nanoGPT checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/ckpt.pt"),
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/processed/tokenizer.json"),
        help="Path to BPE tokenizer JSON",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to generate from (runs default prompts if omitted)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering value (0 disables top-k)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cpu or cuda)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive prompt loop",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.tokenizer.exists():
        print(f"Error: Tokenizer not found: {args.tokenizer}", file=sys.stderr)
        return 1

    try:
        model, checkpoint = load_model(args.checkpoint, args.device)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    top_k = args.top_k if args.top_k > 0 else None

    print(
        f"Loaded model ({model.get_num_params():,} params, "
        f"iter={checkpoint.get('iter_num', '?')}, "
        f"val_loss={checkpoint.get('best_val_loss', float('nan')):.4f})"
    )

    if args.interactive:
        print("Interactive mode (Ctrl+C or empty line to exit):")
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                break
            out = generate(
                model,
                tokenizer,
                prompt,
                args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
            print(out)
        return 0

    prompts = [args.prompt] if args.prompt is not None else DEFAULT_PROMPTS
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] Prompt: {prompt!r}")
        print("-" * 60)
        print(
            generate(
                model,
                tokenizer,
                prompt,
                args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
