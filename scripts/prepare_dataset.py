#!/usr/bin/env python3
"""Prepare tokenized binary datasets (train.bin, val.bin, meta.pkl) from a text corpus."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def prepare_dataset(
    corpus_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    val_split: float = 0.1,
) -> dict:
    """Encode a text corpus with a BPE tokenizer and write train/val binary shards."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be between 0 and 1, got {val_split}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print("Reading corpus...")
    data = corpus_path.read_text(encoding="utf-8")

    print(f"Corpus size: {len(data) / 1e6:.2f} MB")
    print("Encoding text...")
    encoded = tokenizer.encode(data)
    ids = np.array(encoded.ids, dtype=np.uint16)

    print(f"Total tokens: {len(ids):,}")

    split = int((1.0 - val_split) * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")

    train_ids.tofile(output_dir / "train.bin")
    val_ids.tofile(output_dir / "val.bin")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_path": str(tokenizer_path),
        "corpus_size_mb": len(data) / 1e6,
        "total_tokens": len(ids),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved train.bin, val.bin and meta.pkl in: {output_dir}")
    return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tokenize a cleaned Ukrainian corpus into train.bin and val.bin",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/dataset_train_clean.txt"),
        help="Path to the cleaned UTF-8 text corpus",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/processed/tokenizer.json"),
        help="Path to the trained BPE tokenizer JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for train.bin, val.bin, and meta.pkl",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of tokens reserved for validation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_dataset(
            corpus_path=args.corpus,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            val_split=args.val_split,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
