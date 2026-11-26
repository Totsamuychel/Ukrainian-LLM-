"""Smoke tests for Ukrainian-LLM- model, configurator, and CLI scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gpt_forward_and_generate():
    torch = pytest.importorskip("torch")
    from src.model.model import GPT, GPTConfig

    cfg = GPTConfig(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    model = GPT(cfg)
    model.eval()

    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None and loss.item() > 0

    generated = model.generate(x[:, :3], max_new_tokens=4)
    assert generated.shape == (2, 7)


def test_sample_cli_help_and_missing_files():
    pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from scripts import sample

    with pytest.raises(SystemExit) as exc:
        sample.parse_args(["--help"])
    assert exc.value.code == 0

    rc = sample.main(["--tokenizer", "/nonexistent/tokenizer.json"])
    assert rc == 1


def test_prepare_dataset_cli_missing_files(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("tokenizers")
    from scripts import prepare_dataset

    rc = prepare_dataset.main(
        [
            "--corpus",
            str(tmp_path / "missing.txt"),
            "--tokenizer",
            str(tmp_path / "missing.json"),
        ]
    )
    assert rc == 1


def test_configurator_syntax():
    res = subprocess.run(
        [sys.executable, "-m", "py_compile", str(PROJECT_ROOT / "src" / "model" / "configurator.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
