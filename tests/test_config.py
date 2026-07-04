from __future__ import annotations

from pathlib import Path

import pytest

from chatcaht.config import load_config


def test_load_example_config() -> None:
    cfg = load_config(Path("configs/config.example.yaml"))
    assert cfg.openai.model == "qwen3.5:9b"
    assert cfg.wake.url == "ws://127.0.0.1:8766/v1/wake/ws"
    assert cfg.duplex.allow_barge_in is True


def test_load_legacy_lmstudio_config(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "lmstudio:\n"
        "  base_url: http://127.0.0.1:1234/v1\n"
        "  model: legacy-model\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.openai.model == "legacy-model"
    assert cfg.lmstudio is cfg.openai


def test_invalid_start_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("duplex:\n  start_mode: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start_mode"):
        load_config(path)
