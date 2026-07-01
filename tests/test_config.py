from __future__ import annotations

from pathlib import Path

import pytest

from chatcaht.config import load_config


def test_load_example_config() -> None:
    cfg = load_config(Path("configs/config.example.yaml"))
    assert cfg.lmstudio.model == "qwopus3.5-9b-coder"
    assert cfg.wake.url == "ws://127.0.0.1:8766/v1/wake/ws"
    assert cfg.duplex.allow_barge_in is True


def test_invalid_start_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("duplex:\n  start_mode: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start_mode"):
        load_config(path)
