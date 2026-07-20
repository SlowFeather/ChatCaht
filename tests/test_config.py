from __future__ import annotations

from pathlib import Path

import pytest

from chatcaht.config import load_config


def test_load_example_config() -> None:
    cfg = load_config(Path("configs/config.example.yaml"))
    assert cfg.openai.model == "qwen/qwen3.5-9b"
    assert cfg.wake.url == "ws://127.0.0.1:8766/v1/wake/ws"
    assert cfg.duplex.allow_barge_in is True
    assert cfg.services.audio_startup_timeout_sec == 30
    assert cfg.services.wake_startup_timeout_sec == 30
    assert cfg.services.stt_startup_timeout_sec == 30
    assert cfg.services.tts_startup_timeout_sec == 600
    assert cfg.services.llm_startup_timeout_sec == 60


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


def test_legacy_audio_runtime_config_path_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "legacy-audio.yaml"
    path.write_text(
        "audio:\n"
        "  aec_tail_ms: 275\n"
        "services:\n"
        "  audio_runtime_config: configs/old-audio.json\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.audio.aec_tail_ms == 275
    assert not hasattr(cfg.services, "audio_runtime_config")


def test_legacy_service_startup_timeout_applies_to_all_services(tmp_path: Path) -> None:
    path = tmp_path / "legacy-timeout.yaml"
    path.write_text("services:\n  startup_timeout_sec: 42\n", encoding="utf-8")

    cfg = load_config(path)

    assert cfg.services.audio_startup_timeout_sec == 42
    assert cfg.services.wake_startup_timeout_sec == 42
    assert cfg.services.stt_startup_timeout_sec == 42
    assert cfg.services.tts_startup_timeout_sec == 42
    assert cfg.services.llm_startup_timeout_sec == 42


def test_invalid_start_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("duplex:\n  start_mode: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start_mode"):
        load_config(path)
