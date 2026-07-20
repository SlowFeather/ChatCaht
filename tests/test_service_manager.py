from __future__ import annotations

import json
import os
from pathlib import Path

import chatcaht.service_manager as service_manager
from chatcaht.config import Config
from chatcaht.service_manager import (
    ManagedService,
    ProcessIdentity,
    ServiceManager,
    _child_environment,
    _owned_pid,
    _owned_process,
    _terminate_owned_process,
    _write_pid,
)


def test_managed_audio_modes_select_single_microphone_owner(tmp_path) -> None:
    cfg = Config()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    manager = ServiceManager(cfg)

    wake_command = manager._services["wake"].command
    stt_command = manager._services["stt"].command
    assert wake_command[wake_command.index("--input-mode") + 1] == "external_pcm"
    assert "--listen" not in wake_command
    assert "--no-listen" in stt_command
    assert manager.default_names()[0] == "audio"

    cfg.audio.mode = "disabled"
    manager = ServiceManager(cfg)
    wake_command = manager._services["wake"].command
    stt_command = manager._services["stt"].command
    assert wake_command[wake_command.index("--input-mode") + 1] == "microphone"
    assert "--listen" in wake_command
    assert "--no-listen" not in stt_command
    assert "audio" not in manager.default_names()


def test_service_startup_timeouts_are_individual(tmp_path) -> None:
    cfg = Config()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.services.audio_startup_timeout_sec = 11
    cfg.services.wake_startup_timeout_sec = 12
    cfg.services.stt_startup_timeout_sec = 13
    cfg.services.tts_startup_timeout_sec = 14
    cfg.services.llm_startup_timeout_sec = 15
    manager = ServiceManager(cfg)

    assert [manager.startup_timeout(name) for name in ("audio", "wake", "stt", "tts", "llm")] == [
        11,
        12,
        13,
        14,
        15,
    ]


def test_child_environment_drops_inherited_virtual_env(monkeypatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", r"D:\other-project\.venv")
    monkeypatch.setenv("CHATCAHT_TEST_ENV", "kept")

    env = _child_environment()

    assert "VIRTUAL_ENV" not in env
    assert env["CHATCAHT_TEST_ENV"] == "kept"


def test_managed_audio_config_is_generated_from_chatcaht_audio_settings(tmp_path) -> None:
    cfg = Config()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.audio.url = "ws://127.0.0.1:9910/custom/audio"
    cfg.audio.aec_tail_ms = 275
    cfg.audio.barge_in_min_speech_ms = 140
    cfg.audio.input_device = "USB microphone"

    manager = ServiceManager(cfg)

    generated = json.loads(manager.audio_runtime_config_path.read_text(encoding="utf-8"))
    assert generated["host"] == "127.0.0.1"
    assert generated["port"] == 9910
    assert generated["ws_path"] == "/custom/audio"
    assert generated["aec_tail_ms"] == 275
    assert generated["barge_in_min_speech_ms"] == 140
    assert generated["capture_stall_timeout_ms"] == 3000
    assert generated["input_device"] == "USB microphone"
    audio_command = manager._services["audio"].command
    assert Path(audio_command[0]).is_absolute()
    assert Path(audio_command[0]) == manager._services["audio"].cwd / cfg.services.audio_runtime_executable
    assert Path(audio_command[audio_command.index("--config") + 1]) == manager.audio_runtime_config_path


def test_pid_ownership_requires_matching_command_and_create_time(tmp_path) -> None:
    pid_file = tmp_path / "service.pid.json"
    service = ManagedService(
        name="stt",
        cwd=Path.cwd(),
        command=["uv", "run", "sptext"],
        pid_file=pid_file,
        log_file=tmp_path / "service.log",
    )
    _write_pid(pid_file, os.getpid(), service.command, service.cwd)
    assert _owned_pid(service) == os.getpid()

    changed = ManagedService(
        name="stt",
        cwd=tmp_path,
        command=["uv", "run", "other"],
        pid_file=pid_file,
        log_file=service.log_file,
    )
    assert _owned_pid(changed) is None

    data = json.loads(pid_file.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["observed_command_fingerprint"]
    original_cwd = data.pop("cwd")
    pid_file.write_text(json.dumps(data), encoding="utf-8")
    assert _owned_pid(service) is None

    data["cwd"] = original_cwd
    data["process_create_time"] += 1
    pid_file.write_text(json.dumps(data), encoding="utf-8")
    assert _owned_pid(service) is None


def test_pid_ownership_requires_observed_command_and_actual_cwd(tmp_path, monkeypatch) -> None:
    service = ManagedService(
        name="tts",
        cwd=tmp_path,
        command=["uv", "run", "gvoice", "serve"],
        pid_file=tmp_path / "service.pid.json",
        log_file=tmp_path / "service.log",
    )
    process = _FakeProcess(1234)
    identity = ProcessIdentity(process, process.pid, 100.0, tmp_path.resolve(), "observed-command")
    monkeypatch.setattr(service_manager, "_process_identity", lambda _pid: identity)
    _write_pid(service.pid_file, process.pid, service.command, service.cwd)
    assert _owned_process(service) == identity

    data = json.loads(service.pid_file.read_text(encoding="utf-8"))
    data["observed_command_fingerprint"] = "different"
    service.pid_file.write_text(json.dumps(data), encoding="utf-8")
    assert _owned_process(service) is None
    assert not _terminate_owned_process(service)
    assert not process.killed

    data["observed_command_fingerprint"] = identity.command_fingerprint
    service.pid_file.write_text(json.dumps(data), encoding="utf-8")
    moved = ProcessIdentity(process, process.pid, identity.create_time, tmp_path.parent.resolve(), identity.command_fingerprint)
    monkeypatch.setattr(service_manager, "_process_identity", lambda _pid: moved)
    assert _owned_process(service) is None
    assert not _terminate_owned_process(service)
    assert not process.killed


def test_legacy_pid_record_keeps_existing_identity_checks(tmp_path, monkeypatch) -> None:
    service = ManagedService(
        name="wake",
        cwd=tmp_path,
        command=["uv", "run", "wakeup"],
        pid_file=tmp_path / "service.pid.json",
        log_file=tmp_path / "service.log",
    )
    process = _FakeProcess(2345)
    identity = ProcessIdentity(process, process.pid, 200.0, tmp_path.resolve(), "initial")
    monkeypatch.setattr(service_manager, "_process_identity", lambda _pid: identity)
    _write_pid(service.pid_file, process.pid, service.command, service.cwd)
    data = json.loads(service.pid_file.read_text(encoding="utf-8"))
    data.pop("schema_version")
    data.pop("observed_command_fingerprint")
    service.pid_file.write_text(json.dumps(data), encoding="utf-8")

    changed_command = ProcessIdentity(process, process.pid, 200.0, tmp_path.resolve(), "changed")
    monkeypatch.setattr(service_manager, "_process_identity", lambda _pid: changed_command)
    assert _owned_process(service) == changed_command


def test_safe_termination_revalidates_identity(tmp_path, monkeypatch) -> None:
    service = ManagedService(
        name="llm",
        cwd=tmp_path,
        command=["uv", "run", "lollama", "serve"],
        pid_file=tmp_path / "service.pid.json",
        log_file=tmp_path / "service.log",
    )
    process = _FakeProcess(3456)
    identity = ProcessIdentity(process, process.pid, 300.0, tmp_path.resolve(), "observed")
    monkeypatch.setattr(service_manager, "_process_identity", lambda _pid: identity)
    _write_pid(service.pid_file, process.pid, service.command, service.cwd)
    assert _owned_process(service) == identity

    data = json.loads(service.pid_file.read_text(encoding="utf-8"))
    data["process_create_time"] += 1
    service.pid_file.write_text(json.dumps(data), encoding="utf-8")
    assert not _terminate_owned_process(service)
    assert not process.killed


class _FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.killed = False

    def children(self, recursive: bool = False):
        return []

    def kill(self) -> None:
        self.killed = True
