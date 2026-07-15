from __future__ import annotations

import json
import os

from chatcaht.service_manager import ManagedService, _owned_pid, _write_pid


def test_pid_ownership_requires_matching_command_and_create_time(tmp_path) -> None:
    pid_file = tmp_path / "service.pid.json"
    service = ManagedService(
        name="stt",
        cwd=tmp_path,
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
    data["process_create_time"] += 1
    pid_file.write_text(json.dumps(data), encoding="utf-8")
    assert _owned_pid(service) is None
