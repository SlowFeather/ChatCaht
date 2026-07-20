from __future__ import annotations

from pathlib import Path

import pytest

from chatcaht.config import Config
from chatcaht.models import ServiceState
from chatcaht.service_manager import ServiceStatus
from chatcaht.supervisor import Supervisor


class _FakeManager:
    def __init__(self, statuses: list[ServiceStatus]) -> None:
        self.statuses = statuses
        self.wait_ready_calls = 0
        self.start_calls: list[list[str] | None] = []
        self.stop_calls: list[list[str]] = []

    async def wait_ready(self, names=None):
        self.wait_ready_calls += 1
        return self.statuses

    async def status(self, names=None):
        if names is None:
            return self.statuses
        return [status for status in self.statuses if status.name in names]

    async def start(self, names=None, *, wait=True):
        self.start_calls.append(names)
        return self.statuses

    async def stop(self, names=None):
        self.stop_calls.append(names or [])
        return self.statuses


class _FakeSession:
    def __init__(self) -> None:
        self.stop_requested = False

    def request_stop(self) -> None:
        self.stop_requested = True


def _status(state: ServiceState) -> ServiceStatus:
    return ServiceStatus(
        name="tts",
        running=True,
        pid=123,
        ok=state is ServiceState.READY,
        detail=state.value,
        log_file=Path("tts.log"),
        state=state,
    )


def _supervisor(tmp_path: Path, *, manage_services: bool) -> Supervisor:
    cfg = Config()
    cfg.paths.artifacts_dir = str(tmp_path / "artifacts")
    cfg.paths.logs_dir = str(tmp_path / "logs")
    cfg.supervisor.manage_services = manage_services
    return Supervisor(cfg, audio_factory=lambda: None)


@pytest.mark.asyncio
async def test_external_services_still_must_reach_ready(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, manage_services=False)
    manager = _FakeManager([_status(ServiceState.STARTING)])
    supervisor.manager = manager

    with pytest.raises(RuntimeError, match="required services not READY"):
        await supervisor._ensure_services()

    assert manager.wait_ready_calls == 1
    assert manager.start_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [ServiceState.STARTING, ServiceState.DEGRADED])
async def test_monitor_does_not_restart_non_failed_services(tmp_path: Path, state: ServiceState) -> None:
    supervisor = _supervisor(tmp_path, manage_services=True)
    manager = _FakeManager([_status(state)])
    supervisor.manager = manager
    session = _FakeSession()
    supervisor.session = session
    sleeps = iter((False, True))

    async def fake_sleep(_delay: float) -> bool:
        return next(sleeps)

    supervisor._sleep_unless_stopped = fake_sleep
    await supervisor._monitor_services()

    assert session.stop_requested
    assert manager.stop_calls == []
    assert manager.start_calls == []


@pytest.mark.asyncio
async def test_monitor_restarts_only_failed_services(tmp_path: Path) -> None:
    supervisor = _supervisor(tmp_path, manage_services=True)
    manager = _FakeManager([_status(ServiceState.FAILED)])
    supervisor.manager = manager
    sleeps = iter((False, True))

    async def fake_sleep(_delay: float) -> bool:
        return next(sleeps)

    supervisor._sleep_unless_stopped = fake_sleep
    await supervisor._monitor_services()

    assert manager.stop_calls == [["tts"]]
    assert manager.start_calls == [["tts"]]
