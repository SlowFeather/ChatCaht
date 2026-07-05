from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .adapters.llm import LollamaChatClient
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .config import Config

logger = logging.getLogger(__name__)

ServiceName = Literal["wake", "stt", "tts", "llm"]


@dataclass(slots=True)
class ManagedService:
    name: ServiceName
    cwd: Path
    command: list[str]
    pid_file: Path
    log_file: Path


@dataclass(slots=True)
class ServiceStatus:
    name: ServiceName
    running: bool
    pid: int | None
    ok: bool
    detail: str
    log_file: Path


class ServiceManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        run_dir = Path(cfg.paths.artifacts_dir) / "run"
        log_dir = Path(cfg.paths.logs_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._services = {
            item.name: item
            for item in (
                ManagedService(
                    name="wake",
                    cwd=_resolve(cfg.services.wakeup_dir),
                    command=[
                        cfg.services.uv_executable,
                        "run",
                        "wakeup",
                        "serve",
                        "--config",
                        cfg.services.wakeup_config,
                        "--listen",
                    ],
                    pid_file=run_dir / "wakeup.pid.json",
                    log_file=log_dir / "wakeup.service.log",
                ),
                ManagedService(
                    name="stt",
                    cwd=_resolve(cfg.services.sptext_dir),
                    command=[
                        cfg.services.uv_executable,
                        "run",
                        "sptext",
                        "serve",
                        "--config",
                        cfg.services.sptext_config,
                        "--no-listen",
                    ],
                    pid_file=run_dir / "sptext.pid.json",
                    log_file=log_dir / "sptext.service.log",
                ),
                ManagedService(
                    name="tts",
                    cwd=_resolve(cfg.services.gvoice_dir),
                    command=[
                        cfg.services.uv_executable,
                        "run",
                        "gvoice",
                        "--config",
                        cfg.services.gvoice_config,
                        "serve",
                    ],
                    pid_file=run_dir / "gvoice.pid.json",
                    log_file=log_dir / "gvoice.service.log",
                ),
                ManagedService(
                    name="llm",
                    cwd=_resolve(cfg.services.lollama_dir),
                    command=[
                        cfg.services.uv_executable,
                        "run",
                        "lollama",
                        "--config",
                        cfg.services.lollama_config,
                        "serve",
                    ],
                    pid_file=run_dir / "lollama.pid.json",
                    log_file=log_dir / "lollama.service.log",
                ),
            )
        }

    def default_names(self) -> list[ServiceName]:
        """默认托管的服务集合：llm 只在 provider=lollama 时纳入。"""
        names: list[ServiceName] = ["wake", "stt", "tts"]
        if self.cfg.llm.provider == "lollama":
            names.append("llm")
        return names

    async def start(self, names: list[ServiceName] | None = None, *, wait: bool = True) -> list[ServiceStatus]:
        names = names or self.default_names()
        for name in names:
            service = self._services[name]
            status = await self.status_one(name)
            if status.running:
                logger.info("service %s already running pid=%s", name, status.pid)
                continue
            self._start_process(service)
        if wait:
            deadline = time.monotonic() + self.cfg.services.startup_timeout_sec
            while time.monotonic() < deadline:
                statuses = await self.status(names)
                if all(s.ok for s in statuses):
                    logger.info("all requested services healthy: %s", ", ".join(names))
                    return statuses
                await asyncio.sleep(0.5)
            logger.warning(
                "services not all healthy within %.0fs: %s",
                self.cfg.services.startup_timeout_sec,
                ", ".join(f"{s.name}={s.detail}" for s in await self.status(names) if not s.ok),
            )
        return await self.status(names)

    async def stop(self, names: list[ServiceName] | None = None) -> list[ServiceStatus]:
        names = names or self.default_names()
        for name in names:
            if name == "wake":
                await _ignore_errors(create_wake_client(self.cfg.wake, timeout=2.0).stop())
                await _ignore_errors(_shutdown_wake(self.cfg))
            elif name == "stt":
                await _ignore_errors(_shutdown_stt(self.cfg))
            elif name == "llm":
                await _ignore_errors(_shutdown_llm(self.cfg))
            elif name == "tts":
                pass
        await asyncio.sleep(0.5)
        for name in names:
            service = self._services[name]
            pid = _read_pid(service.pid_file)
            if pid and _is_pid_running(pid):
                logger.info("terminating service %s pid=%d", name, pid)
                _terminate_pid(pid)
            else:
                logger.info("service %s already stopped", name)
            _remove_pid_file(service.pid_file)
        return await self.status(names)

    async def status(self, names: list[ServiceName] | None = None) -> list[ServiceStatus]:
        names = names or self.default_names()
        return [await self.status_one(name) for name in names]

    async def status_one(self, name: ServiceName) -> ServiceStatus:
        service = self._services[name]
        pid = _read_pid(service.pid_file)
        running = bool(pid and _is_pid_running(pid))
        ok, detail = await self._health(name)
        return ServiceStatus(name=name, running=running, pid=pid, ok=ok, detail=detail, log_file=service.log_file)

    async def _health(self, name: ServiceName) -> tuple[bool, str]:
        timeout = self.cfg.runtime.health_timeout_sec
        if name == "wake":
            return await create_wake_client(self.cfg.wake, timeout=timeout).health()
        if name == "stt":
            return await create_stt_client(self.cfg.stt, timeout=timeout).health()
        if name == "llm":
            return await LollamaChatClient(self.cfg.lollama, timeout=timeout).health()
        return await create_tts_client(self.cfg.tts, timeout=timeout).health()

    def _start_process(self, service: ManagedService) -> None:
        if not service.cwd.exists():
            raise FileNotFoundError(f"{service.name} project directory not found: {service.cwd}")
        logger.info(
            "starting service %s: %s (cwd=%s, log=%s)",
            service.name,
            " ".join(service.command),
            service.cwd,
            service.log_file,
        )
        service.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = service.log_file.open("ab")
        creationflags = 0
        kwargs = {}
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                service.command,
                cwd=service.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                **kwargs,
            )
        finally:
            log.close()
        logger.info("service %s started pid=%d", service.name, proc.pid)
        _write_pid(service.pid_file, proc.pid, service.command, service.cwd)


async def _shutdown_wake(cfg: Config) -> None:
    import websockets

    async with websockets.connect(cfg.wake.url, open_timeout=2.0, max_size=None) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "shutdown"}))
        await asyncio.wait_for(ws.recv(), timeout=2.0)


async def _shutdown_stt(cfg: Config) -> None:
    import websockets

    async with websockets.connect(cfg.stt.url, open_timeout=2.0, max_size=None) as ws:
        await ws.send(json.dumps({"type": "shutdown"}))
        await asyncio.wait_for(ws.recv(), timeout=2.0)


async def _shutdown_llm(cfg: Config) -> None:
    import websockets

    async with websockets.connect(cfg.lollama.url, open_timeout=2.0, max_size=None) as ws:
        await ws.send(json.dumps({"type": "shutdown"}))
        await asyncio.wait_for(ws.recv(), timeout=2.0)


async def _ignore_errors(awaitable) -> None:
    try:
        await awaitable
    except Exception:
        return None


def _resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _write_pid(path: Path, pid: int, command: list[str], cwd: Path) -> None:
    path.write_text(
        json.dumps({"pid": pid, "command": command, "cwd": str(cwd), "created_at": time.time()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_pid(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data["pid"])
    except Exception:
        return None


def _remove_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_pid_running(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return
