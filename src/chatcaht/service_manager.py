from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import psutil

from .adapters.llm import LollamaChatClient
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .audio_runtime import AudioRuntimeClient
from .config import Config
from .models import ServiceProbe, ServiceState

logger = logging.getLogger(__name__)

ServiceName = Literal["audio", "wake", "stt", "tts", "llm"]


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
    state: ServiceState = ServiceState.FAILED


@dataclass(slots=True)
class ProcessIdentity:
    process: psutil.Process
    pid: int
    create_time: float
    cwd: Path
    command_fingerprint: str


class ServiceManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        run_dir = Path(cfg.paths.artifacts_dir) / "run"
        log_dir = Path(cfg.paths.logs_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.audio_runtime_config_path = (run_dir / "audio-runtime.generated.json").resolve()
        write_audio_runtime_config(cfg, self.audio_runtime_config_path)
        audio_runtime_dir = _resolve(cfg.services.audio_runtime_dir)
        self._services = {
            item.name: item
            for item in (
                ManagedService(
                    name="audio",
                    cwd=audio_runtime_dir,
                    command=[
                        str(_resolve_from(audio_runtime_dir, cfg.services.audio_runtime_executable)),
                        "--config",
                        str(self.audio_runtime_config_path),
                    ],
                    pid_file=run_dir / "audio-runtime.pid.json",
                    log_file=log_dir / "audio-runtime.service.log",
                ),
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
                        "--input-mode",
                        "external_pcm" if cfg.audio.mode == "unified_required" else "microphone",
                        *([] if cfg.audio.mode == "unified_required" else ["--listen"]),
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
                        *(["--no-listen"] if cfg.audio.mode == "unified_required" else []),
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
        names: list[ServiceName] = []
        if self.cfg.audio.mode == "unified_required":
            names.append("audio")
        names.extend(("wake", "stt", "tts"))
        if self.cfg.llm.provider == "lollama":
            names.append("llm")
        return names

    async def start(self, names: list[ServiceName] | None = None, *, wait: bool = True) -> list[ServiceStatus]:
        names = names or self.default_names()
        initial = await self.status(names)
        for name, status in zip(names, initial):
            service = self._services[name]
            if status.running:
                logger.info("service %s already running pid=%s", name, status.pid)
                continue
            if status.ok:
                logger.info("service %s is healthy but externally managed; leaving it untouched", name)
                continue
            unverified_pid = _unverified_live_pid(service)
            if unverified_pid is not None:
                raise RuntimeError(
                    f"refusing to start {name}: PID file points to live but unverified process "
                    f"pid={unverified_pid}; inspect it and remove {service.pid_file} manually"
                )
            _remove_pid_file(service.pid_file)
            self._start_process(service)
        if wait:
            statuses = await self.wait_ready(names)
            if all(status.ok for status in statuses):
                logger.info("all requested services READY: %s", ", ".join(names))
            else:
                logger.warning(
                    "services did not reach READY: %s",
                    ", ".join(f"{s.name}={s.state.value}:{s.detail}" for s in statuses if not s.ok),
                )
            return statuses
        return await self.status(names)

    async def wait_ready(self, names: list[ServiceName] | None = None) -> list[ServiceStatus]:
        names = names or self.default_names()
        return list(await asyncio.gather(*(self._wait_until_ready(name) for name in names)))

    async def _wait_until_ready(self, name: ServiceName) -> ServiceStatus:
        timeout = self.startup_timeout(name)
        deadline = time.monotonic() + timeout
        status = await self.status_one(name)
        while status.state is not ServiceState.READY and time.monotonic() < deadline:
            if status.state is ServiceState.FAILED and not status.running:
                break
            await asyncio.sleep(0.5)
            status = await self.status_one(name)
        if status.state is not ServiceState.READY:
            status.detail = f"startup timeout after {timeout:.0f}s: {status.detail}"
            if status.state is ServiceState.STARTING:
                status.state = ServiceState.FAILED
        return status

    def startup_timeout(self, name: ServiceName) -> float:
        return float(getattr(self.cfg.services, f"{name}_startup_timeout_sec"))

    async def stop(self, names: list[ServiceName] | None = None) -> list[ServiceStatus]:
        names = names or self.default_names()
        owned = {name: _owned_process(self._services[name]) for name in names}
        for name in names:
            if owned[name] is None:
                logger.info("service %s has no verified owned process; skipping shutdown", name)
                continue
            if name == "wake":
                await _ignore_errors(create_wake_client(self.cfg.wake, timeout=2.0).stop())
                await _ignore_errors(_shutdown_wake(self.cfg))
            elif name == "stt":
                await _ignore_errors(_shutdown_stt(self.cfg))
            elif name == "llm":
                await _ignore_errors(_shutdown_llm(self.cfg))
            elif name == "tts":
                pass
            elif name == "audio":
                pass
        await asyncio.sleep(0.5)
        for name in names:
            service = self._services[name]
            if owned[name] is None:
                if _unverified_live_pid(service) is not None:
                    logger.warning("leaving unverified PID record untouched: %s", service.pid_file)
                continue
            process = _owned_process(service)
            if process is not None:
                logger.info("terminating service %s pid=%d", name, process.pid)
                if not _terminate_owned_process(service):
                    logger.warning("service %s identity changed or termination failed; keeping PID record", name)
                    continue
            else:
                unverified_pid = _unverified_live_pid(service)
                if unverified_pid is not None:
                    logger.warning(
                        "service %s identity changed before termination pid=%d; keeping PID record",
                        name,
                        unverified_pid,
                    )
                    continue
                logger.info("service %s already stopped", name)
            _remove_pid_file(service.pid_file)
        return await self.status(names)

    async def status(self, names: list[ServiceName] | None = None) -> list[ServiceStatus]:
        names = names or self.default_names()
        return list(await asyncio.gather(*(self.status_one(name) for name in names)))

    async def status_one(self, name: ServiceName) -> ServiceStatus:
        service = self._services[name]
        process = _owned_process(service)
        pid = process.pid if process is not None else None
        running = process is not None
        probe = await self._probe(name)
        if (
            probe.state is ServiceState.FAILED
            and probe.raw is None
            and running
            and self._owned_process_age(service) < self.startup_timeout(name)
        ):
            probe = ServiceProbe(ServiceState.STARTING, probe.detail, probe.raw)
        if not running and probe.raw is None and probe.state is not ServiceState.READY:
            probe.state = ServiceState.FAILED
        return ServiceStatus(
            name=name,
            running=running,
            pid=pid,
            ok=probe.ready,
            detail=probe.detail,
            log_file=service.log_file,
            state=probe.state,
        )

    async def _probe(self, name: ServiceName) -> ServiceProbe:
        timeout = self.cfg.runtime.health_timeout_sec
        if name == "audio":
            return await AudioRuntimeClient(self.cfg.audio, timeout=timeout).probe()
        if name == "wake":
            return await create_wake_client(self.cfg.wake, timeout=timeout).probe()
        if name == "stt":
            return await create_stt_client(self.cfg.stt, timeout=timeout).probe()
        if name == "llm":
            return await LollamaChatClient(self.cfg.lollama, timeout=timeout).probe()
        return await create_tts_client(self.cfg.tts, timeout=timeout).probe()

    @staticmethod
    def _owned_process_age(service: ManagedService) -> float:
        record = _read_pid_record(service.pid_file) or {}
        try:
            return max(0.0, time.time() - float(record["created_at"]))
        except (KeyError, TypeError, ValueError):
            return float("inf")

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
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen(
                service.command,
                cwd=service.cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_child_environment(),
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        finally:
            log.close()
        logger.info("service %s started pid=%d", service.name, proc.pid)
        try:
            _write_pid(service.pid_file, proc.pid, service.command, service.cwd)
        except Exception:
            logger.exception("failed to capture process identity for %s pid=%d", service.name, proc.pid)
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=5)
            _remove_pid_file(service.pid_file)
            raise


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


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env


def _resolve_from(base: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def audio_runtime_config_payload(cfg: Config) -> dict:
    parsed = urlsplit(cfg.audio.url)
    if parsed.scheme != "ws" or not parsed.hostname or parsed.query or parsed.fragment or parsed.username:
        raise ValueError("managed AudioRuntime requires a plain ws:// URL without credentials, query, or fragment")
    return {
        "config_version": 1,
        "host": parsed.hostname,
        "port": parsed.port or 80,
        "ws_path": parsed.path or "/",
        "device_sample_rate": cfg.audio.device_sample_rate,
        "capture_sample_rate": cfg.audio.capture_sample_rate,
        "frame_ms": cfg.audio.frame_ms,
        "render_queue_ms": cfg.audio.render_queue_ms,
        "capture_queue_ms": cfg.audio.capture_queue_ms,
        "capture_stall_timeout_ms": cfg.audio.capture_stall_timeout_ms,
        "aec_tail_ms": cfg.audio.aec_tail_ms,
        "barge_in_min_speech_ms": cfg.audio.barge_in_min_speech_ms,
        "barge_in_hangover_ms": cfg.audio.barge_in_hangover_ms,
        "vad_aggressiveness": cfg.audio.vad_aggressiveness,
        "input_device": cfg.audio.input_device,
        "output_device": cfg.audio.output_device,
    }


def write_audio_runtime_config(cfg: Config, path: str | Path) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(
            json.dumps(audio_runtime_config_payload(cfg), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return destination


def _write_pid(path: Path, pid: int, command: list[str], cwd: Path) -> None:
    identity = _process_identity(pid)
    if identity is None:
        raise RuntimeError(f"unable to inspect started process pid={pid}")
    expected_cwd = cwd.resolve()
    if _normalized_path(identity.cwd) != _normalized_path(expected_cwd):
        raise RuntimeError(f"started process cwd mismatch: expected={expected_cwd} actual={identity.cwd}")
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "pid": pid,
                "command": command,
                "command_fingerprint": _command_fingerprint(command, cwd),
                "observed_command_fingerprint": identity.command_fingerprint,
                "cwd": str(identity.cwd),
                "process_create_time": identity.create_time,
                "created_at": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_pid_record(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _owned_process(service: ManagedService) -> ProcessIdentity | None:
    record = _read_pid_record(service.pid_file)
    if record is None:
        return None
    try:
        pid = int(record["pid"])
        expected_created = float(record["process_create_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if record.get("command_fingerprint") != _command_fingerprint(service.command, service.cwd):
        return None
    recorded_cwd_value = record.get("cwd")
    if not isinstance(recorded_cwd_value, str) or not recorded_cwd_value.strip():
        return None
    recorded_cwd = Path(recorded_cwd_value).resolve()
    if _normalized_path(recorded_cwd) != _normalized_path(service.cwd.resolve()):
        return None
    identity = _process_identity(pid)
    if identity is None or abs(identity.create_time - expected_created) > 0.01:
        return None
    if _normalized_path(identity.cwd) != _normalized_path(recorded_cwd):
        return None
    observed_fingerprint = record.get("observed_command_fingerprint")
    schema_version = record.get("schema_version", 1)
    if schema_version == 2:
        if not isinstance(observed_fingerprint, str) or observed_fingerprint != identity.command_fingerprint:
            return None
    elif schema_version != 1:
        return None
    return identity


def _owned_pid(service: ManagedService) -> int | None:
    identity = _owned_process(service)
    return identity.pid if identity is not None else None


def _command_fingerprint(command: list[str], cwd: Path) -> str:
    payload = json.dumps(
        {"command": [str(part) for part in command], "cwd": str(cwd.resolve())},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observed_command_fingerprint(command: list[str]) -> str:
    payload = json.dumps([str(part) for part in command], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path.resolve())))


def _process_identity(pid: int) -> ProcessIdentity | None:
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return None
            create_time = float(process.create_time())
            cwd = Path(process.cwd()).resolve()
            command = [str(part) for part in process.cmdline()]
        if not command:
            return None
        return ProcessIdentity(
            process=process,
            pid=pid,
            create_time=create_time,
            cwd=cwd,
            command_fingerprint=_observed_command_fingerprint(command),
        )
    except (psutil.Error, OSError, ValueError):
        return None


def _remove_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _is_pid_running(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except psutil.Error:
        return False


def _unverified_live_pid(service: ManagedService) -> int | None:
    if _owned_process(service) is not None:
        return None
    record = _read_pid_record(service.pid_file)
    try:
        pid = int(record["pid"]) if record is not None else 0
    except (KeyError, TypeError, ValueError):
        return None
    return pid if pid > 0 and _is_pid_running(pid) else None


def _terminate_owned_process(service: ManagedService) -> bool:
    identity = _owned_process(service)
    if identity is None:
        logger.warning("refusing to terminate unverified process for service %s", service.name)
        return False
    process = identity.process
    try:
        children = process.children(recursive=True)
    except (psutil.Error, OSError):
        children = []
    targets = [*reversed(children), process]
    for target in targets:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            continue
        except (psutil.Error, OSError) as exc:
            logger.warning("failed to terminate pid=%d: %s", target.pid, exc)
    _gone, alive = psutil.wait_procs(targets, timeout=5)
    if alive:
        logger.warning("processes still alive after termination: %s", ", ".join(str(item.pid) for item in alive))
        return False
    return True
