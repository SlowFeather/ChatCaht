from __future__ import annotations

import asyncio
import contextlib
import logging
import time

from .adapters.llm import create_llm_client
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .audio import AudioSink
from .config import Config
from .metrics import MetricsRecorder
from .models import ServiceState
from .orchestrator import VoiceSession
from .service_manager import ServiceManager

logger = logging.getLogger(__name__)


class Supervisor:
    """长期运行守护：拉起底层服务、周期健康检查、会话崩溃自动重启。"""

    def __init__(self, cfg: Config, *, audio_factory) -> None:
        self.cfg = cfg
        self.audio_factory = audio_factory
        self.manage_services = cfg.supervisor.manage_services
        self.manager = ServiceManager(cfg)
        self._service_op_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self.session: VoiceSession | None = None
        self.audio: AudioSink | None = None

    def request_stop(self) -> None:
        self._stop.set()
        session = self.session
        if session is not None:
            session.request_stop()

    async def run(self) -> int:
        sup = self.cfg.supervisor
        logger.info(
            "supervisor starting: manage_services=%s health_check_interval=%.0fs",
            self.manage_services,
            sup.health_check_interval_sec,
        )
        await self._ensure_services()

        self.audio = self.audio_factory()
        start_audio = getattr(self.audio, "start", None)
        if start_audio is not None:
            await start_audio()

        monitor = asyncio.create_task(self._monitor_services())

        delay = sup.restart_delay_sec
        try:
            while not self._stop.is_set():
                started_at = time.monotonic()
                try:
                    await self._run_session_once()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("voice session crashed")
                if self._stop.is_set():
                    break
                ran_for = time.monotonic() - started_at
                if ran_for >= sup.stable_run_sec:
                    delay = sup.restart_delay_sec
                logger.warning("voice session exited after %.1fs; restarting in %.1fs", ran_for, delay)
                if await self._sleep_unless_stopped(delay):
                    break
                delay = min(delay * 2, sup.max_restart_delay_sec)
                await self._ensure_services()
        finally:
            monitor.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor
            if self.audio is not None:
                with contextlib.suppress(Exception):
                    await self.audio.close()
                self.audio = None
        logger.info("supervisor stopped")
        return 0

    async def _run_session_once(self) -> None:
        cfg = self.cfg
        audio = self.audio
        if audio is None:
            raise RuntimeError("audio runtime is not initialized")
        wake = create_wake_client(cfg.wake, timeout=cfg.runtime.health_timeout_sec, audio_runtime=audio)
        stt = create_stt_client(
            cfg.stt,
            mock_inputs=cfg.runtime.mock_text_inputs,
            timeout=cfg.runtime.health_timeout_sec,
            audio_runtime=audio,
        )
        tts = create_tts_client(cfg.tts, timeout=cfg.runtime.health_timeout_sec)
        llm = create_llm_client(cfg)
        session = VoiceSession(
            duplex=cfg.duplex,
            wake=wake,
            stt=stt,
            tts=tts,
            llm=llm,
            audio=audio,
            wake_trigger_words=cfg.wake.trigger_words,
            runtime=cfg.runtime,
            metrics=MetricsRecorder(cfg.paths.metrics_file),
        )
        self.session = session
        try:
            stats = await session.run()
            logger.info(
                "session finished: conversations=%d wake=%d user=%d assistant=%d interrupts=%d",
                stats.conversations,
                stats.wake_events,
                stats.user_turns,
                stats.assistant_turns,
                stats.interruptions,
            )
        finally:
            self.session = None
            with contextlib.suppress(Exception):
                await llm.close()

    async def _ensure_services(self) -> None:
        try:
            async with self._service_op_lock:
                if self.manage_services:
                    statuses = await self.manager.start(wait=True)
                else:
                    statuses = await self.manager.wait_ready()
            not_ready = [status for status in statuses if status.state is not ServiceState.READY]
            if not_ready:
                raise RuntimeError(
                    "required services not READY: "
                    + ", ".join(f"{status.name}={status.state.value}:{status.detail}" for status in not_ready)
                )
        except Exception:
            logger.exception("failed to start managed services")
            raise

    async def _monitor_services(self) -> None:
        interval = self.cfg.supervisor.health_check_interval_sec
        while not self._stop.is_set():
            if await self._sleep_unless_stopped(interval):
                return
            try:
                statuses = await self.manager.status()
            except Exception:
                logger.exception("service health check failed")
                continue
            not_ready = [status for status in statuses if status.state is not ServiceState.READY]
            if not not_ready:
                continue
            logger.warning(
                "required services not READY: %s",
                ", ".join(f"{status.name}={status.state.value}" for status in not_ready),
            )
            if self.session is not None:
                logger.error("service readiness lost; terminating the current voice session")
                self.session.request_stop()
            failed = [status.name for status in not_ready if status.state is ServiceState.FAILED]
            if not failed or not self.manage_services:
                continue
            try:
                async with self._service_op_lock:
                    current = await self.manager.status(list(failed))
                    still_failed = [status.name for status in current if status.state is ServiceState.FAILED]
                    if not still_failed:
                        continue
                    logger.warning("restarting FAILED services: %s", ", ".join(still_failed))
                    await self.manager.stop(list(still_failed))
                    await self.manager.start(list(still_failed), wait=True)
            except Exception:
                logger.exception("failed to restart services: %s", failed)

    async def _sleep_unless_stopped(self, delay: float) -> bool:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        return self._stop.is_set()
