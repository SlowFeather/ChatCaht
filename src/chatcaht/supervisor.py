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
from .orchestrator import VoiceSession
from .service_manager import ServiceManager

logger = logging.getLogger(__name__)


class Supervisor:
    """长期运行守护：拉起底层服务、周期健康检查、会话崩溃自动重启。"""

    def __init__(self, cfg: Config, *, audio_factory) -> None:
        self.cfg = cfg
        self.audio_factory = audio_factory
        self.manager = ServiceManager(cfg) if cfg.supervisor.manage_services else None
        self._stop = asyncio.Event()
        self.session: VoiceSession | None = None

    def request_stop(self) -> None:
        self._stop.set()
        session = self.session
        if session is not None:
            session.request_stop()

    async def run(self) -> int:
        sup = self.cfg.supervisor
        logger.info(
            "supervisor starting: manage_services=%s health_check_interval=%.0fs",
            self.manager is not None,
            sup.health_check_interval_sec,
        )
        if self.manager is not None:
            await self._ensure_services()

        monitor: asyncio.Task | None = None
        if self.manager is not None:
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
                if self.manager is not None:
                    await self._ensure_services()
        finally:
            if monitor is not None:
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor
        logger.info("supervisor stopped")
        return 0

    async def _run_session_once(self) -> None:
        cfg = self.cfg
        wake = create_wake_client(cfg.wake, timeout=cfg.runtime.health_timeout_sec)
        stt = create_stt_client(cfg.stt, mock_inputs=cfg.runtime.mock_text_inputs, timeout=cfg.runtime.health_timeout_sec)
        tts = create_tts_client(cfg.tts, timeout=cfg.runtime.health_timeout_sec)
        llm = create_llm_client(cfg)
        audio: AudioSink = self.audio_factory()
        session = VoiceSession(
            duplex=cfg.duplex,
            wake=wake,
            stt=stt,
            tts=tts,
            llm=llm,
            audio=audio,
            wake_trigger_words=cfg.wake.trigger_words,
            runtime=cfg.runtime,
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
        assert self.manager is not None
        try:
            statuses = await self.manager.start(wait=True)
            for status in statuses:
                if not status.ok:
                    logger.warning("service %s not healthy after start: %s", status.name, status.detail)
        except Exception:
            logger.exception("failed to start managed services")

    async def _monitor_services(self) -> None:
        assert self.manager is not None
        interval = self.cfg.supervisor.health_check_interval_sec
        while not self._stop.is_set():
            if await self._sleep_unless_stopped(interval):
                return
            try:
                statuses = await self.manager.status()
            except Exception:
                logger.exception("service health check failed")
                continue
            unhealthy = [s.name for s in statuses if not s.ok]
            if not unhealthy:
                continue
            logger.warning("unhealthy services detected: %s; restarting them", ", ".join(unhealthy))
            try:
                await self.manager.stop(list(unhealthy))
                await self.manager.start(list(unhealthy), wait=True)
            except Exception:
                logger.exception("failed to restart services: %s", unhealthy)

    async def _sleep_unless_stopped(self, delay: float) -> bool:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        return self._stop.is_set()
