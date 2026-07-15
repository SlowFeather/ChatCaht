from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import websockets

from chatcaht.config import SttConfig
from chatcaht.models import Transcript, TranscriptKind

logger = logging.getLogger(__name__)


class SttClient:
    restart_on_stream_end = False
    restart_on_stream_error = False

    async def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def transcripts(self) -> AsyncIterator[Transcript]:
        raise NotImplementedError


class DisabledSttClient(SttClient):
    async def health(self) -> tuple[bool, str]:
        return True, "stt disabled"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def transcripts(self) -> AsyncIterator[Transcript]:
        if False:
            yield Transcript(text="", kind=TranscriptKind.FINAL)


class MockSttClient(SttClient):
    def __init__(self, inputs: list[str] | None = None, turn_delay: float = 0.0):
        self.inputs = list(inputs or [])
        self.turn_delay = turn_delay
        self._started = asyncio.Event()

    async def health(self) -> tuple[bool, str]:
        return True, "mock stt ready"

    async def start(self) -> None:
        self._started.set()

    async def stop(self) -> None:
        return None

    async def transcripts(self) -> AsyncIterator[Transcript]:
        await self._started.wait()
        for index, text in enumerate(self.inputs):
            await asyncio.sleep(0)
            yield Transcript(text=text, kind=TranscriptKind.FINAL, source="mock", segment_id=index)
            if self.turn_delay:
                await asyncio.sleep(self.turn_delay)


class ServiceSttClient(SttClient):
    restart_on_stream_end = True
    restart_on_stream_error = True

    def __init__(self, cfg: SttConfig, timeout: float = 5.0):
        self.cfg = cfg
        self.timeout = timeout

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                msg = json.loads(raw)
                if msg.get("type") != "status":
                    return False, f"stt service returned unexpected health response: {msg.get('type')}"
                if not msg.get("ready"):
                    return False, str(msg.get("last_error") or f"stt service state={msg.get('state')}")
                return True, f"stt ready state={msg.get('state')} model_loaded={msg.get('model_loaded')}"
        except Exception as exc:
            return False, str(exc)

    async def start(self) -> None:
        resp = await self._command("start")
        logger.info("stt start command response: %s", _summarize_message(resp))

    async def stop(self) -> None:
        resp = await self._command("stop")
        logger.info("stt stop command response: %s", _summarize_message(resp))

    async def transcripts(self) -> AsyncIterator[Transcript]:
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, ping_interval=20, ping_timeout=self.timeout, max_size=None) as ws:
            logger.info("stt ws connected: %s", self.cfg.url)
            if self.cfg.auto_start_listening:
                await ws.send(json.dumps({"type": "start"}))
                logger.info("stt transcript stream sent start command")
            pending_partial: Transcript | None = None
            pending_partial_at = 0.0
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "status":
                    logger.info("stt stream status: %s", _summarize_message(msg))
                    logger.debug("stt stream status text fields: %s", _summarize_message(msg, include_text=True))
                    continue
                if msg_type == "error":
                    logger.warning("stt stream error: %s", _summarize_message(msg))
                    continue
                if msg_type == "ack":
                    logger.info("stt stream ack: %s", _summarize_message(msg))
                    continue
                if msg_type != "transcript":
                    logger.debug("stt stream ignored message: %s", _summarize_message(msg))
                    continue
                is_final = bool(msg.get("is_final")) or msg.get("event") == "final"
                kind = TranscriptKind.FINAL if is_final else TranscriptKind.PARTIAL
                text = str(msg.get("text") or "").strip()
                logger.info(
                    "stt transcript received kind=%s source=%s segment=%s chars=%d",
                    kind.value,
                    msg.get("source"),
                    msg.get("segment_id"),
                    len(text),
                )
                logger.debug("stt transcript raw text=%s", text)
                if text:
                    transcript = Transcript(
                        text=text,
                        kind=kind,
                        source=str(msg.get("source") or "microphone"),
                        segment_id=_optional_int(msg.get("segment_id")),
                        raw=msg,
                    )
                    if kind == TranscriptKind.FINAL:
                        pending_partial = None
                        yield transcript
                    elif self.cfg.final_events_only:
                        if len(text) >= self.cfg.partial_min_chars:
                            pending_partial = transcript
                            pending_partial_at = time.monotonic()
                            yield transcript
                            if self.cfg.partial_fallback_sec <= 0:
                                continue
                            async for fallback in self._wait_for_partial_fallback(ws, pending_partial, pending_partial_at):
                                if fallback.is_final:
                                    pending_partial = None
                                yield fallback
                        else:
                            logger.debug("stt partial ignored because final_events_only=true text=%s", text)
                    else:
                        yield transcript

    async def _wait_for_partial_fallback(
        self,
        ws,
        pending: Transcript,
        pending_at: float,
    ) -> AsyncIterator[Transcript]:
        while True:
            elapsed = time.monotonic() - pending_at
            remaining = self.cfg.partial_fallback_sec - elapsed
            if remaining <= 0:
                logger.info(
                    "stt partial fallback promoted to final source=%s segment=%s chars=%d",
                    pending.source,
                    pending.segment_id,
                    len(pending.text),
                )
                logger.debug("stt partial fallback promoted raw text=%s", pending.text)
                self._schedule_reset_after_partial_fallback()
                yield Transcript(
                    text=pending.text,
                    kind=TranscriptKind.FINAL,
                    source=pending.source,
                    segment_id=pending.segment_id,
                    raw=pending.raw,
                )
                return
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                logger.info(
                    "stt partial fallback promoted to final source=%s segment=%s chars=%d",
                    pending.source,
                    pending.segment_id,
                    len(pending.text),
                )
                logger.debug("stt partial fallback promoted raw text=%s", pending.text)
                self._schedule_reset_after_partial_fallback()
                yield Transcript(
                    text=pending.text,
                    kind=TranscriptKind.FINAL,
                    source=pending.source,
                    segment_id=pending.segment_id,
                    raw=pending.raw,
                )
                return
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            msg_type = msg.get("type")
            if msg_type == "status":
                logger.info("stt stream status: %s", _summarize_message(msg))
                logger.debug("stt stream status text fields: %s", _summarize_message(msg, include_text=True))
                continue
            if msg_type == "error":
                logger.warning("stt stream error: %s", _summarize_message(msg))
                continue
            if msg_type == "ack":
                logger.info("stt stream ack: %s", _summarize_message(msg))
                continue
            if msg_type != "transcript":
                logger.debug("stt stream ignored message: %s", _summarize_message(msg))
                continue

            is_final = bool(msg.get("is_final")) or msg.get("event") == "final"
            kind = TranscriptKind.FINAL if is_final else TranscriptKind.PARTIAL
            text = str(msg.get("text") or "").strip()
            logger.info(
                "stt transcript received kind=%s source=%s segment=%s chars=%d",
                kind.value,
                msg.get("source"),
                msg.get("segment_id"),
                len(text),
            )
            logger.debug("stt transcript raw text=%s", text)
            if not text:
                continue
            transcript = Transcript(
                text=text,
                kind=kind,
                source=str(msg.get("source") or "microphone"),
                segment_id=_optional_int(msg.get("segment_id")),
                raw=msg,
            )
            if kind == TranscriptKind.FINAL:
                yield transcript
                return
            if len(text) < self.cfg.partial_min_chars:
                logger.debug("stt partial ignored because text is shorter than partial_min_chars text=%s", text)
                continue
            yield transcript
            if self.cfg.partial_fallback_sec <= 0:
                continue
            pending = transcript
            pending_at = time.monotonic()

    def _schedule_reset_after_partial_fallback(self) -> None:
        task = asyncio.create_task(self._reset_service_after_partial_fallback())
        task.add_done_callback(_log_background_task_error)

    async def _reset_service_after_partial_fallback(self) -> None:
        logger.info("stt resetting service after partial fallback to clear ASR segment state")
        try:
            await self.stop()
            await self.start()
        except Exception:
            logger.exception("stt reset after partial fallback failed")

    async def _command(self, typ: str) -> dict | None:
        logger.debug("stt command: %s", typ)
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
            await ws.send(json.dumps({"type": typ}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                msg_type = msg.get("type")
                if msg_type == "error":
                    logger.warning("stt command %s error response: %s", typ, _summarize_message(msg))
                    raise RuntimeError(str(msg.get("message") or msg.get("code") or f"stt {typ} failed"))
                if typ == "status" and msg_type == "status":
                    logger.info("stt command %s status response: %s", typ, _summarize_message(msg))
                    return msg
                if msg_type == "ack" and msg.get("cmd") == typ:
                    logger.info("stt command %s ack response: %s", typ, _summarize_message(msg))
                    return msg
                if typ == "ping" and msg_type == "pong":
                    logger.info("stt command %s pong response: %s", typ, _summarize_message(msg))
                    return msg


def _summarize_message(msg: dict | None, *, include_text: bool = False) -> dict | None:
    if msg is None:
        return None
    keys = [
        "type",
        "cmd",
        "ok",
        "ready",
        "listening",
        "worker_state",
        "last_error",
        "audio_restart_count",
        "event",
        "is_final",
        "source",
        "segment_id",
    ]
    if include_text:
        keys.extend(("last_text", "text"))
    return {key: msg.get(key) for key in keys if key in msg}


def _log_background_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("stt background reset task failed", exc_info=exc)


def _optional_int(value) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def create_stt_client(cfg: SttConfig, *, mock_inputs: list[str] | None = None, timeout: float = 5.0) -> SttClient:
    if not cfg.enabled or cfg.mode == "disabled":
        return DisabledSttClient()
    if cfg.mode == "mock":
        return MockSttClient(mock_inputs)
    return ServiceSttClient(cfg, timeout=timeout)
