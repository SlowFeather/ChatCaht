from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from chatcaht.config import SttConfig
from chatcaht.models import Transcript, TranscriptKind


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
                await ws.send(json.dumps({"type": "ping"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    return True, "stt service reachable"
                msg = json.loads(raw)
                return True, f"stt service reachable; response={msg.get('type')}"
        except Exception as exc:
            return False, str(exc)

    async def start(self) -> None:
        await self._command("start")

    async def stop(self) -> None:
        await self._command("stop")

    async def transcripts(self) -> AsyncIterator[Transcript]:
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, ping_interval=20, ping_timeout=self.timeout, max_size=None) as ws:
            if self.cfg.auto_start_listening:
                await ws.send(json.dumps({"type": "start"}))
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") != "transcript":
                    continue
                is_final = bool(msg.get("is_final")) or msg.get("event") == "final"
                kind = TranscriptKind.FINAL if is_final else TranscriptKind.PARTIAL
                if self.cfg.final_events_only and kind != TranscriptKind.FINAL:
                    continue
                text = str(msg.get("text") or "").strip()
                if text:
                    yield Transcript(
                        text=text,
                        kind=kind,
                        source=str(msg.get("source") or "microphone"),
                        segment_id=_optional_int(msg.get("segment_id")),
                        raw=msg,
                    )

    async def _command(self, typ: str) -> dict | None:
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
            await ws.send(json.dumps({"type": typ}))
            raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            if isinstance(raw, bytes):
                return None
            return json.loads(raw)


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
