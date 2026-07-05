from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import websockets

from chatcaht.config import TtsConfig
from chatcaht.models import TtsChunk

logger = logging.getLogger(__name__)


class TtsClient:
    async def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        raise NotImplementedError


class DisabledTtsClient(TtsClient):
    async def health(self) -> tuple[bool, str]:
        return True, "tts disabled"

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        if False:
            yield TtsChunk(pcm=b"", sample_rate=16000)


class MockTtsClient(TtsClient):
    async def health(self) -> tuple[bool, str]:
        return True, "mock tts ready"

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        await asyncio.sleep(0)
        yield TtsChunk(pcm=(text.encode("utf-8") or b"mock"), sample_rate=16000)


class ServiceTtsClient(TtsClient):
    def __init__(self, cfg: TtsConfig, timeout: float = 5.0):
        self.cfg = cfg
        self.timeout = timeout

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                await ws.send(json.dumps({"type": "ping"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    return True, "tts service reachable"
                msg = json.loads(raw)
                return True, f"tts service reachable; response={msg.get('type')}"
        except Exception as exc:
            return False, str(exc)

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        sample_rate = 16000
        channels = 1
        payload = {
            "type": "text",
            "text": text,
        }
        if self.cfg.speed is not None:
            payload["speed"] = self.cfg.speed
        if self.cfg.speaker:
            payload["speaker"] = self.cfg.speaker
        if self.cfg.speaker_id is not None:
            payload["speaker_id"] = self.cfg.speaker_id

        started = time.monotonic()
        chunks = 0
        pcm_bytes = 0
        logger.debug("tts synthesize chars=%d text=%s", len(text), text[:40])
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, ping_interval=20, ping_timeout=self.timeout, max_size=None) as ws:
            await ws.send(json.dumps(payload, ensure_ascii=False))
            await ws.send(json.dumps({"type": "flush"}))
            async for raw in ws:
                if isinstance(raw, bytes):
                    chunks += 1
                    pcm_bytes += len(raw)
                    yield TtsChunk(pcm=raw, sample_rate=sample_rate, channels=channels)
                    continue
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ == "start":
                    sample_rate = int(msg.get("sample_rate") or sample_rate)
                    channels = int(msg.get("channels") or channels)
                elif typ == "flushed":
                    break
                elif typ == "error":
                    logger.warning("tts service error for text=%s: %s", text[:40], msg)
                    raise RuntimeError(str(msg.get("error") or msg.get("message") or "tts error"))
        logger.debug(
            "tts synthesize done chars=%d chunks=%d bytes=%d elapsed=%.2fs",
            len(text),
            chunks,
            pcm_bytes,
            time.monotonic() - started,
        )


def create_tts_client(cfg: TtsConfig, timeout: float = 5.0) -> TtsClient:
    if not cfg.enabled or cfg.mode == "disabled":
        return DisabledTtsClient()
    if cfg.mode == "mock":
        return MockTtsClient()
    return ServiceTtsClient(cfg, timeout=timeout)
