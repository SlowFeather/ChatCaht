from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed

from chatcaht.config import TtsConfig
from chatcaht.models import TtsChunk

logger = logging.getLogger(__name__)


class TtsClient:
    async def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        raise NotImplementedError

    async def close(self) -> None:
        return None


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
        self._ws = None
        self._request_lock = asyncio.Lock()
        self._closed = False

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                await ws.send(json.dumps({"type": "ping"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    return False, "tts service returned binary health response"
                msg = json.loads(raw)
                if msg.get("type") != "pong" or not msg.get("ready"):
                    return False, str(msg.get("last_error") or f"tts service state={msg.get('state')}")
                return True, f"tts ready state={msg.get('state')} model_loaded={msg.get('model_loaded')}"
        except Exception as exc:
            return False, str(exc)

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        sample_rate = 16000
        channels = 1
        payload: dict[str, object] = {
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
        logger.debug("tts synthesize chars=%d text=%s", len(text), text[:40])
        async with self._request_lock:
            for attempt in range(2):
                chunks = 0
                pcm_bytes = 0
                ws = None
                try:
                    ws = await self._connection()
                    await ws.send(json.dumps(payload, ensure_ascii=False))
                    await ws.send(json.dumps({"type": "flush"}))
                    while True:
                        raw = await ws.recv()
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
                except asyncio.CancelledError:
                    await asyncio.shield(self._disconnect(ws))
                    raise
                except (ConnectionClosed, OSError, TimeoutError):
                    await self._disconnect(ws)
                    if chunks == 0 and attempt == 0 and not self._closed:
                        logger.warning("tts connection failed before first PCM; retrying once")
                        continue
                    raise
                except Exception:
                    await self._disconnect(ws)
                    raise
                logger.debug(
                    "tts synthesize done chars=%d chunks=%d bytes=%d elapsed=%.2fs",
                    len(text),
                    chunks,
                    pcm_bytes,
                    time.monotonic() - started,
                )
                return

    async def close(self) -> None:
        self._closed = True
        await self._disconnect()

    async def _connection(self):
        if self._closed:
            raise RuntimeError("tts client is closed")
        ws = self._ws
        if ws is not None and ws.close_code is None:
            return ws
        await self._disconnect(ws)
        ws = await websockets.connect(
            self.cfg.url,
            open_timeout=self.timeout,
            ping_interval=20,
            ping_timeout=self.timeout,
            max_size=None,
        )
        self._ws = ws
        logger.info("tts ws connected: %s", self.cfg.url)
        return ws

    async def _disconnect(self, expected=None) -> None:
        ws = self._ws
        if expected is not None and ws is not expected:
            return
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                logger.debug("failed to close tts websocket", exc_info=True)


def create_tts_client(cfg: TtsConfig, timeout: float = 5.0) -> TtsClient:
    if not cfg.enabled or cfg.mode == "disabled":
        return DisabledTtsClient()
    if cfg.mode == "mock":
        return MockTtsClient()
    return ServiceTtsClient(cfg, timeout=timeout)
