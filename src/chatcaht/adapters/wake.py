from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import websockets

from chatcaht.config import WakeConfig
from chatcaht.models import WakeEvent

logger = logging.getLogger(__name__)


class WakeClient:
    restart_on_stream_error = False

    async def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    async def events(self) -> AsyncIterator[WakeEvent]:
        raise NotImplementedError


class DisabledWakeClient(WakeClient):
    async def health(self) -> tuple[bool, str]:
        return True, "wake disabled"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def events(self) -> AsyncIterator[WakeEvent]:
        if False:
            yield WakeEvent(model="disabled", score=0.0)


class MockWakeClient(WakeClient):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[WakeEvent] = asyncio.Queue()

    async def health(self) -> tuple[bool, str]:
        return True, "mock wake ready"

    async def start(self) -> None:
        await self._queue.put(WakeEvent(model="mock", score=1.0, raw={"type": "wake"}))

    async def stop(self) -> None:
        return None

    async def events(self) -> AsyncIterator[WakeEvent]:
        if self._queue.empty():
            await self.start()
        while True:
            yield await self._queue.get()


class ServiceWakeClient(WakeClient):
    restart_on_stream_error = True

    def __init__(self, cfg: WakeConfig, timeout: float = 5.0, *, audio_runtime=None):
        self.cfg = cfg
        self.timeout = timeout
        self.audio_runtime = audio_runtime
        self._external_ws = None
        self._external_reader: asyncio.Task | None = None
        self._external_pump: asyncio.Task | None = None
        self._external_lock = asyncio.Lock()
        self._external_events: asyncio.Queue[WakeEvent | Exception] = asyncio.Queue(maxsize=8)
        self._external_waiters: dict[str, asyncio.Future[dict]] = {}

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                if msg.get("type") != "status":
                    return False, f"wake service returned unexpected health response: {msg.get('type')}"
                if not msg.get("ready"):
                    return False, str(msg.get("last_error") or msg.get("error") or f"wake service state={msg.get('state')}")
                return True, f"wake ready state={msg.get('state')} model_loaded={msg.get('model_loaded')}"
        except Exception as exc:
            return False, str(exc)

    async def start(self) -> None:
        if self.cfg.input_mode == "external_pcm":
            await self._ensure_external_stream()
            await self._external_command("start")
        else:
            await self._command("start")

    async def stop(self) -> None:
        if self.cfg.input_mode == "external_pcm":
            if self._external_ws is not None:
                await self._external_command("stop")
        else:
            await self._command("stop")

    async def close(self) -> None:
        for task in (self._external_pump, self._external_reader):
            if task is not None:
                task.cancel()
        await asyncio.gather(
            *(task for task in (self._external_pump, self._external_reader) if task is not None),
            return_exceptions=True,
        )
        self._external_pump = None
        self._external_reader = None
        ws = self._external_ws
        self._external_ws = None
        if ws is not None:
            await ws.close()

    async def events(self) -> AsyncIterator[WakeEvent]:
        if self.cfg.input_mode == "external_pcm":
            await self._ensure_external_stream()
            if self.cfg.auto_start_listening:
                await self._external_command("start")
            while True:
                item = await self._external_events.get()
                if isinstance(item, Exception):
                    raise item
                yield item
            return
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, ping_interval=20, ping_timeout=self.timeout, max_size=None) as ws:
            await ws.recv()
            logger.info("wake ws connected: %s", self.cfg.url)
            if self.cfg.auto_start_listening:
                await ws.send(json.dumps({"type": "start"}))
                logger.debug("wake listening start command sent")
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "wake":
                    yield WakeEvent(
                        model=str(msg.get("model") or "wake"),
                        score=float(msg.get("score") or 0.0),
                        raw=msg,
                    )

    async def _ensure_external_stream(self) -> None:
        if self.audio_runtime is None:
            raise RuntimeError("wake external_pcm mode requires AudioRuntimeClient")
        ws = self._external_ws
        if ws is not None and ws.close_code is None:
            return
        async with self._external_lock:
            ws = self._external_ws
            if ws is not None and ws.close_code is None:
                return
            ws = await websockets.connect(
                self.cfg.url,
                open_timeout=self.timeout,
                ping_interval=20,
                ping_timeout=self.timeout,
                max_size=None,
            )
            await ws.recv()
            await ws.send(
                json.dumps(
                    {
                        "type": "audio_open",
                        "sample_rate": self.audio_runtime.cfg.capture_sample_rate,
                        "channels": 1,
                        "frame_ms": self.audio_runtime.cfg.frame_ms,
                    }
                )
            )
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "error":
                    await ws.close()
                    raise RuntimeError(str(msg.get("message") or "wake audio_open failed"))
                if msg.get("type") == "ack" and msg.get("cmd") == "audio_open":
                    break
            self._external_ws = ws
            self._external_reader = asyncio.create_task(self._external_read_loop(ws))
            self._external_pump = asyncio.create_task(self._external_audio_pump(ws))
            logger.info("wake external PCM stream connected")

    async def _external_command(self, cmd: str) -> dict:
        await self._ensure_external_stream()
        ws = self._external_ws
        assert ws is not None
        existing = self._external_waiters.get(cmd)
        if existing is not None and not existing.done():
            return await existing
        future = asyncio.get_running_loop().create_future()
        self._external_waiters[cmd] = future
        try:
            await ws.send(json.dumps({"type": cmd}))
            return await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._external_waiters.pop(cmd, None)

    async def _external_read_loop(self, ws) -> None:
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                typ = msg.get("type")
                cmd = str(msg.get("cmd") or "")
                waiter = self._external_waiters.get(cmd)
                if typ == "error" and waiter is not None and not waiter.done():
                    waiter.set_exception(RuntimeError(str(msg.get("message") or msg.get("code") or "wake command failed")))
                elif typ == "ack" and waiter is not None and not waiter.done():
                    waiter.set_result(msg)
                elif typ == "wake":
                    event = WakeEvent(model=str(msg.get("model") or "wake"), score=float(msg.get("score") or 0.0), raw=msg)
                    if self._external_events.full():
                        self._external_events.get_nowait()
                    self._external_events.put_nowait(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for waiter in self._external_waiters.values():
                if not waiter.done():
                    waiter.set_exception(exc)
        finally:
            if self._external_ws is ws:
                self._external_ws = None

    async def _external_audio_pump(self, ws) -> None:
        try:
            async for frame in self.audio_runtime.wake_frames():
                await ws.send(frame.pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("wake external PCM pump failed")
            if self._external_events.full():
                self._external_events.get_nowait()
            self._external_events.put_nowait(exc)
            await ws.close()

    async def _command(self, cmd: str) -> dict | None:
        logger.debug("wake command: %s", cmd)
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, ping_interval=20, ping_timeout=self.timeout, max_size=None) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": cmd}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ == "error":
                    raise RuntimeError(str(msg.get("message") or msg.get("code") or f"wake {cmd} failed"))
                if cmd == "status" and typ == "status":
                    return msg
                if typ == "ack" and msg.get("cmd") == cmd:
                    return msg
                if cmd == "ping" and typ == "pong":
                    return msg


def create_wake_client(cfg: WakeConfig, timeout: float = 5.0, *, audio_runtime=None) -> WakeClient:
    if not cfg.enabled or cfg.mode == "disabled":
        return DisabledWakeClient()
    if cfg.mode == "mock":
        return MockWakeClient()
    return ServiceWakeClient(cfg, timeout=timeout, audio_runtime=audio_runtime)
