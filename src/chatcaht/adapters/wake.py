from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from chatcaht.config import WakeConfig
from chatcaht.models import WakeEvent


class WakeClient:
    async def health(self) -> tuple[bool, str]:
        raise NotImplementedError

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

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
    def __init__(self, cfg: WakeConfig, timeout: float = 5.0):
        self.cfg = cfg
        self.timeout = timeout

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                await ws.send(json.dumps({"type": "ping"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                msg = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                if msg and msg.get("type") == "pong":
                    return True, f"wake service reachable at {self.cfg.url}"
                return True, f"wake service reachable; ping response={msg}"
        except Exception as exc:
            return False, str(exc)

    async def start(self) -> None:
        await self._command("start")

    async def stop(self) -> None:
        await self._command("stop")

    async def events(self) -> AsyncIterator[WakeEvent]:
        async with websockets.connect(self.cfg.url, max_size=None) as ws:
            await ws.recv()
            if self.cfg.auto_start_listening:
                await ws.send(json.dumps({"type": "start"}))
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

    async def _command(self, cmd: str) -> dict | None:
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": cmd}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                typ = msg.get("type")
                if typ == "error":
                    return msg
                if cmd == "status" and typ == "status":
                    return msg
                if typ == "ack" and msg.get("cmd") == cmd:
                    return msg
                if cmd == "ping" and typ == "pong":
                    return msg


def create_wake_client(cfg: WakeConfig, timeout: float = 5.0) -> WakeClient:
    if not cfg.enabled or cfg.mode == "disabled":
        return DisabledWakeClient()
    if cfg.mode == "mock":
        return MockWakeClient()
    return ServiceWakeClient(cfg, timeout=timeout)
