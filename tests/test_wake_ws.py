from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from chatcaht.adapters.wake import ServiceWakeClient
from chatcaht.config import WakeConfig


@pytest.mark.asyncio
async def test_service_wake_client_uses_websocket() -> None:
    async def handler(ws):
        await ws.send(json.dumps({"type": "status", "ready": True}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                await ws.send(json.dumps({"type": "pong"}))
            elif msg.get("type") == "start":
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))
                await ws.send(json.dumps({"type": "wake", "model": "xiaoyuan", "score": 0.91}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceWakeClient(WakeConfig(url=f"ws://127.0.0.1:{port}/v1/wake/ws"), timeout=2.0)
    try:
        ok, detail = await client.health()
        assert ok, detail

        events = client.events()
        event = await asyncio.wait_for(anext(events), timeout=2.0)
        assert event.model == "xiaoyuan"
        assert event.score == 0.91
        await events.aclose()
    finally:
        server.close()
        await server.wait_closed()
