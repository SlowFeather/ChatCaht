from __future__ import annotations

import asyncio
import json

import pytest
import websockets

from chatcaht.adapters.tts import ServiceTtsClient
from chatcaht.config import TtsConfig


@pytest.mark.asyncio
async def test_service_tts_does_not_send_default_speed_override() -> None:
    received: dict | None = None

    async def handler(ws):
        nonlocal received
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                received = msg
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
                await ws.send(b"\x01\x00")
            elif msg.get("type") == "flush":
                await ws.send(json.dumps({"type": "flushed"}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        chunks = [chunk async for chunk in client.synthesize("\u4f60\u597d")]
        assert chunks[0].pcm == b"\x01\x00"
        assert received is not None
        assert "speed" not in received
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_sends_explicit_speed() -> None:
    received: dict | None = None

    async def handler(ws):
        nonlocal received
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                received = msg
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
            elif msg.get("type") == "flush":
                await ws.send(json.dumps({"type": "flushed"}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws", speed=0.85), timeout=2.0)
    try:
        await asyncio.wait_for(_drain(client.synthesize("\u4f60\u597d")), timeout=2.0)
        assert received is not None
        assert received["speed"] == 0.85
    finally:
        server.close()
        await server.wait_closed()


async def _drain(stream) -> None:
    async for _chunk in stream:
        pass
