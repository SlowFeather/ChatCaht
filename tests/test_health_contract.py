from __future__ import annotations

import json

import pytest
import websockets

from chatcaht.adapters.stt import ServiceSttClient
from chatcaht.adapters.tts import ServiceTtsClient
from chatcaht.config import SttConfig, TtsConfig


HEALTH = {
    "ready": True,
    "state": "ready",
    "model_loaded": True,
    "audio_open": False,
    "last_error": None,
}


@pytest.mark.asyncio
async def test_stt_health_requires_ready_status() -> None:
    async def handler(ws):
        await ws.send(json.dumps({"type": "status", **HEALTH}))
        await ws.wait_closed()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = ServiceSttClient(SttConfig(url=f"ws://127.0.0.1:{port}/v1/stt/ws"), timeout=1)
        ok, detail = await client.health()
        assert ok, detail
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_tts_health_requires_ready_pong() -> None:
    async def handler(ws):
        async for raw in ws:
            if json.loads(raw).get("type") == "ping":
                await ws.send(json.dumps({"type": "pong", **HEALTH}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=1)
        ok, detail = await client.health()
        assert ok, detail
    finally:
        server.close()
        await server.wait_closed()
