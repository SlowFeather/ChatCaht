from __future__ import annotations

import json

import pytest
import websockets

import chatcaht.health as health
from chatcaht.adapters.stt import ServiceSttClient
from chatcaht.adapters.tts import ServiceTtsClient
from chatcaht.config import Config, SttConfig, TtsConfig
from chatcaht.models import ServiceState, service_probe_from_status


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


def test_service_probe_normalizes_lifecycle_states() -> None:
    assert service_probe_from_status({"ready": False, "state": "loading"}, service="test").state is ServiceState.STARTING
    assert service_probe_from_status({"ready": False, "state": "degraded"}, service="test").state is ServiceState.DEGRADED
    assert service_probe_from_status({"ready": False, "state": "fatal"}, service="test").state is ServiceState.FAILED
    assert service_probe_from_status({"ready": True, "state": "failed"}, service="test").state is ServiceState.READY


def test_deep_soundcard_check_requires_input_and_output(monkeypatch) -> None:
    devices = {
        "input": {"name": "Microphone", "max_input_channels": 2, "max_output_channels": 0},
        "output": {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    }
    monkeypatch.setattr(health.sd, "query_devices", lambda _device, kind: devices[kind])

    check = health._soundcard_check(Config())

    assert check.ok
    assert check.state is ServiceState.READY
    assert "Microphone" in check.detail
    assert "Speakers" in check.detail
