from __future__ import annotations

import json
import asyncio

import pytest
import websockets

from chatcaht.adapters.stt import ServiceSttClient
from chatcaht.config import SttConfig


@pytest.mark.asyncio
async def test_service_stt_command_waits_for_matching_ack() -> None:
    received: list[dict] = []

    async def handler(ws):
        await ws.send(json.dumps({"type": "status", "listening": False}))
        async for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            if msg.get("type") == "start":
                await ws.send(json.dumps({"type": "status", "listening": False}))
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceSttClient(SttConfig(url=f"ws://127.0.0.1:{port}/v1/stt/ws"), timeout=2.0)
    try:
        await client.start()
        assert received == [{"type": "start"}]
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_stt_promotes_stale_partial_to_final() -> None:
    async def handler(ws):
        await ws.send(json.dumps({"type": "status", "listening": True}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "start":
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript",
                            "event": "partial",
                            "is_final": False,
                            "text": "今天天气怎么样",
                            "source": "microphone",
                            "segment_id": 7,
                        },
                        ensure_ascii=False,
                    )
                )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = SttConfig(
        url=f"ws://127.0.0.1:{port}/v1/stt/ws",
        final_events_only=True,
        partial_fallback_sec=0.05,
    )
    client = ServiceSttClient(cfg, timeout=2.0)
    try:
        stream = client.transcripts()
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert transcript.is_final
        assert transcript.text == "今天天气怎么样"
        assert transcript.segment_id == 7
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_stt_uses_real_final_before_partial_fallback() -> None:
    async def handler(ws):
        await ws.send(json.dumps({"type": "status", "listening": True}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "start":
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript",
                            "event": "partial",
                            "is_final": False,
                            "text": "今天天气",
                            "segment_id": 0,
                        },
                        ensure_ascii=False,
                    )
                )
                await asyncio.sleep(0.01)
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript",
                            "event": "final",
                            "is_final": True,
                            "text": "今天天气怎么样",
                            "segment_id": 0,
                        },
                        ensure_ascii=False,
                    )
                )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = SttConfig(
        url=f"ws://127.0.0.1:{port}/v1/stt/ws",
        final_events_only=True,
        partial_fallback_sec=0.2,
    )
    client = ServiceSttClient(cfg, timeout=2.0)
    try:
        stream = client.transcripts()
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert transcript.is_final
        assert transcript.text == "今天天气怎么样"
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()
