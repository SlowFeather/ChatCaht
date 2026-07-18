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
        await client.close()
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
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_reuses_connection_and_serializes_requests() -> None:
    connections = 0
    received: list[str] = []

    async def handler(ws):
        nonlocal connections
        connections += 1
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                received.append(msg["text"])
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
                await ws.send(msg["text"].encode())
            elif msg.get("type") == "flush":
                await ws.send(json.dumps({"type": "end"}))
                await ws.send(json.dumps({"type": "flushed"}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        first, second = await asyncio.gather(
            _collect(client.synthesize("first")),
            _collect(client.synthesize("second")),
        )
        assert first[0].pcm == b"first"
        assert second[0].pcm == b"second"
        assert received == ["first", "second"]
        assert connections == 1
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_retries_once_when_connection_fails_before_pcm() -> None:
    connections = 0

    async def handler(ws):
        nonlocal connections
        connections += 1
        connection = connections
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                if connection == 1:
                    await ws.close(code=1011, reason="retry")
                    return
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
                await ws.send(b"ok")
            elif msg.get("type") == "flush":
                await ws.send(json.dumps({"type": "flushed"}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        chunks = await _collect(client.synthesize("retry"))
        assert [chunk.pcm for chunk in chunks] == [b"ok"]
        assert connections == 2
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_does_not_retry_service_errors() -> None:
    connections = 0

    async def handler(ws):
        nonlocal connections
        connections += 1
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                await ws.send(json.dumps({"type": "error", "code": "QUEUE_FULL", "error": "busy"}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            await _drain(client.synthesize("do not retry"))
        assert connections == 1
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_does_not_retry_after_pcm() -> None:
    connections = 0

    async def handler(ws):
        nonlocal connections
        connections += 1
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "text":
                await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
                await ws.send(b"partial")
                await ws.close(code=1011, reason="after pcm")
                return

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        stream = client.synthesize("no retry")
        first = await anext(stream)
        assert first.pcm == b"partial"
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await anext(stream)
        assert connections == 1
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_tts_cancellation_disconnects_and_next_request_reconnects() -> None:
    connections = 0
    first_started = asyncio.Event()
    first_closed = asyncio.Event()

    async def handler(ws):
        nonlocal connections
        connections += 1
        connection = connections
        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "text" and connection == 1:
                    first_started.set()
                    await ws.wait_closed()
                    return
                if msg.get("type") == "text":
                    await ws.send(json.dumps({"type": "start", "sample_rate": 16000, "channels": 1}))
                    await ws.send(b"next")
                elif msg.get("type") == "flush":
                    await ws.send(json.dumps({"type": "flushed"}))
        finally:
            if connection == 1:
                first_closed.set()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client = ServiceTtsClient(TtsConfig(url=f"ws://127.0.0.1:{port}/v1/tts/ws"), timeout=2.0)
    try:
        task = asyncio.create_task(_drain(client.synthesize("cancel")))
        await asyncio.wait_for(first_started.wait(), timeout=2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(first_closed.wait(), timeout=2.0)
        chunks = await _collect(client.synthesize("next"))
        assert [chunk.pcm for chunk in chunks] == [b"next"]
        assert connections == 2
        await client.close()
        await client.close()
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


async def _drain(stream) -> None:
    async for _chunk in stream:
        pass


async def _collect(stream):
    return [chunk async for chunk in stream]
