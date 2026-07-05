from __future__ import annotations

import json
import asyncio
import logging

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
    commands: list[str] = []
    sent_partial = False

    async def handler(ws):
        nonlocal sent_partial
        await ws.send(json.dumps({"type": "status", "listening": True}))
        async for raw in ws:
            msg = json.loads(raw)
            cmd = msg.get("type")
            commands.append(cmd)
            if cmd == "start":
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))
                if not sent_partial:
                    sent_partial = True
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
            elif cmd == "stop":
                await ws.send(json.dumps({"type": "ack", "cmd": "stop", "ok": True}))

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
        partial = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert not partial.is_final
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert transcript.is_final
        assert transcript.text == partial.text
        assert transcript.segment_id == 7
        for _ in range(20):
            if commands[:3] == ["start", "stop", "start"]:
                break
            await asyncio.sleep(0.05)
        assert commands[:3] == ["start", "stop", "start"]
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_stt_waits_for_final_when_partial_fallback_disabled() -> None:
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
                            "text": "今天",
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
        partial_fallback_sec=0,
    )
    client = ServiceSttClient(cfg, timeout=2.0)
    try:
        stream = client.transcripts()
        partial = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert not partial.is_final
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(stream), timeout=0.15)
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_stt_resets_partial_fallback_timer_on_updated_partial() -> None:
    sent = asyncio.Event()

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
                            "text": "西红柿",
                            "segment_id": 0,
                        },
                        ensure_ascii=False,
                    )
                )
                await asyncio.sleep(0.06)
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript",
                            "event": "partial",
                            "is_final": False,
                            "text": "西红柿和香蕉哪个好吃",
                            "segment_id": 0,
                        },
                        ensure_ascii=False,
                    )
                )
                sent.set()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = SttConfig(
        url=f"ws://127.0.0.1:{port}/v1/stt/ws",
        final_events_only=True,
        partial_fallback_sec=0.1,
    )
    client = ServiceSttClient(cfg, timeout=2.0)
    try:
        stream = client.transcripts()
        first_partial = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert not first_partial.is_final
        second_partial = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert not second_partial.is_final
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert sent.is_set()
        assert transcript.is_final
        assert transcript.text == second_partial.text
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
        partial = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert not partial.is_final
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert transcript.is_final
        assert transcript.text == "今天天气怎么样"
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_service_stt_info_logs_do_not_include_raw_transcript_text(caplog) -> None:
    accumulated_text = "first questionsecond question"

    async def handler(ws):
        await ws.send(
            json.dumps(
                {
                    "type": "status",
                    "ready": True,
                    "listening": True,
                    "last_text": accumulated_text,
                },
                ensure_ascii=False,
            )
        )
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "start":
                await ws.send(json.dumps({"type": "ack", "cmd": "start", "ok": True}))
                await ws.send(
                    json.dumps(
                        {
                            "type": "transcript",
                            "event": "final",
                            "is_final": True,
                            "text": accumulated_text,
                            "source": "microphone",
                            "segment_id": 3,
                        },
                        ensure_ascii=False,
                    )
                )

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg = SttConfig(url=f"ws://127.0.0.1:{port}/v1/stt/ws")
    client = ServiceSttClient(cfg, timeout=2.0)
    try:
        caplog.set_level(logging.INFO, logger="chatcaht.adapters.stt")
        stream = client.transcripts()
        transcript = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert transcript.text == accumulated_text
        await stream.aclose()
    finally:
        server.close()
        await server.wait_closed()

    info_logs = "\n".join(record.getMessage() for record in caplog.records if record.levelno == logging.INFO)
    assert accumulated_text not in info_logs
    assert "last_text" not in info_logs
    assert f"chars={len(accumulated_text)}" in info_logs
