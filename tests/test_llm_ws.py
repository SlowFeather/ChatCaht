from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest
import websockets

from chatcaht.adapters.llm import LollamaChatClient, create_llm_client
from chatcaht.config import Config, LollamaConfig, load_config
from chatcaht.openai_client import OpenAICompatibleClient


@contextlib.asynccontextmanager
async def fake_lollama(handler):
    server = await websockets.serve(handler, "127.0.0.1", 0, max_size=None)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}/v1/llm/ws"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_stream_chat_yields_deltas_until_done() -> None:
    received: dict | None = None

    async def handler(ws):
        nonlocal received
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "chat":
                received = msg
                rid = msg["request_id"]
                for ch in "你好呀":
                    await ws.send(json.dumps({"type": "delta", "request_id": rid, "text": ch}))
                await ws.send(json.dumps({"type": "tool", "request_id": rid, "name": "get_current_time", "status": "done", "detail": ""}))
                await ws.send(json.dumps({"type": "done", "request_id": rid, "text": "你好呀", "canceled": False}))

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)
        messages = [{"role": "user", "content": "你好"}]
        chunks = [chunk async for chunk in client.stream_chat(messages)]
        assert "".join(chunks) == "你好呀"
        assert received is not None
        assert received["text"] == "你好"
        assert "messages" not in received
        assert "system_messages" not in received


@pytest.mark.asyncio
async def test_stream_chat_reuses_connection_for_lollama_working_history() -> None:
    received: list[dict] = []
    connections = 0

    async def handler(ws):
        nonlocal connections
        connections += 1
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "chat":
                received.append(msg)
                await ws.send(
                    json.dumps(
                        {
                            "type": "done",
                            "request_id": msg["request_id"],
                            "text": "",
                            "canceled": False,
                        }
                    )
                )

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)
        async for _ in client.stream_chat([{"role": "user", "content": "第一句"}]):
            pass
        async for _ in client.stream_chat([{"role": "user", "content": "第二句"}]):
            pass
        await client.close()

    assert connections == 1
    assert [msg["text"] for msg in received] == ["第一句", "第二句"]


@pytest.mark.asyncio
async def test_stream_chat_raises_on_error_message() -> None:
    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "chat":
                await ws.send(json.dumps({"type": "error", "request_id": msg["request_id"], "message": "上游挂了"}))

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)
        with pytest.raises(RuntimeError, match="上游挂了"):
            async for _ in client.stream_chat([{"role": "user", "content": "你好"}]):
                pass


@pytest.mark.asyncio
async def test_stream_chat_cancellation_sends_cancel_without_closing_connection() -> None:
    cancel_received = asyncio.Event()
    disconnected = asyncio.Event()

    async def handler(ws):
        stream_task: asyncio.Task | None = None

        async def stream(rid: str) -> None:
            while True:
                await ws.send(json.dumps({"type": "delta", "request_id": rid, "text": "字"}))
                await asyncio.sleep(0.02)

        try:
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "chat":
                    stream_task = asyncio.create_task(stream(msg["request_id"]))
                elif msg.get("type") == "cancel":
                    cancel_received.set()
                    if stream_task is not None:
                        stream_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await stream_task
                    await ws.send(json.dumps({"type": "ack", "cmd": "cancel", "ok": True, "canceled": True}))
        finally:
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            disconnected.set()

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)

        async def consume():
            async for _chunk in client.stream_chat([{"role": "user", "content": "长回答"}]):
                await asyncio.sleep(0)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancel_received.wait(), timeout=2.0)
        assert not disconnected.is_set()
        await client.close()
        await asyncio.wait_for(disconnected.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_lollama_health_ping_pong() -> None:
    async def handler(ws):
        async for raw in ws:
            if json.loads(raw).get("type") == "ping":
                await ws.send(json.dumps({"type": "pong"}))

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)
        ok, detail = await client.health()
        assert ok
        assert "pong" in detail


@pytest.mark.asyncio
async def test_agent_status_invokes_on_status_callback() -> None:
    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "chat":
                rid = msg["request_id"]
                await ws.send(json.dumps({"type": "agent_status", "request_id": rid, "stage": "tool_start", "announce": "我算一下。"}))
                await ws.send(json.dumps({"type": "delta", "request_id": rid, "text": "42"}))
                await ws.send(json.dumps({"type": "done", "request_id": rid, "text": "42", "canceled": False}))

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url), timeout=2.0)
        received: list[dict] = []

        async def on_status(event: dict) -> None:
            received.append(event)

        chunks = [chunk async for chunk in client.stream_chat([{"role": "user", "content": "算"}], on_status=on_status)]
        assert "".join(chunks) == "42"
        assert len(received) == 1
        assert received[0]["stage"] == "tool_start"
        assert received[0]["announce"] == "我算一下。"


@pytest.mark.asyncio
async def test_agent_status_suppressed_when_announce_disabled() -> None:
    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("type") == "chat":
                rid = msg["request_id"]
                await ws.send(json.dumps({"type": "agent_status", "request_id": rid, "stage": "llm_waiting", "announce": "让我想想。"}))
                await ws.send(json.dumps({"type": "done", "request_id": rid, "text": "", "canceled": False}))

    async with fake_lollama(handler) as url:
        client = LollamaChatClient(LollamaConfig(url=url, announce_status=False), timeout=2.0)
        received: list[dict] = []

        async def on_status(event: dict) -> None:
            received.append(event)

        async for _ in client.stream_chat([{"role": "user", "content": "你好"}], on_status=on_status):
            pass
        assert received == []


def test_create_llm_client_by_provider() -> None:
    cfg = Config()
    assert isinstance(create_llm_client(cfg), OpenAICompatibleClient)
    cfg.llm.provider = "lollama"
    assert isinstance(create_llm_client(cfg), LollamaChatClient)


def test_example_config_has_llm_section() -> None:
    cfg = load_config(Path("configs/config.example.yaml"))
    assert cfg.llm.provider == "openai"
    assert cfg.lollama.url == "ws://127.0.0.1:8801/v1/llm/ws"
    assert cfg.services.lollama_dir == "../LoLLama"
