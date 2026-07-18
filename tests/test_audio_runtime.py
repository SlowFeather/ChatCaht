import asyncio
import json

import pytest
import websockets

from chatcaht.audio_runtime import AudioRouter, AudioRuntimeClient, SpeechEvent
from chatcaht.config import AudioRuntimeConfig


@pytest.mark.asyncio
async def test_router_suppresses_render_echo_until_near_end_is_confirmed() -> None:
    cfg = AudioRuntimeConfig(frame_ms=10, barge_in_preroll_ms=20, aec_tail_ms=10)
    router = AudioRouter(cfg)
    router.activate_stt()
    router.playback_started()

    echo = b"\x01\x00" * 160
    router.publish_capture(echo)
    suppressed = await anext(router.stt_frames())
    assert suppressed.pcm == bytes(len(echo))
    assert suppressed.speech_id is None

    event = SpeechEvent("near_end_start", "speech-1", True)
    router.publish_speech_event(event)
    preroll = await anext(router.stt_frames())
    assert preroll.pcm == echo
    assert preroll.speech_id == "speech-1"
    assert await anext(router.speech_events()) == event

    router.publish_capture(echo)
    live = await anext(router.stt_frames())
    assert live.pcm == echo
    assert live.speech_id == "speech-1"


@pytest.mark.asyncio
async def test_router_queues_are_bounded_and_drop_oldest() -> None:
    cfg = AudioRuntimeConfig(frame_ms=10, capture_queue_ms=1000)
    router = AudioRouter(cfg)
    frame = bytes(320)
    for index in range(150):
        router.publish_capture(index.to_bytes(2, "little") + frame[2:])

    assert router._wake.qsize() == 100
    assert router.capture_drops == 50
    first = await anext(router.wake_frames())
    assert int.from_bytes(first.pcm[:2], "little") == 50

    small = AudioRouter(AudioRuntimeConfig(frame_ms=10, capture_queue_ms=20))
    for index in range(3):
        small.publish_capture(index.to_bytes(2, "little") + frame[2:])
    assert small._wake.qsize() == 2
    assert int.from_bytes((await anext(small.wake_frames())).pcm[:2], "little") == 1


@pytest.mark.asyncio
async def test_playback_tail_remains_active_then_closes() -> None:
    router = AudioRouter(AudioRuntimeConfig(aec_tail_ms=10))
    router.playback_started()
    router.playback_ended()
    assert router.render_active
    await asyncio.sleep(0.02)
    assert not router.render_active


@pytest.mark.asyncio
async def test_router_connection_loss_unblocks_capture_consumers() -> None:
    router = AudioRouter(AudioRuntimeConfig())
    router.connection_lost("device restart required")

    with pytest.raises(RuntimeError, match="device restart required"):
        await anext(router.wake_frames())
    with pytest.raises(RuntimeError, match="device restart required"):
        await anext(router.stt_frames())


def test_one_capture_generation_survives_100_wake_stt_route_switches() -> None:
    router = AudioRouter(AudioRuntimeConfig())
    stream_id = router.stream_id
    for _ in range(100):
        router.activate_stt()
        router.deactivate_stt()
    assert router.stream_id == stream_id


@pytest.mark.asyncio
async def test_audio_runtime_streams_multiple_chunks_between_one_start_and_end() -> None:
    cfg = AudioRuntimeConfig()
    commands: list[str] = []
    chunks: list[bytes] = []
    pending_request_id = ""

    async def handler(ws):
        nonlocal pending_request_id
        await ws.send(json.dumps(_status(cfg)))
        async for raw in ws:
            if isinstance(raw, bytes):
                chunks.append(raw)
                await ws.send(json.dumps({"type": "ack", "ok": True, "request_id": pending_request_id}))
                continue
            message = json.loads(raw)
            typ = message["type"]
            commands.append(typ)
            if typ == "play_chunk":
                pending_request_id = message["request_id"]
                continue
            await ws.send(json.dumps({"type": "ack", "ok": True, "request_id": message["request_id"]}))
            if typ == "play_start":
                await ws.send(json.dumps({"type": "playback_started", "stream_id": message["stream_id"]}))
            elif typ == "play_end":
                await ws.send(json.dumps({"type": "playback_ended", "stream_id": message["stream_id"]}))

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg.url = f"ws://127.0.0.1:{port}/v1/audio/ws"
    client = AudioRuntimeClient(cfg, timeout=2.0)
    try:
        await client.begin_playback()
        await client.play(b"\x01\x00", sample_rate=16000)
        await client.play(b"\x02\x00", sample_rate=16000)
        assert commands == ["play_start", "play_chunk", "play_chunk"]
        await client.end_playback()
        assert commands == ["play_start", "play_chunk", "play_chunk", "play_end"]
        assert chunks == [b"\x01\x00", b"\x02\x00"]
    finally:
        await client.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_audio_runtime_health_rejects_effective_config_drift() -> None:
    cfg = AudioRuntimeConfig()

    async def handler(ws):
        status = _status(cfg)
        status["config"]["aec_tail_ms"] += 1
        await ws.send(json.dumps(status))
        await ws.wait_closed()

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cfg.url = f"ws://127.0.0.1:{port}/v1/audio/ws"
    try:
        ok, detail = await AudioRuntimeClient(cfg, timeout=2.0).health()
        assert not ok
        assert "aec_tail_ms" in detail
        assert "config mismatch" in detail
    finally:
        server.close()
        await server.wait_closed()


def _status(cfg: AudioRuntimeConfig) -> dict:
    return {
        "type": "status",
        "protocol_version": 2,
        "ready": True,
        "aec_ready": True,
        "input_device": "test input",
        "output_device": "test output",
        "config": {
            "device_sample_rate": cfg.device_sample_rate,
            "capture_sample_rate": cfg.capture_sample_rate,
            "frame_ms": cfg.frame_ms,
            "render_queue_ms": cfg.render_queue_ms,
            "capture_queue_ms": cfg.capture_queue_ms,
            "capture_stall_timeout_ms": cfg.capture_stall_timeout_ms,
            "aec_tail_ms": cfg.aec_tail_ms,
            "barge_in_min_speech_ms": cfg.barge_in_min_speech_ms,
            "barge_in_hangover_ms": cfg.barge_in_hangover_ms,
            "vad_aggressiveness": cfg.vad_aggressiveness,
            "input_device": cfg.input_device,
            "output_device": cfg.output_device,
        },
    }
