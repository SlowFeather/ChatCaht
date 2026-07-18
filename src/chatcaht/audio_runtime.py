from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass

import websockets

from .config import AudioRuntimeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    pcm: bytes
    stream_id: str
    speech_id: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    kind: str
    speech_id: str
    render_active: bool
    ts: float | None = None
    error: str | None = None


def _put_latest(queue: asyncio.Queue, item) -> None:
    try:
        queue.put_nowait(item)
        return
    except asyncio.QueueFull:
        pass
    with contextlib.suppress(asyncio.QueueEmpty):
        queue.get_nowait()
    with contextlib.suppress(asyncio.QueueFull):
        queue.put_nowait(item)


class AudioRouter:
    """Routes one AEC-clean capture stream to wake and STT consumers."""

    def __init__(self, cfg: AudioRuntimeConfig) -> None:
        capture_frames = max(1, cfg.capture_queue_ms // cfg.frame_ms)
        self.stream_id = uuid.uuid4().hex
        self._wake: asyncio.Queue[CaptureFrame | Exception] = asyncio.Queue(maxsize=capture_frames)
        self._stt: asyncio.Queue[CaptureFrame | Exception] = asyncio.Queue(maxsize=capture_frames)
        self._speech: asyncio.Queue[SpeechEvent] = asyncio.Queue(maxsize=32)
        self._preroll = deque(maxlen=max(1, cfg.barge_in_preroll_ms // cfg.frame_ms))
        self._stt_active = False
        self._render_active = False
        self._render_stream_id: str | None = None
        self._speech_id: str | None = None
        self._tail_seconds = cfg.aec_tail_ms / 1000.0
        self._playback_epoch = 0
        self._tail_task: asyncio.Task | None = None
        self.capture_drops = 0

    @property
    def render_active(self) -> bool:
        return self._render_active

    def activate_stt(self) -> None:
        self._stt_active = True
        self._speech_id = None
        self._preroll.clear()
        self._drain(self._stt)

    def deactivate_stt(self) -> None:
        self._stt_active = False
        self._speech_id = None
        self._preroll.clear()
        self._drain(self._stt)

    def playback_started(self, stream_id: str | None = None) -> None:
        if stream_id and self._render_active and stream_id == self._render_stream_id:
            return
        self._playback_epoch += 1
        self._render_active = True
        self._render_stream_id = stream_id
        self._speech_id = None
        self._preroll.clear()
        if self._tail_task is not None:
            self._tail_task.cancel()
            self._tail_task = None

    def playback_ended(self, stream_id: str | None = None) -> None:
        if stream_id and self._render_stream_id and stream_id != self._render_stream_id:
            return
        epoch = self._playback_epoch
        self._render_stream_id = None
        if self._tail_task is not None:
            self._tail_task.cancel()
        self._tail_task = asyncio.create_task(self._finish_tail(epoch))

    async def _finish_tail(self, epoch: int) -> None:
        try:
            await asyncio.sleep(self._tail_seconds)
            if epoch == self._playback_epoch:
                self._render_active = False
                self._render_stream_id = None
                self._speech_id = None
                self._preroll.clear()
        except asyncio.CancelledError:
            return

    def publish_capture(self, pcm: bytes) -> None:
        wake_frame = CaptureFrame(pcm=pcm, stream_id=self.stream_id)
        self._put_frame(self._wake, wake_frame)
        if not self._stt_active:
            return
        if self._render_active and self._speech_id is None:
            self._preroll.append(pcm)
            silence = bytes(len(pcm))
            self._put_frame(self._stt, CaptureFrame(silence, self.stream_id))
            return
        self._put_frame(self._stt, CaptureFrame(pcm, self.stream_id, self._speech_id))

    def publish_speech_event(self, event: SpeechEvent) -> None:
        if event.kind == "near_end_start" and event.render_active:
            self._speech_id = event.speech_id
            while self._preroll:
                self._put_frame(
                    self._stt,
                    CaptureFrame(self._preroll.popleft(), self.stream_id, event.speech_id),
                )
        elif event.kind == "near_end_end" and event.speech_id == self._speech_id:
            self._speech_id = None
        _put_latest(self._speech, event)

    def connection_lost(self, error: str) -> None:
        self._playback_epoch += 1
        self._render_active = False
        self._render_stream_id = None
        self._speech_id = None
        self._preroll.clear()
        self._drain(self._wake)
        self._drain(self._stt)
        _put_latest(self._wake, RuntimeError(error))
        _put_latest(self._stt, RuntimeError(error))
        if self._tail_task is not None:
            self._tail_task.cancel()
            self._tail_task = None
        _put_latest(self._speech, SpeechEvent("runtime_error", "", False, error=error))

    async def wake_frames(self) -> AsyncIterator[CaptureFrame]:
        while True:
            item = await self._wake.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def stt_frames(self) -> AsyncIterator[CaptureFrame]:
        while True:
            item = await self._stt.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def speech_events(self) -> AsyncIterator[SpeechEvent]:
        while True:
            yield await self._speech.get()

    def _put_frame(self, queue: asyncio.Queue[CaptureFrame | Exception], frame: CaptureFrame) -> None:
        before = queue.full()
        _put_latest(queue, frame)
        if before:
            self.capture_drops += 1

    @staticmethod
    def _drain(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return


class AudioRuntimeClient:
    """Strict client for the native, AEC-required full-duplex audio runtime."""

    is_unified_runtime = True

    def __init__(self, cfg: AudioRuntimeConfig, *, timeout: float = 5.0) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.router = AudioRouter(cfg)
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._waiters: dict[str, asyncio.Future[dict]] = {}
        self._playback_stream_id: str | None = None
        self._playback_started = False
        self._playback_format: tuple[int, int] | None = None
        self._closed = False

    async def start(self) -> None:
        await self._ensure_connection()

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                msg = json.loads(raw)
                if msg.get("type") != "status":
                    return False, f"audio runtime returned {msg.get('type')}"
                error = self._status_error(msg)
                if error:
                    return False, error
                return True, f"audio ready input={msg.get('input_device')} output={msg.get('output_device')}"
        except Exception as exc:
            return False, str(exc)

    async def begin_playback(self) -> None:
        if self._playback_stream_id is not None:
            raise RuntimeError("audio playback stream is already active")
        self._playback_stream_id = uuid.uuid4().hex
        self._playback_started = False
        self._playback_format = None

    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        if not pcm:
            return
        implicit_stream = False
        if self._playback_stream_id is None:
            self._playback_stream_id = uuid.uuid4().hex
            self._playback_started = False
            self._playback_format = None
            implicit_stream = True
        stream_id = self._playback_stream_id
        audio_format = (sample_rate, channels)
        if self._playback_format is not None and self._playback_format != audio_format:
            raise RuntimeError(
                "audio format changed within one playback stream: "
                f"{self._playback_format} -> {audio_format}"
            )
        async with self._send_lock:
            if not self._playback_started:
                await self._request(
                    "play_start",
                    stream_id=stream_id,
                    sample_rate=sample_rate,
                    channels=channels,
                )
                if self._playback_stream_id != stream_id:
                    await self._cancel_runtime_playback()
                    return
                self._playback_started = True
                self._playback_format = audio_format
            ws = await self._ensure_connection()
            await self._send_play_chunk(ws, stream_id=stream_id, pcm=pcm)
        if implicit_stream:
            await self.end_playback()

    async def end_playback(self) -> None:
        stream_id = self._playback_stream_id
        started = self._playback_started
        if stream_id is None:
            return
        try:
            if started:
                async with self._send_lock:
                    if self._playback_stream_id == stream_id:
                        await self._request("play_end", stream_id=stream_id)
        finally:
            if self._playback_stream_id == stream_id:
                self._clear_playback_state()

    async def cancel_playback(self) -> None:
        stream_id = self._playback_stream_id
        self._clear_playback_state()
        self.router.playback_ended(stream_id)
        await self._cancel_runtime_playback()

    async def _cancel_runtime_playback(self) -> None:
        # play_end deliberately waits for render drain on the streaming connection.
        # Use a separate control connection so barge-in can flush that render queue.
        async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
            status_raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            if isinstance(status_raw, bytes):
                raise RuntimeError("audio runtime did not send initial status")
            status = json.loads(status_raw)
            error = self._status_error(status)
            if error:
                raise RuntimeError(error)
            request_id = uuid.uuid4().hex
            await ws.send(json.dumps({"type": "play_cancel", "request_id": request_id}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                response = json.loads(raw)
                if response.get("request_id") == request_id:
                    if response.get("type") == "error" or not response.get("ok", True):
                        raise RuntimeError(str(response.get("message") or "play_cancel failed"))
                    break

    async def stop(self) -> None:
        await self.cancel_playback()

    async def close(self) -> None:
        self._closed = True
        reader = self._reader
        self._reader = None
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        error = RuntimeError("audio runtime client closed")
        for future in self._waiters.values():
            if not future.done():
                future.set_exception(error)
        self._waiters.clear()

    async def wake_frames(self) -> AsyncIterator[CaptureFrame]:
        await self._ensure_connection()
        async for frame in self.router.wake_frames():
            yield frame

    async def stt_frames(self) -> AsyncIterator[CaptureFrame]:
        await self._ensure_connection()
        async for frame in self.router.stt_frames():
            yield frame

    def speech_events(self) -> AsyncIterator[SpeechEvent]:
        return self.router.speech_events()

    def activate_stt(self) -> None:
        self.router.activate_stt()

    def deactivate_stt(self) -> None:
        self.router.deactivate_stt()

    async def _ensure_connection(self):
        if self._closed:
            raise RuntimeError("audio runtime client is closed")
        ws = self._ws
        if ws is not None and ws.close_code is None:
            return ws
        async with self._connect_lock:
            ws = self._ws
            if ws is not None and ws.close_code is None:
                return ws
            ws = await websockets.connect(
                self.cfg.url,
                open_timeout=self.timeout,
                ping_interval=20,
                ping_timeout=self.timeout,
                max_size=None,
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
            if isinstance(raw, bytes):
                await ws.close()
                raise RuntimeError("audio runtime did not send initial status")
            status = json.loads(raw)
            error = self._status_error(status)
            if status.get("type") != "status" or error:
                await ws.close()
                raise RuntimeError(error or "audio runtime did not send initial status")
            self._ws = ws
            self._reader = asyncio.create_task(self._read_loop(ws))
            logger.info("audio runtime connected: %s", self.cfg.url)
            return ws

    async def _request(self, typ: str, **payload) -> dict:
        ws = await self._ensure_connection()
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        try:
            await ws.send(json.dumps({"type": typ, "request_id": request_id, **payload}))
            response = await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._waiters.pop(request_id, None)
        if response.get("type") == "error" or not response.get("ok", True):
            raise RuntimeError(str(response.get("message") or response.get("code") or f"audio {typ} failed"))
        return response

    async def _send_play_chunk(self, ws, *, stream_id: str, pcm: bytes) -> None:
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._waiters[request_id] = future
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "play_chunk",
                        "request_id": request_id,
                        "stream_id": stream_id,
                        "bytes": len(pcm),
                    }
                )
            )
            await ws.send(pcm)
            response = await asyncio.wait_for(future, timeout=self.timeout)
        finally:
            self._waiters.pop(request_id, None)
        if response.get("type") == "error" or not response.get("ok", True):
            raise RuntimeError(str(response.get("message") or response.get("code") or "audio play_chunk failed"))

    async def _read_loop(self, ws) -> None:
        failure: str | None = None
        try:
            async for raw in ws:
                if isinstance(raw, bytes):
                    self.router.publish_capture(raw)
                    continue
                msg = json.loads(raw)
                request_id = str(msg.get("request_id") or "")
                waiter = self._waiters.get(request_id)
                if waiter is not None and not waiter.done():
                    waiter.set_result(msg)
                    continue
                typ = str(msg.get("type") or "")
                if typ in {"near_end_start", "near_end_end"}:
                    self.router.publish_speech_event(
                        SpeechEvent(
                            kind=typ,
                            speech_id=str(msg.get("speech_id") or uuid.uuid4().hex),
                            render_active=bool(msg.get("render_active")),
                            ts=float(msg["ts"]) if msg.get("ts") is not None else None,
                        )
                    )
                elif typ == "playback_started":
                    self.router.playback_started(str(msg.get("stream_id") or ""))
                elif typ == "playback_ended":
                    self.router.playback_ended(str(msg.get("stream_id") or "") or None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = str(exc)
            logger.error("audio runtime stream failed: %s", exc)
            for future in self._waiters.values():
                if not future.done():
                    future.set_exception(exc)
        finally:
            if self._ws is ws:
                self._ws = None
                self._clear_playback_state()
                if not self._closed:
                    self.router.connection_lost(failure or "audio runtime connection closed")

    def _clear_playback_state(self) -> None:
        self._playback_stream_id = None
        self._playback_started = False
        self._playback_format = None

    def _status_error(self, status: dict) -> str | None:
        if not status.get("ready") or not status.get("aec_ready"):
            return str(status.get("last_error") or "audio runtime/AEC is not ready")
        if status.get("protocol_version") != 2:
            return f"unsupported audio runtime protocol version: {status.get('protocol_version')!r}"
        expected = {
            "device_sample_rate": self.cfg.device_sample_rate,
            "capture_sample_rate": self.cfg.capture_sample_rate,
            "frame_ms": self.cfg.frame_ms,
            "render_queue_ms": self.cfg.render_queue_ms,
            "capture_queue_ms": self.cfg.capture_queue_ms,
            "capture_stall_timeout_ms": self.cfg.capture_stall_timeout_ms,
            "aec_tail_ms": self.cfg.aec_tail_ms,
            "barge_in_min_speech_ms": self.cfg.barge_in_min_speech_ms,
            "barge_in_hangover_ms": self.cfg.barge_in_hangover_ms,
            "vad_aggressiveness": self.cfg.vad_aggressiveness,
            "input_device": self.cfg.input_device,
            "output_device": self.cfg.output_device,
        }
        actual = status.get("config")
        if not isinstance(actual, dict):
            return "audio runtime did not report its effective config"
        mismatches = [
            f"{name}: expected {value!r}, got {actual.get(name)!r}"
            for name, value in expected.items()
            if actual.get(name) != value
        ]
        if mismatches:
            return "audio runtime config mismatch (" + "; ".join(mismatches) + ")"
        return None
