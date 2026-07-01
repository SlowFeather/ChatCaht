from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .adapters.stt import SttClient
from .adapters.tts import TtsClient
from .adapters.wake import WakeClient
from .audio import AudioSink
from .config import DuplexConfig
from .lmstudio import LmStudioClient
from .models import Transcript

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = set("。！？!?；;\n")


class ChatModel:
    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        raise NotImplementedError


@dataclass(slots=True)
class SessionStats:
    wake_events: int = 0
    user_turns: int = 0
    assistant_turns: int = 0
    interruptions: int = 0
    tts_chunks: int = 0


class VoiceSession:
    def __init__(
        self,
        *,
        duplex: DuplexConfig,
        wake: WakeClient,
        stt: SttClient,
        tts: TtsClient,
        llm: LmStudioClient | ChatModel,
        audio: AudioSink,
    ) -> None:
        self.duplex = duplex
        self.wake = wake
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.audio = audio
        self.stats = SessionStats()
        self._history: list[dict[str, str]] = [{"role": "system", "content": duplex.system_prompt}]
        self._current_response: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def run(self) -> SessionStats:
        if self.duplex.start_mode == "wake":
            await self._wait_for_wake()
        else:
            await self.stt.start()

        try:
            async for transcript in self.stt.transcripts():
                if self._stop.is_set():
                    break
                await self.handle_transcript(transcript)
            if not self._stop.is_set():
                await self.wait_for_idle()
        finally:
            await self.stop()
        return self.stats

    async def stop(self) -> None:
        self._stop.set()
        await self._cancel_response()
        await self.audio.stop()
        await self.stt.stop()
        await self.wake.stop()

    async def handle_transcript(self, transcript: Transcript) -> None:
        text = transcript.text.strip()
        if not text:
            return
        logger.info("user transcript kind=%s text=%s", transcript.kind.value, text)

        if self._is_end_session(text):
            logger.info("end session phrase detected")
            self._stop.set()
            return

        if self._current_response and not self._current_response.done():
            if self.duplex.allow_barge_in:
                self.stats.interruptions += 1
                logger.info("barge-in detected; canceling current assistant response")
                await self._cancel_response()
            else:
                logger.info("assistant is speaking; ignoring transcript while barge-in disabled")
                return

        if transcript.is_final:
            self.stats.user_turns += 1
            self._current_response = asyncio.create_task(self._respond(text))

    async def wait_for_idle(self) -> None:
        task = self._current_response
        if task is not None:
            await task

    async def _wait_for_wake(self) -> None:
        logger.info("waiting for wake word")
        async for event in self.wake.events():
            self.stats.wake_events += 1
            logger.info("wake detected model=%s score=%.3f", event.model, event.score)
            await self.wake.stop()
            await self.stt.start()
            return

    async def _respond(self, user_text: str) -> None:
        self._append_history("user", user_text)
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        full_text: list[str] = []

        async def produce_segments() -> None:
            buffer = SentenceBuffer()
            try:
                async for token in self.llm.stream_chat(list(self._history)):
                    full_text.append(token)
                    for segment in buffer.push(token):
                        await queue.put(segment)
                tail = buffer.flush()
                if tail:
                    await queue.put(tail)
            finally:
                await queue.put(None)

        async def consume_segments() -> None:
            while True:
                segment = await queue.get()
                if segment is None:
                    return
                async for chunk in self.tts.synthesize(segment):
                    self.stats.tts_chunks += 1
                    await self.audio.play(chunk.pcm, sample_rate=chunk.sample_rate, channels=chunk.channels)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce_segments())
                tg.create_task(consume_segments())
        except asyncio.CancelledError:
            if self.duplex.cancel_tts_on_user_speech:
                await self.audio.stop()
            raise
        except Exception:
            logger.exception("assistant response failed")
            raise
        else:
            assistant_text = "".join(full_text).strip()
            if assistant_text:
                self._append_history("assistant", assistant_text)
                self.stats.assistant_turns += 1
                logger.info("assistant response complete chars=%d", len(assistant_text))

    async def _cancel_response(self) -> None:
        task = self._current_response
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._current_response = None

    def _append_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        keep_messages = self.duplex.max_history_turns * 2
        if len(self._history) > keep_messages + 1:
            self._history = [self._history[0], *self._history[-keep_messages:]]

    def _is_end_session(self, text: str) -> bool:
        normalized = text.replace("，", "").replace("。", "").strip()
        return any(word and word in normalized for word in self.duplex.end_session_words)


class SentenceBuffer:
    def __init__(self, *, min_chars: int = 8, max_chars: int = 80) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf: list[str] = []

    def push(self, token: str) -> list[str]:
        segments: list[str] = []
        for ch in token:
            self._buf.append(ch)
            if self._should_flush(ch):
                segments.append(self.flush())
        return [s for s in segments if s]

    def flush(self) -> str:
        text = "".join(self._buf).strip()
        self._buf.clear()
        return text

    def _should_flush(self, ch: str) -> bool:
        size = len(self._buf)
        if size >= self.max_chars:
            return True
        return size >= self.min_chars and ch in SENTENCE_ENDINGS
