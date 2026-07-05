from __future__ import annotations

import asyncio
import logging
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from .adapters.stt import SttClient
from .adapters.tts import TtsClient
from .adapters.wake import WakeClient
from .audio import AudioSink
from .config import DuplexConfig, RuntimeConfig
from .models import Transcript
from .openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

SENTENCE_ENDINGS = set("。！？!?；;\n")

_STREAM_END = object()
SHORT_REPLY_PROMPT = (
    "回答必须简短自然，通常 1 到 2 句话。不要 Markdown、表情、长列表或客套铺垫；"
    "除非用户明确要求展开，否则直接给结论。"
)


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
    conversations: int = 0


class VoiceSession:
    """待唤醒 ⇄ 对话 的持久状态机。

    start_mode == "wake" 且 loop_forever 时：唤醒 → 对话 → (结束口令/空闲超时) →
    回到待唤醒，可无限循环，直到 stop() 或 Ctrl+C。
    """

    def __init__(
        self,
        *,
        duplex: DuplexConfig,
        wake: WakeClient,
        stt: SttClient,
        tts: TtsClient,
        llm: OpenAICompatibleClient | ChatModel,
        audio: AudioSink,
        wake_trigger_words: list[str] | None = None,
        runtime: RuntimeConfig | None = None,
    ) -> None:
        self.duplex = duplex
        self.wake = wake
        self.stt = stt
        self.tts = tts
        self.llm = llm
        self.audio = audio
        self.wake_trigger_words = [word.strip() for word in wake_trigger_words or [] if word.strip()]
        self.stats = SessionStats()
        self._history: list[dict[str, str]] = [{"role": "system", "content": duplex.system_prompt}]
        self._user_turn_cache: dict[int, dict[str, str]] = {}
        self._next_user_turn_id = 0
        self._current_response: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._session_end = asyncio.Event()
        self._last_activity = time.monotonic()
        runtime = runtime or RuntimeConfig()
        self._reconnect_initial = runtime.reconnect_initial_delay_sec
        self._reconnect_max = runtime.reconnect_max_delay_sec

    async def run(self) -> SessionStats:
        try:
            while not self._stop.is_set():
                if self.duplex.start_mode == "wake":
                    woke = await self._wait_for_wake()
                    if not woke:
                        break
                    with contextlib.suppress(Exception):
                        await self.wake.stop()
                    await self._start_stt_with_retry()
                else:
                    await self._start_stt_with_retry()

                self.stats.conversations += 1
                logger.info(
                    "conversation #%d started; waiting for stt transcripts (final_events_only may filter partials)",
                    self.stats.conversations,
                )
                ack_task = None
                if self.duplex.start_mode == "wake":
                    ack_task = asyncio.create_task(self._speak_wake_ack())
                reason = await self._conversation()
                if ack_task is not None:
                    ack_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await ack_task
                logger.info("conversation #%d ended: %s", self.stats.conversations, reason)
                await self.wait_for_idle()

                if self._stop.is_set() or reason in {"stream_end", "stopped"}:
                    break
                if self.duplex.start_mode != "wake" or not self.duplex.loop_forever:
                    break
                # 回到待唤醒状态
                with contextlib.suppress(Exception):
                    await self.stt.stop()
                if self.duplex.reset_history_per_session:
                    self._history = [{"role": "system", "content": self.duplex.system_prompt}]
                    self._clear_user_turn_cache()
                logger.info("returning to wake-word standby")
        finally:
            await self.stop()
        return self.stats

    def request_stop(self) -> None:
        """线程/信号安全的停止请求：让 run() 在下一个检查点退出。"""
        self._stop.set()

    async def stop(self) -> None:
        self._stop.set()
        await self._cancel_response()
        with contextlib.suppress(Exception):
            await self.audio.stop()
        with contextlib.suppress(Exception):
            await self.stt.stop()
        with contextlib.suppress(Exception):
            await self.wake.stop()

    # ------------------------------------------------------------------ wake

    async def _wait_for_wake(self) -> bool:
        """阻塞直到唤醒事件；连接断开时自动重连（指数退避）。返回 False 表示应停止。"""
        logger.info("waiting for wake word")
        delay = self._reconnect_initial
        while not self._stop.is_set():
            stream = self.wake.events()
            try:
                async for event in stream:
                    self.stats.wake_events += 1
                    logger.info("wake detected model=%s score=%.3f", event.model, event.score)
                    return True
                # 事件流正常结束但没有唤醒事件（如 disabled 客户端）
                logger.warning("wake event stream ended without wake event")
                return False
            except asyncio.CancelledError:
                raise
            except Exception:
                if not getattr(self.wake, "restart_on_stream_error", False):
                    raise
                logger.exception("wake event stream failed; reconnecting in %.1fs", delay)
                if await self._sleep_unless_stopped(delay):
                    return False
                delay = min(delay * 2, self._reconnect_max)
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()
        return False

    async def _speak_wake_ack(self) -> None:
        text = (self.duplex.wake_ack_text or "").strip()
        if not text:
            return
        logger.info("speaking wake acknowledgement: %s", text)
        try:
            async for chunk in self.tts.synthesize(text):
                await self.audio.play(chunk.pcm, sample_rate=chunk.sample_rate, channels=chunk.channels)
        except Exception:
            logger.exception("failed to speak wake acknowledgement")

    # ---------------------------------------------------------- conversation

    async def _conversation(self) -> str:
        """跑一轮完整对话，返回结束原因: end_word / idle / stream_end / stopped。"""
        self._session_end = asyncio.Event()
        self._touch_activity()
        queue: asyncio.Queue = asyncio.Queue()
        pump = asyncio.create_task(self._pump_transcripts(queue))
        reason = "stopped"
        tick = 1.0
        if self.duplex.idle_timeout_sec > 0:
            tick = min(1.0, max(0.05, self.duplex.idle_timeout_sec / 4))
        try:
            while not self._stop.is_set():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=tick)
                except TimeoutError:
                    if self._idle_expired():
                        reason = "idle"
                        break
                    if pump.done() and queue.empty():
                        reason = "stream_end"
                        break
                    continue
                if item is _STREAM_END:
                    reason = "stream_end"
                    break
                await self.handle_transcript(item)
                if self._session_end.is_set():
                    reason = "end_word"
                    break
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
        return reason

    async def _pump_transcripts(self, queue: asyncio.Queue) -> None:
        """持续读取 STT 转写并送入队列；流断开时按退避策略重连。"""
        delay = self._reconnect_initial
        while not self._stop.is_set() and not self._session_end.is_set():
            stream = self.stt.transcripts()
            try:
                async for transcript in stream:
                    delay = self._reconnect_initial
                    await queue.put(transcript)
                if not getattr(self.stt, "restart_on_stream_end", False):
                    await queue.put(_STREAM_END)
                    return
                logger.warning("stt transcript stream ended; restarting listener")
            except asyncio.CancelledError:
                raise
            except Exception:
                if not getattr(self.stt, "restart_on_stream_error", False):
                    logger.exception("stt transcript stream failed; not restartable")
                    await queue.put(_STREAM_END)
                    return
                logger.exception("stt transcript stream failed; restarting in %.1fs", delay)
            finally:
                with contextlib.suppress(Exception):
                    await stream.aclose()
            with contextlib.suppress(Exception):
                await self.stt.stop()
            if await self._sleep_unless_stopped(delay):
                return
            delay = min(delay * 2, self._reconnect_max)
            with contextlib.suppress(Exception):
                await self.stt.start()

    async def handle_transcript(self, transcript: Transcript) -> None:
        raw_text = transcript.text.strip()
        if not raw_text:
            return
        stt_text = self._strip_active_wake_trigger(raw_text)
        if not stt_text:
            logger.info("wake trigger ignored during active conversation")
            return
        text = self._strip_repeated_user_prefix(stt_text)
        if not text:
            logger.info("stt transcript ignored after turn splitting")
            return
        logger.info("user transcript kind=%s text=%s", transcript.kind.value, text)
        self._touch_activity()

        if not transcript.is_final:
            if self._current_response and not self._current_response.done():
                if self.duplex.allow_barge_in:
                    self.stats.interruptions += 1
                    logger.info("barge-in detected from partial transcript; canceling current assistant response")
                    await self._cancel_response()
                else:
                    logger.info("assistant is speaking; ignoring partial transcript while barge-in disabled")
            return

        if self._is_end_session(text):
            logger.info("end session phrase detected")
            self._session_end.set()
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
            print(f"\nUser: {text}", flush=True)
            self.stats.user_turns += 1
            self._remember_user_turn(stt_text, text)
            self._current_response = asyncio.create_task(self._respond(text))
            self._current_response.add_done_callback(self._on_response_done)

    async def wait_for_idle(self) -> None:
        task = self._current_response
        if task is not None:
            with contextlib.suppress(Exception):
                await task

    # -------------------------------------------------------------- response

    async def _respond(self, user_text: str) -> None:
        self._append_history("user", user_text)
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=8)
        full_text: list[str] = []
        messages = [*self._history, {"role": "system", "content": SHORT_REPLY_PROMPT}]

        async def produce_segments() -> None:
            buffer = SentenceBuffer()
            try:
                async for token in self._stream_llm(messages):
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
                try:
                    async for chunk in self.tts.synthesize(segment):
                        self.stats.tts_chunks += 1
                        await self.audio.play(chunk.pcm, sample_rate=chunk.sample_rate, channels=chunk.channels)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("tts/audio failed for segment; skipping: %s", segment[:40])

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce_segments())
                tg.create_task(consume_segments())
        except asyncio.CancelledError:
            if self.duplex.cancel_tts_on_user_speech:
                with contextlib.suppress(Exception):
                    await self.audio.stop()
            raise
        except Exception:
            logger.exception("assistant response failed")
        else:
            assistant_text = "".join(full_text).strip()
            if assistant_text:
                self._append_history("assistant", assistant_text)
                self.stats.assistant_turns += 1
                logger.info("assistant response complete chars=%d", len(assistant_text))
                print(f"\nAssistant: {assistant_text}", flush=True)

    async def _stream_llm(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """LLM 流式输出；若尚未产出任何 token 即失败，则重试一次。"""
        attempts = 2
        for attempt in range(attempts):
            received = False
            try:
                async for token in self.llm.stream_chat(messages):
                    received = True
                    yield token
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                if received or attempt == attempts - 1:
                    raise
                logger.exception("llm stream failed before any output; retrying once")
                await asyncio.sleep(0.5)

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

    def _on_response_done(self, task: asyncio.Task) -> None:
        self._touch_activity()
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception("assistant response task failed", exc_info=exc)

    # --------------------------------------------------------------- helpers

    def _touch_activity(self) -> None:
        self._last_activity = time.monotonic()

    def _idle_expired(self) -> bool:
        timeout = self.duplex.idle_timeout_sec
        if timeout <= 0:
            return False
        if self._current_response is not None and not self._current_response.done():
            return False
        return (time.monotonic() - self._last_activity) > timeout

    async def _sleep_unless_stopped(self, delay: float) -> bool:
        """等待 delay 秒；期间 stop 被触发则返回 True。"""
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        return self._stop.is_set()

    async def _start_stt_with_retry(self) -> None:
        delay = self._reconnect_initial
        while not self._stop.is_set():
            try:
                await self.stt.start()
                logger.info("stt listening started; conversation active")
                return
            except Exception:
                logger.exception("failed to start stt; retrying in %.1fs", delay)
                if await self._sleep_unless_stopped(delay):
                    return
                delay = min(delay * 2, self._reconnect_max)

    def _append_history(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})
        keep_messages = self.duplex.max_history_turns * 2
        if len(self._history) > keep_messages + 1:
            self._history = [self._history[0], *self._history[-keep_messages:]]

    def _is_end_session(self, text: str) -> bool:
        normalized = text.replace("，", "").replace("。", "").strip()
        return any(word and word in normalized for word in self.duplex.end_session_words)

    def _strip_active_wake_trigger(self, text: str) -> str:
        for word in sorted(self.wake_trigger_words, key=len, reverse=True):
            if _normalize_phrase(text) == _normalize_phrase(word):
                return ""
            if text.startswith(word):
                return _trim_wake_separators(text[len(word) :])
            if text.endswith(word):
                return _trim_wake_separators(text[: -len(word)])
        return text

    def _strip_repeated_user_prefix(self, text: str) -> str:
        if self._matches_cached_user_turn(text):
            logger.info("ignored repeated stt transcript text=%s", text)
            return ""
        previous = self._longest_cached_user_prefix(text)
        if not previous:
            return text
        stripped = _trim_wake_separators(text[len(previous) :])
        if stripped:
            logger.info("stripped repeated stt prefix previous=%s text=%s stripped=%s", previous, text, stripped)
            return stripped
        return text

    def _longest_cached_user_prefix(self, text: str) -> str:
        prefixes = (
            turn["stt_text"].strip()
            for turn in self._user_turn_cache.values()
            if turn.get("stt_text")
        )
        return max(
            (prefix for prefix in prefixes if len(text) > len(prefix) and text.startswith(prefix)),
            key=len,
            default="",
        )

    def _matches_cached_user_turn(self, text: str) -> bool:
        return any(
            text == turn["stt_text"].strip()
            for turn in self._user_turn_cache.values()
            if turn.get("stt_text")
        )

    def _remember_user_turn(self, stt_text: str, submitted_text: str) -> None:
        self._user_turn_cache[self._next_user_turn_id] = {
            "stt_text": stt_text,
            "submitted_text": submitted_text,
        }
        self._next_user_turn_id += 1
        max_cached_turns = max(4, self.duplex.max_history_turns * 2)
        while len(self._user_turn_cache) > max_cached_turns:
            oldest_turn_id = next(iter(self._user_turn_cache))
            del self._user_turn_cache[oldest_turn_id]

    def _clear_user_turn_cache(self) -> None:
        self._user_turn_cache.clear()
        self._next_user_turn_id = 0


def _normalize_phrase(text: str) -> str:
    return "".join(ch for ch in text.strip() if ch not in " \t\r\n,，.。!！?？、:：;；\"'“”‘’()（）[]【】")


def _trim_wake_separators(text: str) -> str:
    return text.strip(" \t\r\n,，.。!！?？、:：;；\"'“”‘’()（）[]【】")


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
