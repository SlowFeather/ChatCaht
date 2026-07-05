from __future__ import annotations

import asyncio

import pytest

from chatcaht.adapters.stt import MockSttClient
from chatcaht.adapters.tts import MockTtsClient
from chatcaht.adapters.wake import MockWakeClient
from chatcaht.audio import NullAudioSink
from chatcaht.config import Config
from chatcaht.models import Transcript, TranscriptKind
from chatcaht.orchestrator import ChatModel, SentenceBuffer, VoiceSession
from chatcaht.selftest import ScriptedModel, run_selftest


def test_sentence_buffer_flushes_on_punctuation() -> None:
    buf = SentenceBuffer(min_chars=2, max_chars=20)
    assert buf.push("你好。继续") == ["你好。"]
    assert buf.flush() == "继续"


def test_sentence_buffer_flushes_on_max_chars() -> None:
    buf = SentenceBuffer(min_chars=3, max_chars=5)
    assert buf.push("abcdef") == ["abcde"]
    assert buf.flush() == "f"


@pytest.mark.asyncio
async def test_selftest_chain_passes() -> None:
    cfg = Config()
    results = await run_selftest(cfg)
    assert all(ok for _, ok, _ in results)


@pytest.mark.asyncio
async def test_barge_in_cancels_current_response() -> None:
    cfg = Config()
    model = ScriptedModel(["慢一点。", "还没结束。"], delay=0.05)
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("第一个问题", TranscriptKind.FINAL))
    await asyncio.sleep(0.01)
    await session.handle_transcript(Transcript("第二个问题", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert session.stats.interruptions == 1
    assert model.calls == 2
    assert session.stats.assistant_turns == 1


@pytest.mark.asyncio
async def test_partial_barge_in_cancels_without_starting_new_response() -> None:
    cfg = Config()
    model = ScriptedModel(["slow first response", "second response"], delay=0.05)
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("first question", TranscriptKind.FINAL))
    await asyncio.sleep(0.01)
    await session.handle_transcript(Transcript("second question", TranscriptKind.PARTIAL))
    await asyncio.sleep(0.01)

    assert session.stats.interruptions == 1
    assert model.calls == 1
    assert session.stats.user_turns == 1

    await session.handle_transcript(Transcript("second question", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert model.calls == 2
    assert session.stats.user_turns == 2
    assert session.stats.assistant_turns == 1


@pytest.mark.asyncio
async def test_session_handles_100_turns_without_history_growth() -> None:
    cfg = Config()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
    )

    for index in range(100):
        await session.handle_transcript(Transcript(f"turn {index}", TranscriptKind.FINAL))
        await session.wait_for_idle()

    assert session.stats.user_turns == 100
    assert session.stats.assistant_turns == 100
    assert len(session._history) <= cfg.duplex.max_history_turns * 2 + 1


@pytest.mark.asyncio
async def test_response_failure_does_not_stop_later_turns() -> None:
    cfg = Config()
    cfg.duplex.start_mode = "manual"
    model = FailingOnceModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient(["first", "second"], turn_delay=0.01),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert model.calls == 2
    assert stats.user_turns == 2
    assert stats.assistant_turns == 1


@pytest.mark.asyncio
async def test_stt_stream_restarts_after_disconnect() -> None:
    cfg = Config()
    cfg.duplex.start_mode = "manual"
    stt = FlakySttClient()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=stt,
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert stt.starts == 2
    assert stats.user_turns == 1
    assert stats.assistant_turns == 1


@pytest.mark.asyncio
async def test_wake_trigger_is_ignored_during_active_conversation() -> None:
    cfg = Config()
    model = ScriptedModel(["ok."])
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
        wake_trigger_words=["小元", "你好小元"],
    )

    await session.handle_transcript(Transcript("小元", TranscriptKind.FINAL))
    await session.handle_transcript(Transcript("你好小元。", TranscriptKind.FINAL))

    assert model.calls == 0
    assert session.stats.user_turns == 0
    assert session.stats.assistant_turns == 0


@pytest.mark.asyncio
async def test_wake_trigger_prefix_is_removed_during_active_conversation() -> None:
    cfg = Config()
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
        wake_trigger_words=["小元", "你好小元"],
    )

    await session.handle_transcript(Transcript("小元，帮我查一下天气", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert model.calls == 1
    assert model.last_user_message == "帮我查一下天气"
    assert session.stats.user_turns == 1


@pytest.mark.asyncio
async def test_repeated_user_prefix_is_removed_from_stt_carryover() -> None:
    cfg = Config()
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("tomato or potato", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("tomato or potatowater or cola", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert model.calls == 2
    assert model.last_user_message == "water or cola"


@pytest.mark.asyncio
async def test_accumulated_stt_finals_are_split_by_cached_turns() -> None:
    cfg = Config()
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("cola or sprite", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("cola or spritewatermelon", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("cola or spritewatermelon or cucumber", TranscriptKind.FINAL))
    await session.wait_for_idle()

    user_messages = [
        message["content"]
        for message in model.last_messages
        if message["role"] == "user"
    ]
    assert model.calls == 3
    assert user_messages == ["cola or sprite", "watermelon", "or cucumber"]


@pytest.mark.asyncio
async def test_duplicate_stt_final_is_ignored() -> None:
    cfg = Config()
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("same question", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("same question", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert model.calls == 1
    assert session.stats.user_turns == 1


class FailingOnceModel(ChatModel):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages: list[dict[str, str]]):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("llm failed")
        yield "recovered."


class FlakySttClient(MockSttClient):
    restart_on_stream_error = True

    def __init__(self) -> None:
        super().__init__([])
        self.starts = 0

    async def start(self) -> None:
        self.starts += 1
        await super().start()

    async def transcripts(self):
        if self.starts == 1:
            raise RuntimeError("stt disconnected")
        yield Transcript("after reconnect", TranscriptKind.FINAL)
        await self.stop()


class CapturingModel(ChatModel):
    def __init__(self) -> None:
        self.calls = 0
        self.last_user_message = ""
        self.last_messages: list[dict[str, str]] = []

    async def stream_chat(self, messages: list[dict[str, str]]):
        self.calls += 1
        self.last_messages = messages
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        self.last_user_message = user_messages[-1]
        yield "ok."
