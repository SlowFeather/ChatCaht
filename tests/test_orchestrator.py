from __future__ import annotations

import asyncio
import json
import logging

import pytest

from chatcaht.adapters.stt import MockSttClient
from chatcaht.adapters.tts import MockTtsClient
from chatcaht.adapters.wake import MockWakeClient
from chatcaht.audio import NullAudioSink
from chatcaht.audio_runtime import SpeechEvent
from chatcaht.config import Config
from chatcaht.models import Transcript, TranscriptKind
from chatcaht.metrics import MetricsRecorder
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
async def test_unpunctuated_reply_flushes_after_timeout_and_records_metrics(tmp_path) -> None:
    cfg = Config()
    cfg.duplex.tts_segment_min_chars = 3
    cfg.duplex.tts_segment_flush_ms = 20
    tts = CapturingTts()
    metrics_path = tmp_path / "metrics.jsonl"
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=tts,
        llm=PausingModel(),
        audio=NullAudioSink(),
        metrics=MetricsRecorder(metrics_path),
    )

    await session.handle_transcript(Transcript("test", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert tts.texts == ["abcdef", "gh"]
    events = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    names = {event["metric"] for event in events}
    assert {"asr_final", "llm_first_token", "tts_first_pcm", "playback_start", "turn_total"} <= names


@pytest.mark.asyncio
async def test_assistant_response_uses_one_continuous_playback_stream() -> None:
    cfg = Config()
    cfg.duplex.tts_segment_min_chars = 2
    audio = RecordingAudio()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["第一句。", "第二句。"]),
        audio=audio,
    )

    await session.handle_transcript(Transcript("test", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert audio.events[0] == "begin"
    assert audio.events[-1] == "end"
    assert audio.events.count("begin") == 1
    assert audio.events.count("end") == 1
    assert audio.events.count("chunk") == 2


@pytest.mark.asyncio
async def test_selftest_chain_passes() -> None:
    cfg = Config()
    results = await run_selftest(cfg)
    assert all(ok for _, ok, _ in results)


@pytest.mark.asyncio
async def test_session_stop_closes_tts_client() -> None:
    cfg = Config()
    tts = ClosingTts()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=tts,
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
    )

    await session.stop()

    assert tts.closed


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
async def test_unified_audio_requires_confirmed_near_end_for_barge_in() -> None:
    cfg = Config()
    audio = UnifiedNullAudio()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["a long response that remains active long enough"], delay=0.05),
        audio=audio,
    )
    await session.handle_transcript(Transcript("question", TranscriptKind.FINAL))
    await asyncio.sleep(0.01)

    await session.handle_transcript(Transcript("speaker echo", TranscriptKind.PARTIAL))
    assert session._current_response is not None
    assert not session._current_response.done()

    await session._handle_speech_event(SpeechEvent("near_end_start", "speech-1", True))
    assert session.stats.interruptions == 1
    assert audio.cancellations == 1
    assert session._current_response is None


@pytest.mark.asyncio
async def test_unified_audio_ignores_playback_speech_when_barge_in_is_disabled() -> None:
    cfg = Config()
    cfg.duplex.allow_barge_in = False
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["must not run"]),
        audio=UnifiedNullAudio(),
    )

    await session._handle_speech_event(SpeechEvent("near_end_start", "speech-1", True))
    await session.handle_transcript(
        Transcript("ignored during playback", TranscriptKind.FINAL, raw={"speech_id": "speech-1"})
    )

    assert session.stats.user_turns == 0
    assert session.stats.interruptions == 0


@pytest.mark.asyncio
async def test_unified_audio_runtime_failure_terminates_session() -> None:
    cfg = Config()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["unused"]),
        audio=UnifiedNullAudio(),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        await session._handle_speech_event(SpeechEvent("runtime_error", "", False, error="capture failed"))


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
async def test_end_session_words_require_standalone_phrase() -> None:
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

    await session.handle_transcript(Transcript("不要跟我说再见", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert not session._session_end.is_set()
    assert model.calls == 1
    assert session.stats.user_turns == 1

    await session.handle_transcript(Transcript("再见", TranscriptKind.FINAL))

    assert session._session_end.is_set()
    assert model.calls == 1
    assert session.stats.user_turns == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phrase", ["再见", "再见。", "拜拜", "闭嘴"])
async def test_end_session_words_accept_configured_standalone_phrases(phrase: str) -> None:
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

    await session.handle_transcript(Transcript(phrase, TranscriptKind.FINAL))

    assert session._session_end.is_set()
    assert model.calls == 0
    assert session.stats.user_turns == 0


@pytest.mark.asyncio
async def test_end_session_word_does_not_enter_user_turn_cache() -> None:
    cfg = Config()
    cfg.duplex.end_session_words = ["关闭"]
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("关闭", TranscriptKind.FINAL))

    assert session._session_end.is_set()
    assert session._user_turn_cache == {}
    assert model.calls == 0
    assert session.stats.user_turns == 0


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
    assert session._user_turn_cache[1]["stt_text"] == "water or cola"
    assert session._user_turn_cache[1]["raw_stt_text"] == "tomato or potatowater or cola"


@pytest.mark.asyncio
async def test_stripped_stt_prefix_info_log_only_shows_current_turn(caplog) -> None:
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

    caplog.set_level(logging.INFO, logger="chatcaht.orchestrator")
    await session.handle_transcript(Transcript("tomato or potato", TranscriptKind.FINAL))
    await session.wait_for_idle()
    caplog.clear()
    await session.handle_transcript(Transcript("tomato or potatowater or cola", TranscriptKind.FINAL))
    await session.wait_for_idle()

    info_logs = "\n".join(record.getMessage() for record in caplog.records if record.levelno == logging.INFO)
    assert "tomato or potato" not in info_logs
    assert "tomato or potatowater or cola" not in info_logs
    assert "stripped=water or cola" in info_logs
    assert "user transcript kind=final text=water or cola" in info_logs


@pytest.mark.asyncio
async def test_end_session_word_is_discarded_from_stt_carryover() -> None:
    cfg = Config()
    cfg.duplex.end_session_words = ["关闭"]
    model = CapturingModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
        wake_trigger_words=["小源"],
    )

    await session.handle_transcript(Transcript("你觉得这个夏天吃什么好", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("你觉得这个夏天吃什么好关闭小源你觉得九九感冒灵好不好", TranscriptKind.FINAL))
    await session.wait_for_idle()

    assert model.calls == 2
    assert model.last_user_message == "你觉得九九感冒灵好不好"
    assert session._user_turn_cache[1]["stt_text"] == "你觉得九九感冒灵好不好"
    assert session._user_turn_cache[1]["raw_stt_text"] == "你觉得这个夏天吃什么好关闭小源你觉得九九感冒灵好不好"


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
async def test_backend_managed_history_receives_only_current_turn() -> None:
    cfg = Config()
    model = BackendHistoryModel()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=model,
        audio=NullAudioSink(),
    )

    await session.handle_transcript(Transcript("第一句", TranscriptKind.FINAL))
    await session.wait_for_idle()
    await session.handle_transcript(Transcript("第二句", TranscriptKind.FINAL))
    await session.wait_for_idle()

    user_messages = [
        message["content"]
        for message in model.last_messages
        if message["role"] == "user"
    ]
    assert user_messages == ["第二句"]
    assert model.last_messages == [{"role": "user", "content": "第二句"}]
    assert all("backend ok" not in message["content"] for message in session._history)


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


class BackendHistoryModel(CapturingModel):
    manages_conversation_history = True

    async def stream_chat(self, messages: list[dict[str, str]]):
        self.calls += 1
        self.last_messages = messages
        user_messages = [message["content"] for message in messages if message["role"] == "user"]
        self.last_user_message = user_messages[-1]
        yield "backend ok."


class PausingModel(ChatModel):
    async def stream_chat(self, messages: list[dict[str, str]]):
        yield "abcdef"
        await asyncio.sleep(0.06)
        yield "gh"


class CapturingTts:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str):
        self.texts.append(text)
        yield type("Chunk", (), {"pcm": b"\x00\x00", "sample_rate": 16000, "channels": 1})()


class ClosingTts(MockTtsClient):
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class UnifiedNullAudio(NullAudioSink):
    is_unified_runtime = True

    def __init__(self) -> None:
        super().__init__()
        self.cancellations = 0

    async def cancel_playback(self) -> None:
        self.cancellations += 1


class RecordingAudio(NullAudioSink):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    async def begin_playback(self) -> None:
        self.events.append("begin")

    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        self.events.append("chunk")
        await super().play(pcm, sample_rate=sample_rate, channels=channels)

    async def end_playback(self) -> None:
        self.events.append("end")
