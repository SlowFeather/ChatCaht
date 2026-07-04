from __future__ import annotations

import asyncio

import pytest

from chatcaht.adapters.stt import MockSttClient
from chatcaht.adapters.tts import MockTtsClient
from chatcaht.adapters.wake import MockWakeClient
from chatcaht.audio import NullAudioSink
from chatcaht.config import Config, RuntimeConfig
from chatcaht.models import Transcript, TranscriptKind, WakeEvent
from chatcaht.orchestrator import VoiceSession
from chatcaht.selftest import ScriptedModel


class LimitedWakeClient(MockWakeClient):
    """只提供固定次数的唤醒事件，超过后事件流直接结束（让测试收敛）。"""

    def __init__(self, wakes: int) -> None:
        super().__init__()
        self.max_wakes = wakes
        self.count = 0

    async def events(self):
        if self.count >= self.max_wakes:
            return
        self.count += 1
        yield WakeEvent(model="mock", score=1.0)


class FlakyWakeClient(LimitedWakeClient):
    """第一次事件流抛异常，验证唤醒监听自动重连。"""

    restart_on_stream_error = True

    def __init__(self) -> None:
        super().__init__(wakes=1)
        self.attempts = 0

    async def events(self):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("wake ws disconnected")
        async for event in super().events():
            yield event


class ScriptedConversationsStt(MockSttClient):
    """每次 start() 进入下一段对话脚本。"""

    def __init__(self, conversations: list[list[str]]) -> None:
        super().__init__([])
        self.conversations = conversations
        self.session_index = -1

    async def start(self) -> None:
        self.session_index += 1
        await super().start()

    async def transcripts(self):
        if 0 <= self.session_index < len(self.conversations):
            texts = self.conversations[self.session_index]
        else:
            texts = []
        for index, text in enumerate(texts):
            await asyncio.sleep(0)
            yield Transcript(text, TranscriptKind.FINAL, source="mock", segment_id=index)
            await asyncio.sleep(0.02)


class SilentSttClient(MockSttClient):
    """连接保持打开但一直没有语音，用于验证空闲超时。"""

    async def transcripts(self):
        await asyncio.sleep(30)
        if False:
            yield Transcript(text="", kind=TranscriptKind.FINAL)


def _fast_runtime() -> RuntimeConfig:
    return RuntimeConfig(reconnect_initial_delay_sec=0.01, reconnect_max_delay_sec=0.05)


@pytest.mark.asyncio
async def test_end_word_returns_to_wake_standby_and_wakes_again() -> None:
    cfg = Config()
    cfg.duplex.loop_forever = True
    wake = LimitedWakeClient(wakes=2)
    stt = ScriptedConversationsStt([["退出"], ["你好"]])
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=wake,
        stt=stt,
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
        runtime=_fast_runtime(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert stats.wake_events == 2
    assert stats.conversations == 2
    assert stats.user_turns == 1  # "退出" 不算用户轮次
    assert stats.assistant_turns == 1


@pytest.mark.asyncio
async def test_idle_timeout_returns_to_wake_standby() -> None:
    cfg = Config()
    cfg.duplex.loop_forever = True
    cfg.duplex.idle_timeout_sec = 0.2
    wake = LimitedWakeClient(wakes=2)
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=wake,
        stt=SilentSttClient(),
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
        runtime=_fast_runtime(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert stats.wake_events == 2
    assert stats.conversations == 2
    assert stats.user_turns == 0


@pytest.mark.asyncio
async def test_wake_stream_reconnects_after_error() -> None:
    cfg = Config()
    cfg.duplex.loop_forever = False
    wake = FlakyWakeClient()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=wake,
        stt=ScriptedConversationsStt([["退出"]]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
        runtime=_fast_runtime(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert wake.attempts == 2
    assert stats.wake_events == 1


@pytest.mark.asyncio
async def test_wake_ack_is_spoken_after_wake() -> None:
    cfg = Config()
    cfg.duplex.loop_forever = False
    cfg.duplex.wake_ack_text = "我在"
    audio = NullAudioSink()
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=LimitedWakeClient(wakes=1),
        stt=ScriptedConversationsStt([["退出"]]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=audio,
        runtime=_fast_runtime(),
    )

    await asyncio.wait_for(session.run(), timeout=5)

    assert audio.bytes_played > 0


@pytest.mark.asyncio
async def test_history_reset_per_session() -> None:
    cfg = Config()
    cfg.duplex.loop_forever = True
    cfg.duplex.reset_history_per_session = True
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=LimitedWakeClient(wakes=2),
        stt=ScriptedConversationsStt([["记住这句话", "退出"], ["你好"]]),
        tts=MockTtsClient(),
        llm=ScriptedModel(["ok."]),
        audio=NullAudioSink(),
        runtime=_fast_runtime(),
    )

    stats = await asyncio.wait_for(session.run(), timeout=5)

    assert stats.conversations == 2
    # 第二段对话开始前历史被重置：system + 第二段的 user/assistant
    texts = [message["content"] for message in session._history]
    assert all("记住这句话" not in text for text in texts)
