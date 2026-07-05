from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from chatcaht.adapters.stt import MockSttClient
from chatcaht.adapters.tts import TtsClient
from chatcaht.adapters.wake import MockWakeClient
from chatcaht.audio import NullAudioSink
from chatcaht.config import DuplexConfig
from chatcaht.models import Transcript, TranscriptKind, TtsChunk
from chatcaht.orchestrator import VoiceSession


class RecordingTts(TtsClient):
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def health(self) -> tuple[bool, str]:
        return True, "recording tts ready"

    async def synthesize(self, text: str) -> AsyncIterator[TtsChunk]:
        self.spoken.append(text)
        await asyncio.sleep(0)
        yield TtsChunk(pcm=text.encode("utf-8"), sample_rate=16000)


class StatusAnnouncingModel:
    """模拟 LoLLama 客户端：先回调状态播报，再流式输出正文。"""

    supports_status_events = True

    def __init__(self) -> None:
        self.calls = 0

    async def stream_chat(self, messages, *, on_status=None) -> AsyncIterator[str]:
        self.calls += 1
        if on_status is not None:
            await on_status({"stage": "tool_start", "announce": "我算一下。"})
            await on_status({"stage": "llm_first_token", "announce": ""})
        yield "答案是42。"


@pytest.mark.asyncio
async def test_status_announce_is_spoken_but_not_in_history() -> None:
    tts = RecordingTts()
    model = StatusAnnouncingModel()
    session = VoiceSession(
        duplex=DuplexConfig(start_mode="text", loop_forever=False, wake_ack_text=""),
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=tts,
        llm=model,
        audio=NullAudioSink(),
    )
    await session.handle_transcript(Transcript("帮我算个数", TranscriptKind.FINAL))
    await session.wait_for_idle()

    # 播报文案先于正文被朗读
    assert "我算一下。" in tts.spoken
    assert tts.spoken.index("我算一下。") < tts.spoken.index("答案是42。")
    # 播报不进入对话历史，正文进入
    history_texts = [m["content"] for m in session._history]
    assert all("我算一下" not in text for text in history_texts)
    assert any("答案是42" in text for text in history_texts)


@pytest.mark.asyncio
async def test_plain_model_without_status_support_still_works() -> None:
    class PlainModel:
        async def stream_chat(self, messages) -> AsyncIterator[str]:
            yield "你好。"

    tts = RecordingTts()
    session = VoiceSession(
        duplex=DuplexConfig(start_mode="text", loop_forever=False, wake_ack_text=""),
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=tts,
        llm=PlainModel(),
        audio=NullAudioSink(),
    )
    await session.handle_transcript(Transcript("你好", TranscriptKind.FINAL))
    await session.wait_for_idle()
    assert tts.spoken == ["你好。"]
