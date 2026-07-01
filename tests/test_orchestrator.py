from __future__ import annotations

import asyncio

import pytest

from chatcaht.adapters.stt import MockSttClient
from chatcaht.adapters.tts import MockTtsClient
from chatcaht.adapters.wake import MockWakeClient
from chatcaht.audio import NullAudioSink
from chatcaht.config import Config
from chatcaht.models import Transcript, TranscriptKind
from chatcaht.orchestrator import SentenceBuffer, VoiceSession
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
