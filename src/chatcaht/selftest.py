from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .adapters.stt import MockSttClient
from .adapters.tts import MockTtsClient
from .adapters.wake import MockWakeClient
from .audio import NullAudioSink
from .config import Config
from .orchestrator import ChatModel, VoiceSession


class ScriptedModel(ChatModel):
    def __init__(self, chunks: list[str] | None = None, delay: float = 0.0):
        self.chunks = chunks or ["你好，我已经准备好了。"]
        self.delay = delay
        self.calls = 0

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls += 1
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


async def run_selftest(cfg: Config) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    wake = MockWakeClient()
    stt = MockSttClient(["你好，做个自检。"])
    tts = MockTtsClient()
    llm = ScriptedModel(["自检链路正常。"])
    audio = NullAudioSink()
    session = VoiceSession(duplex=cfg.duplex, wake=wake, stt=stt, tts=tts, llm=llm, audio=audio)

    stats = await asyncio.wait_for(session.run(), timeout=5)
    results.append(("wake", stats.wake_events == 1, f"wake_events={stats.wake_events}"))
    results.append(("asr", stats.user_turns == 1, f"user_turns={stats.user_turns}"))
    results.append(("llm", llm.calls == 1, f"llm_calls={llm.calls}"))
    results.append(("tts", stats.tts_chunks >= 1 and audio.bytes_played > 0, f"chunks={stats.tts_chunks} bytes={audio.bytes_played}"))

    loop_model = ScriptedModel(["第一轮正常。", "第二轮正常。"])
    loop_audio = NullAudioSink()
    loop_session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient(["第一轮问题", "第二轮问题"], turn_delay=0.1),
        tts=MockTtsClient(),
        llm=loop_model,
        audio=loop_audio,
    )
    loop_stats = await asyncio.wait_for(loop_session.run(), timeout=5)
    results.append(
        (
            "dialog-loop",
            loop_stats.wake_events == 1
            and loop_stats.user_turns == 2
            and loop_stats.assistant_turns == 2
            and loop_model.calls == 2,
            (
                f"wake_events={loop_stats.wake_events} user_turns={loop_stats.user_turns} "
                f"assistant_turns={loop_stats.assistant_turns} llm_calls={loop_model.calls}"
            ),
        )
    )

    interrupt_model = ScriptedModel(["第一句会被打断。", "第二句不应播完。"], delay=0.03)
    interrupt_session = VoiceSession(
        duplex=cfg.duplex,
        wake=MockWakeClient(),
        stt=MockSttClient([]),
        tts=MockTtsClient(),
        llm=interrupt_model,
        audio=NullAudioSink(),
    )
    from .models import Transcript, TranscriptKind

    await interrupt_session.handle_transcript(Transcript("先说一个长回答", TranscriptKind.FINAL))
    await asyncio.sleep(0.01)
    await interrupt_session.handle_transcript(Transcript("打断并换个问题", TranscriptKind.FINAL))
    await interrupt_session.wait_for_idle()
    results.append(
        (
            "barge-in",
            interrupt_session.stats.interruptions == 1 and interrupt_model.calls == 2,
            f"interruptions={interrupt_session.stats.interruptions} llm_calls={interrupt_model.calls}",
        )
    )
    return results
