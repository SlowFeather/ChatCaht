from __future__ import annotations

import asyncio
import wave
from pathlib import Path


class AudioSink:
    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        return None


class NullAudioSink(AudioSink):
    def __init__(self) -> None:
        self.bytes_played = 0

    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        self.bytes_played += len(pcm)
        await asyncio.sleep(0)


class WaveFileSink(AudioSink):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wav: wave.Wave_write | None = None
        self._sample_rate: int | None = None
        self._channels: int | None = None

    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        if self._wav is None:
            self._sample_rate = sample_rate
            self._channels = channels
            self._wav = wave.open(str(self.path), "wb")
            self._wav.setnchannels(channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
        self._wav.writeframes(pcm)
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None


class SoundDeviceSink(AudioSink):
    def __init__(self) -> None:
        self._stream = None

    async def play(self, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
        await asyncio.to_thread(self._write, pcm, sample_rate, channels)

    async def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            await asyncio.to_thread(stream.stop)
            await asyncio.to_thread(stream.close)

    def _write(self, pcm: bytes, sample_rate: int, channels: int) -> None:
        import numpy as np
        import sounddevice as sd

        samples = np.frombuffer(pcm, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape((-1, channels))
        if self._stream is None:
            self._stream = sd.OutputStream(samplerate=sample_rate, channels=channels, dtype="int16")
            self._stream.start()
        self._stream.write(samples)
