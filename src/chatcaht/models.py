from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TranscriptKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


@dataclass(slots=True)
class Transcript:
    text: str
    kind: TranscriptKind
    source: str = "microphone"
    segment_id: int | None = None
    raw: dict[str, Any] | None = None

    @property
    def is_final(self) -> bool:
        return self.kind == TranscriptKind.FINAL


@dataclass(slots=True)
class WakeEvent:
    model: str
    score: float
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class TtsChunk:
    pcm: bytes
    sample_rate: int
    channels: int = 1


@dataclass(slots=True)
class HealthCheck:
    name: str
    ok: bool
    detail: str = ""
