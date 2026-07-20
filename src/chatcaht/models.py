from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class TranscriptKind(str, Enum):
    PARTIAL = "partial"
    FINAL = "final"


class ServiceState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass(slots=True)
class ServiceProbe:
    state: ServiceState
    detail: str = ""
    raw: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self.state is ServiceState.READY


def service_probe_from_status(payload: dict[str, Any], *, service: str) -> ServiceProbe:
    ready = bool(payload.get("ready"))
    raw_state = str(payload.get("state") or "").strip().upper()
    if ready:
        state = ServiceState.READY
    elif raw_state in ServiceState._value2member_map_:
        state = ServiceState(raw_state)
    elif raw_state in {"STARTING", "LOADING", "WARMING", "OPENING_AUDIO"}:
        state = ServiceState.STARTING
    elif raw_state in {"FAILED", "STOPPED", "FATAL"}:
        state = ServiceState.FAILED
    else:
        state = ServiceState.DEGRADED
    error = payload.get("last_error") or payload.get("error")
    if error:
        detail = str(error)
    else:
        detail = (
            f"{service} state={state.value} model_loaded={payload.get('model_loaded')} "
            f"audio_open={payload.get('audio_open')}"
        )
    return ServiceProbe(state=state, detail=detail, raw=payload)


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
    state: ServiceState | None = None
