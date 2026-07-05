from __future__ import annotations

import asyncio

from .adapters.llm import create_llm_client
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .config import Config
from .models import HealthCheck


async def run_health_checks(cfg: Config) -> list[HealthCheck]:
    timeout = cfg.runtime.health_timeout_sec
    wake = create_wake_client(cfg.wake, timeout=timeout)
    stt = create_stt_client(cfg.stt, mock_inputs=cfg.runtime.mock_text_inputs, timeout=timeout)
    tts = create_tts_client(cfg.tts, timeout=timeout)
    lm = create_llm_client(cfg)
    try:
        checks = await asyncio.gather(
            _check("wake", wake.health()),
            _check("stt", stt.health()),
            _check("tts", tts.health()),
            _check(f"llm({cfg.llm.provider})", lm.health()),
        )
    finally:
        await lm.close()
    return list(checks)


async def _check(name: str, awaitable) -> HealthCheck:
    try:
        ok, detail = await awaitable
        return HealthCheck(name=name, ok=ok, detail=detail)
    except Exception as exc:
        return HealthCheck(name=name, ok=False, detail=str(exc))
