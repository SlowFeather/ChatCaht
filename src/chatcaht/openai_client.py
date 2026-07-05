from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import OpenAIConfig

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(self, cfg: OpenAIConfig):
        self.cfg = cfg
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.timeout_sec),
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> tuple[bool, str]:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            data = response.json()
            model_ids = [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)]
            if self.cfg.model in model_ids:
                return True, f"model available: {self.cfg.model}"
            if model_ids:
                return True, f"OpenAI-compatible API reachable; configured model not listed. available={', '.join(model_ids[:5])}"
            return True, "OpenAI-compatible API reachable; no models returned by /models"
        except Exception as exc:
            return False, str(exc)

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
        }
        if self.cfg.extra_body:
            payload.update(self.cfg.extra_body)
        started = time.monotonic()
        first_token_at: float | None = None
        chars = 0
        logger.debug("llm request model=%s messages=%d", self.cfg.model, len(messages))
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if line == "[DONE]":
                        break
                    if not line:
                        continue
                    data = _loads_json(line)
                    if data is None:
                        continue
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            chars += len(str(content))
                            yield str(content)
        except Exception:
            logger.warning(
                "llm stream failed model=%s after %.2fs (chars=%d)",
                self.cfg.model,
                time.monotonic() - started,
                chars,
            )
            raise
        total = time.monotonic() - started
        first = (first_token_at - started) if first_token_at is not None else total
        logger.info(
            "llm stream done model=%s chars=%d first_token=%.2fs total=%.2fs",
            self.cfg.model,
            chars,
            first,
            total,
        )
        if chars == 0:
            logger.warning(
                "llm returned no content; if %s is a reasoning model, "
                'set openai.extra_body {"reasoning_effort": "none"} or raise max_tokens',
                self.cfg.model,
            )


def _loads_json(text: str) -> dict[str, Any] | None:
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
