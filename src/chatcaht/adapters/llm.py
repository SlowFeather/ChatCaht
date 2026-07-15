from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed

from chatcaht.config import Config, LollamaConfig
from chatcaht.openai_client import OpenAICompatibleClient

logger = logging.getLogger(__name__)

# agent_status 事件回调：入参为 LoLLama 发来的整条消息（含 stage/announce）
StatusHandler = Callable[[dict], Awaitable[None]]


class LollamaChatClient:
    """LoLLama 智能体服务（分层记忆 + 工具调用）的全双工 WebSocket 客户端。

    复用一条连接发送 text 请求，让 LoLLama 服务端维护连接内工作记忆。
    上层（orchestrator）取消消费任务时发送 cancel，LoLLama 侧会中止上游生成（barge-in）。

    LoLLama 的 agent_status 状态钩子（"正在调用工具""让我想想"等）通过
    stream_chat(on_status=...) 回调交给上层播报。
    """

    supports_status_events = True
    manages_conversation_history = True

    def __init__(self, cfg: LollamaConfig, timeout: float = 5.0):
        self.cfg = cfg
        self.timeout = timeout
        self._ws = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await self._close_ws()

    async def health(self) -> tuple[bool, str]:
        try:
            async with websockets.connect(self.cfg.url, open_timeout=self.timeout, max_size=None) as ws:
                await ws.send(json.dumps({"type": "status"}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    return False, "lollama service returned binary health response"
                msg = json.loads(raw)
                if msg.get("type") != "status" or not msg.get("ready"):
                    return False, str(msg.get("last_error") or f"lollama service state={msg.get('state')}")
                return True, f"lollama ready state={msg.get('state')} model_loaded={msg.get('model_loaded')}"
        except Exception as exc:
            return False, str(exc)

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        on_status: StatusHandler | None = None,
    ) -> AsyncIterator[str]:
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        chars = 0
        deadline = started + self.cfg.response_timeout_sec
        text = _last_user_text(messages)
        payload = _chat_payload(request_id, messages, text)
        logger.debug(
            "lollama chat request=%s mode=%s messages=%d",
            request_id,
            "text" if text else "messages",
            len(messages),
        )
        async with self._lock:
            ws = await self._ensure_ws()
            canceled = False
            try:
                await ws.send(json.dumps(payload, ensure_ascii=False))
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f"lollama response timed out after {self.cfg.response_timeout_sec:.0f}s")
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    if isinstance(raw, bytes):
                        continue
                    msg = json.loads(raw)
                    msg_type = msg.get("type")
                    if msg.get("request_id") not in {None, request_id}:
                        continue
                    if msg_type == "delta":
                        chunk = str(msg.get("text") or "")
                        if chunk:
                            chars += len(chunk)
                            yield chunk
                    elif msg_type == "agent_status":
                        logger.info(
                            "lollama status stage=%s name=%s announce=%s result=%s count=%s",
                            msg.get("stage"),
                            msg.get("name"),
                            str(msg.get("announce") or "")[:80],
                            str(msg.get("result") or "")[:120],
                            msg.get("count"),
                        )
                        if on_status is not None and self.cfg.announce_status:
                            try:
                                await on_status(msg)
                            except Exception:
                                logger.exception("agent status handler failed for stage=%s", msg.get("stage"))
                    elif msg_type == "tool":
                        logger.info(
                            "lollama tool call name=%s status=%s detail=%s",
                            msg.get("name"),
                            msg.get("status"),
                            str(msg.get("detail") or "")[:120],
                        )
                    elif msg_type == "done":
                        break
                    elif msg_type == "error":
                        raise RuntimeError(str(msg.get("message") or "lollama error"))
                    elif msg_type == "ack":
                        continue
            except asyncio.CancelledError:
                canceled = True
                await self._send_cancel()
                raise
            except ConnectionClosed:
                await self._close_ws()
                raise
            finally:
                if canceled:
                    logger.info("lollama chat canceled request=%s", request_id)
        logger.info(
            "lollama chat done request=%s chars=%d elapsed=%.2fs",
            request_id,
            chars,
            time.monotonic() - started,
        )

    async def reset(self) -> None:
        """Clear LoLLama's connection-local working history."""
        async with self._lock:
            ws = await self._ensure_ws()
            await ws.send(json.dumps({"type": "reset"}))
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "ack" and msg.get("cmd") == "reset":
                    logger.info("lollama working history reset")
                    return

    async def _ensure_ws(self):
        if self._ws is None:
            self._ws = await websockets.connect(
                self.cfg.url,
                open_timeout=self.timeout,
                ping_interval=20,
                ping_timeout=self.timeout,
                max_size=None,
            )
        return self._ws

    async def _send_cancel(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"type": "cancel"}))
        except Exception:
            await self._close_ws()

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()


def create_llm_client(cfg: Config) -> OpenAICompatibleClient | LollamaChatClient:
    """按 llm.provider 选择回复生成后端。"""
    if cfg.llm.provider == "lollama":
        return LollamaChatClient(cfg.lollama, timeout=cfg.runtime.health_timeout_sec)
    return OpenAICompatibleClient(cfg.openai)


def _chat_payload(request_id: str, messages: list[dict[str, str]], text: str) -> dict:
    if not text:
        raise ValueError("lollama chat requires a user message")
    return {"type": "chat", "request_id": request_id, "text": text}


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""
