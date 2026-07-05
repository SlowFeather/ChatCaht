from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    artifacts_dir: str = "artifacts"
    logs_dir: str = "artifacts/logs"
    output_dir: str = "artifacts/output"


@dataclass(slots=True)
class OpenAIConfig:
    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwopus3.5-9b-coder"
    api_key: str = "lm-studio"
    temperature: float = 0.6
    max_tokens: int = 512
    timeout_sec: float = 60.0
    # 附加请求字段，原样并入 /chat/completions 请求体。
    # 思考型模型（如 Ollama 的 qwen3.5）默认把输出耗在 reasoning 上导致
    # content 为空，语音场景需要 {"reasoning_effort": "none"} 直接出正文。
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WakeConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8766/v1/wake/ws"
    auto_start_listening: bool = True
    trigger_words: list[str] = field(default_factory=lambda: ["小元", "你好小元"])


@dataclass(slots=True)
class SttConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8790/v1/stt/ws"
    auto_start_listening: bool = True
    final_events_only: bool = True


@dataclass(slots=True)
class TtsConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8787/v1/tts/ws"
    speaker: str | None = None
    speaker_id: int | None = None
    speed: float = 1.0
    save_last_response_wav: bool = False


@dataclass(slots=True)
class DuplexConfig:
    start_mode: str = "wake"
    allow_barge_in: bool = True
    cancel_tts_on_user_speech: bool = True
    end_session_words: list[str] = field(default_factory=lambda: ["退出", "停止聊天", "再见"])
    system_prompt: str = (
        "你是一个运行在本地电脑上的中文实时语音助手。"
        "回答要自然、简洁、适合直接朗读。需要写代码时保持清晰严谨。"
    )
    max_history_turns: int = 8
    # 会话结束(结束口令/空闲超时)后回到待唤醒状态，而不是退出程序
    loop_forever: bool = True
    # 对话中连续无语音输入超过该秒数则自动回到待唤醒；0 表示禁用
    idle_timeout_sec: float = 60.0
    # 唤醒后播报的应答语，空字符串表示不播报
    wake_ack_text: str = ""
    # 每次重新唤醒开启新会话时清空对话历史
    reset_history_per_session: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    log_level: str = "INFO"
    health_timeout_sec: float = 5.0
    mock_text_inputs: list[str] = field(default_factory=list)
    # 连接断开后的重连退避（秒）
    reconnect_initial_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 30.0


@dataclass(slots=True)
class SupervisorConfig:
    # run 模式下是否由 ChatCaht 自动拉起并守护 WakeUp/SpText/GVoice 服务
    manage_services: bool = True
    # 会话崩溃后的重启退避（秒）
    restart_delay_sec: float = 2.0
    max_restart_delay_sec: float = 60.0
    # 会话稳定运行超过该秒数后重置重启退避
    stable_run_sec: float = 300.0
    # 周期性健康检查间隔（秒），发现服务不健康时自动重启
    health_check_interval_sec: float = 30.0


@dataclass(slots=True)
class ServicesConfig:
    uv_executable: str = "uv"
    wakeup_dir: str = "../WakeUp/WakeUp_Project"
    wakeup_config: str = "configs/config.yaml"
    sptext_dir: str = "../SpText"
    sptext_config: str = "configs/config.example.yaml"
    gvoice_dir: str = "../GVoice"
    gvoice_config: str = "configs/config.yaml"
    startup_timeout_sec: float = 20.0


@dataclass(slots=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    duplex: DuplexConfig = field(default_factory=DuplexConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    def ensure_dirs(self) -> None:
        for path in (self.paths.artifacts_dir, self.paths.logs_dir, self.paths.output_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def lmstudio(self) -> OpenAIConfig:
        return self.openai


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path is not None:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("config file must contain a mapping")
        data = _normalize_config_keys(data)
        _merge(cfg, data)
    validate_config(cfg)
    return cfg


LmStudioConfig = OpenAIConfig


def _normalize_config_keys(data: dict[str, Any]) -> dict[str, Any]:
    if "lmstudio" not in data:
        return data
    if "openai" in data:
        raise KeyError("config cannot contain both 'openai' and legacy 'lmstudio'")
    normalized = dict(data)
    normalized["openai"] = normalized.pop("lmstudio")
    return normalized


def _merge(target: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"unknown config key {key!r} for {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)


def validate_config(cfg: Config) -> None:
    for name, level in {"runtime.log_level": cfg.runtime.log_level}.items():
        if level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"{name} must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    if cfg.duplex.start_mode not in {"wake", "manual", "text"}:
        raise ValueError("duplex.start_mode must be one of: wake, manual, text")
    for mode_name, mode in {
        "wake.mode": cfg.wake.mode,
        "stt.mode": cfg.stt.mode,
        "tts.mode": cfg.tts.mode,
    }.items():
        if mode not in {"service", "mock", "disabled"}:
            raise ValueError(f"{mode_name} must be one of: service, mock, disabled")
    if not cfg.wake.url.startswith(("ws://", "wss://")):
        raise ValueError("wake.url must start with ws:// or wss://")
    if cfg.duplex.max_history_turns < 1:
        raise ValueError("duplex.max_history_turns must be positive")
    if cfg.openai.max_tokens < 1:
        raise ValueError("openai.max_tokens must be positive")
    if cfg.runtime.health_timeout_sec <= 0:
        raise ValueError("runtime.health_timeout_sec must be positive")
    if cfg.services.startup_timeout_sec <= 0:
        raise ValueError("services.startup_timeout_sec must be positive")
    if cfg.duplex.idle_timeout_sec < 0:
        raise ValueError("duplex.idle_timeout_sec must be >= 0 (0 disables idle timeout)")
    if cfg.runtime.reconnect_initial_delay_sec <= 0:
        raise ValueError("runtime.reconnect_initial_delay_sec must be positive")
    if cfg.runtime.reconnect_max_delay_sec < cfg.runtime.reconnect_initial_delay_sec:
        raise ValueError("runtime.reconnect_max_delay_sec must be >= reconnect_initial_delay_sec")
    if cfg.supervisor.restart_delay_sec <= 0:
        raise ValueError("supervisor.restart_delay_sec must be positive")
    if cfg.supervisor.max_restart_delay_sec < cfg.supervisor.restart_delay_sec:
        raise ValueError("supervisor.max_restart_delay_sec must be >= restart_delay_sec")
    if cfg.supervisor.health_check_interval_sec <= 0:
        raise ValueError("supervisor.health_check_interval_sec must be positive")
