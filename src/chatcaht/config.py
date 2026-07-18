from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    artifacts_dir: str = "artifacts"
    logs_dir: str = "artifacts/logs"
    output_dir: str = "artifacts/output"
    metrics_file: str = "artifacts/metrics/turns.jsonl"


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
class LlmConfig:
    # 回复由谁生成：
    #   openai  —— 直连 OpenAI 兼容 API（LM Studio），走 openai: 段
    #   lollama —— 经 LoLLama 智能体服务（分层记忆 + 工具调用），走 lollama: 段
    provider: str = "openai"


@dataclass(slots=True)
class LollamaConfig:
    url: str = "ws://127.0.0.1:8801/v1/llm/ws"
    # 单条回复的整体超时（秒）；工具调用可能较慢，默认放宽
    response_timeout_sec: float = 180.0
    # 把 LoLLama 的 agent_status 状态播报（"我算一下""让我想想"等）送 TTS 朗读；
    # 播报文案在 LoLLama 侧 config 的 status.announce 段配置
    announce_status: bool = True


@dataclass(slots=True)
class WakeConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8766/v1/wake/ws"
    auto_start_listening: bool = True
    input_mode: str = "microphone"
    trigger_words: list[str] = field(default_factory=lambda: ["小元", "你好小元"])


@dataclass(slots=True)
class SttConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8790/v1/stt/ws"
    auto_start_listening: bool = True
    input_mode: str = "microphone"
    final_events_only: bool = True
    partial_fallback_sec: float = 1.2
    partial_min_chars: int = 2


@dataclass(slots=True)
class TtsConfig:
    enabled: bool = True
    mode: str = "service"
    url: str = "ws://127.0.0.1:8787/v1/tts/ws"
    speaker: str | None = None
    speaker_id: int | None = None
    speed: float | None = None
    save_last_response_wav: bool = False


@dataclass(slots=True)
class AudioRuntimeConfig:
    mode: str = "unified_required"
    url: str = "ws://127.0.0.1:8810/v1/audio/ws"
    device_sample_rate: int = 48000
    capture_sample_rate: int = 16000
    frame_ms: int = 10
    aec_tail_ms: int = 200
    barge_in_min_speech_ms: int = 120
    barge_in_preroll_ms: int = 200
    barge_in_hangover_ms: int = 300
    render_queue_ms: int = 500
    capture_queue_ms: int = 1000
    capture_stall_timeout_ms: int = 3000
    vad_aggressiveness: int = 2
    input_device: str | None = None
    output_device: str | None = None


@dataclass(slots=True)
class DuplexConfig:
    start_mode: str = "wake"
    allow_barge_in: bool = True
    cancel_tts_on_user_speech: bool = True
    end_session_words: list[str] = field(default_factory=lambda: ["退出", "停止聊天", "再见", "拜拜", "闭嘴"])
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
    tts_segment_min_chars: int = 6
    tts_segment_max_chars: int = 80
    tts_segment_flush_ms: int = 350


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
    audio_runtime_dir: str = "../AudioRuntime"
    audio_runtime_executable: str = "build/Release/chat-audio-runtime.exe"
    wakeup_dir: str = "../WakeUp"
    wakeup_config: str = "configs/config.yaml"
    sptext_dir: str = "../SpText"
    sptext_config: str = "configs/config.example.yaml"
    gvoice_dir: str = "../GVoice"
    gvoice_config: str = "configs/config.yaml"
    lollama_dir: str = "../LoLLama"
    lollama_config: str = "configs/config.yaml"
    startup_timeout_sec: float = 20.0


@dataclass(slots=True)
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    lollama: LollamaConfig = field(default_factory=LollamaConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioRuntimeConfig = field(default_factory=AudioRuntimeConfig)
    duplex: DuplexConfig = field(default_factory=DuplexConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    def ensure_dirs(self) -> None:
        for path in (self.paths.artifacts_dir, self.paths.logs_dir, self.paths.output_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.paths.metrics_file).parent.mkdir(parents=True, exist_ok=True)

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
    if api_key := os.environ.get("CHATCAHT_OPENAI_API_KEY"):
        cfg.openai.api_key = api_key
    validate_config(cfg)
    return cfg


LmStudioConfig = OpenAIConfig


def _normalize_config_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    services = normalized.get("services")
    if isinstance(services, dict) and "audio_runtime_config" in services:
        normalized["services"] = dict(services)
        normalized["services"].pop("audio_runtime_config", None)
    if "lmstudio" in normalized:
        if "openai" in normalized:
            raise KeyError("config cannot contain both 'openai' and legacy 'lmstudio'")
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
    for input_name, input_mode in {
        "wake.input_mode": cfg.wake.input_mode,
        "stt.input_mode": cfg.stt.input_mode,
    }.items():
        if input_mode not in {"microphone", "external_pcm"}:
            raise ValueError(f"{input_name} must be one of: microphone, external_pcm")
    if cfg.audio.mode not in {"unified_required", "legacy", "disabled"}:
        raise ValueError("audio.mode must be one of: unified_required, legacy, disabled")
    if not cfg.audio.url.startswith(("ws://", "wss://")):
        raise ValueError("audio.url must start with ws:// or wss://")
    if cfg.audio.mode == "unified_required" and not cfg.audio.url.startswith("ws://"):
        raise ValueError("managed AudioRuntime currently requires a ws:// URL")
    for name, value in {
        "audio.device_sample_rate": cfg.audio.device_sample_rate,
        "audio.capture_sample_rate": cfg.audio.capture_sample_rate,
        "audio.frame_ms": cfg.audio.frame_ms,
        "audio.aec_tail_ms": cfg.audio.aec_tail_ms,
        "audio.barge_in_min_speech_ms": cfg.audio.barge_in_min_speech_ms,
        "audio.barge_in_preroll_ms": cfg.audio.barge_in_preroll_ms,
        "audio.barge_in_hangover_ms": cfg.audio.barge_in_hangover_ms,
        "audio.render_queue_ms": cfg.audio.render_queue_ms,
        "audio.capture_queue_ms": cfg.audio.capture_queue_ms,
        "audio.capture_stall_timeout_ms": cfg.audio.capture_stall_timeout_ms,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg.audio.vad_aggressiveness not in {0, 1, 2, 3}:
        raise ValueError("audio.vad_aggressiveness must be between 0 and 3")
    if cfg.audio.mode == "unified_required" and (
        cfg.audio.device_sample_rate != 48000
        or cfg.audio.capture_sample_rate != 16000
        or cfg.audio.frame_ms != 10
    ):
        raise ValueError(
            "AudioRuntime currently requires device_sample_rate=48000, "
            "capture_sample_rate=16000, and frame_ms=10"
        )
    if not cfg.wake.url.startswith(("ws://", "wss://")):
        raise ValueError("wake.url must start with ws:// or wss://")
    if cfg.llm.provider not in {"openai", "lollama"}:
        raise ValueError("llm.provider must be one of: openai, lollama")
    if not cfg.lollama.url.startswith(("ws://", "wss://")):
        raise ValueError("lollama.url must start with ws:// or wss://")
    if cfg.lollama.response_timeout_sec <= 0:
        raise ValueError("lollama.response_timeout_sec must be positive")
    if cfg.duplex.max_history_turns < 1:
        raise ValueError("duplex.max_history_turns must be positive")
    if cfg.duplex.tts_segment_min_chars < 1:
        raise ValueError("duplex.tts_segment_min_chars must be positive")
    if cfg.duplex.tts_segment_max_chars < cfg.duplex.tts_segment_min_chars:
        raise ValueError("duplex.tts_segment_max_chars must be >= tts_segment_min_chars")
    if cfg.duplex.tts_segment_flush_ms < 1:
        raise ValueError("duplex.tts_segment_flush_ms must be positive")
    if cfg.openai.max_tokens < 1:
        raise ValueError("openai.max_tokens must be positive")
    if cfg.runtime.health_timeout_sec <= 0:
        raise ValueError("runtime.health_timeout_sec must be positive")
    if cfg.stt.partial_fallback_sec < 0:
        raise ValueError("stt.partial_fallback_sec must be >= 0 (0 disables partial fallback)")
    if cfg.stt.partial_min_chars < 1:
        raise ValueError("stt.partial_min_chars must be positive")
    if cfg.tts.speed is not None and cfg.tts.speed <= 0:
        raise ValueError("tts.speed must be positive when set")
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
