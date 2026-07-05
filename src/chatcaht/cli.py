from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

from .adapters.llm import create_llm_client
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .audio import NullAudioSink, SoundDeviceSink, WaveFileSink
from .config import Config, load_config
from .health import run_health_checks
from .logging import setup_logging
from .orchestrator import VoiceSession
from .service_manager import ServiceManager
from .selftest import run_selftest
from .supervisor import Supervisor


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\n已停止。")
        return 130


def default_config_path() -> str:
    for candidate in ("configs/config.yaml", "configs/config.example.yaml"):
        if Path(candidate).exists():
            return candidate
    return "configs/config.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chatcaht", description="Full-duplex local AI voice chat orchestrator")
    parser.add_argument("--config", default=default_config_path(), help="Path to ChatCaht YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="Check OpenAI-compatible model API and voice service connectivity")
    sub.add_parser("selftest", help="Run mock end-to-end orchestration checks")

    services = sub.add_parser("services", help="Start, stop, or inspect WakeUp/SpText/GVoice services")
    services.add_argument("action", choices=["start", "stop", "status"])
    services.add_argument("--only", nargs="+", choices=["wake", "stt", "tts", "llm"], help="Limit action to selected services")
    services.add_argument("--no-wait", action="store_true", help="Do not wait for health checks after start")

    chat = sub.add_parser("chat", help="Run realtime voice chat")
    chat.add_argument("--mock", action="store_true", help="Use mock wake/STT/TTS and configured mock_text_inputs")
    chat.add_argument("--no-audio", action="store_true", help="Discard TTS audio instead of playing it")
    chat.add_argument("--save-wav", help="Save synthesized audio to a WAV file instead of playing it")
    chat.add_argument("--once", action="store_true", help="Exit after one conversation instead of returning to wake standby")

    run = sub.add_parser("run", help="Long-running daemon: manage services, auto-restart, wake anytime")
    run.add_argument("--no-services", action="store_true", help="Do not start/supervise WakeUp/SpText/GVoice services")
    run.add_argument("--no-audio", action="store_true", help="Discard TTS audio instead of playing it")

    text = sub.add_parser("text", help="Send one text prompt to the configured OpenAI-compatible model API")
    text.add_argument("prompt", nargs="+")

    init = sub.add_parser("init-config", help="Copy the example config to a writable path")
    init.add_argument("--out", default="configs/config.yaml")
    init.add_argument("--force", action="store_true")
    return parser


async def _amain(args: argparse.Namespace) -> int:
    if args.command == "init-config":
        return _init_config(args)

    cfg = load_config(args.config)
    cfg.ensure_dirs()
    log_file = Path(cfg.paths.logs_dir) / "chatcaht.log"
    setup_logging(cfg.runtime.log_level, str(log_file))
    logging.getLogger("chatcaht.cli").info(
        "chatcaht %s starting; config=%s log=%s", args.command, args.config, log_file
    )

    if args.command == "doctor":
        return await _doctor(cfg)
    if args.command == "selftest":
        return await _selftest(cfg)
    if args.command == "services":
        return await _services(cfg, args)
    if args.command == "chat":
        return await _chat(cfg, args)
    if args.command == "run":
        return await _run_daemon(cfg, args)
    if args.command == "text":
        return await _text(cfg, " ".join(args.prompt))
    raise AssertionError(args.command)


def _init_config(args: argparse.Namespace) -> int:
    src = Path(__file__).resolve().parents[2] / "configs" / "config.example.yaml"
    dst = Path(args.out)
    if dst.exists() and not args.force:
        print(f"配置已存在: {dst}")
        print("需要覆盖时加 --force。")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"已创建配置: {dst}")
    return 0


async def _doctor(cfg: Config) -> int:
    checks = await run_health_checks(cfg)
    ok_all = True
    for check in checks:
        mark = "OK" if check.ok else "FAIL"
        print(f"[{mark}] {check.name}: {check.detail}")
        ok_all = ok_all and check.ok
    return 0 if ok_all else 2


async def _selftest(cfg: Config) -> int:
    cfg.wake.mode = "mock"
    cfg.stt.mode = "mock"
    cfg.tts.mode = "mock"
    results = await run_selftest(cfg)
    ok_all = True
    for name, ok, detail in results:
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        ok_all = ok_all and ok
    return 0 if ok_all else 2


async def _services(cfg: Config, args: argparse.Namespace) -> int:
    manager = ServiceManager(cfg)
    names = args.only
    if args.action == "start":
        statuses = await manager.start(names, wait=not args.no_wait)
    elif args.action == "stop":
        statuses = await manager.stop(names)
    else:
        statuses = await manager.status(names)
    ok_all = True
    for status in statuses:
        display_ok = (not status.running) if args.action == "stop" else status.ok
        mark = "OK" if display_ok else "FAIL"
        running = "running" if status.running else "stopped"
        pid = status.pid if status.pid is not None else "-"
        detail = "stopped" if args.action == "stop" and display_ok else status.detail
        print(f"[{mark}] {status.name}: {running} pid={pid} {detail} log={status.log_file}")
        ok_all = ok_all and display_ok
    return 0 if ok_all else 2


async def _chat(cfg: Config, args: argparse.Namespace) -> int:
    if args.mock:
        cfg.wake.mode = "mock"
        cfg.stt.mode = "mock"
        cfg.tts.mode = "mock"
        cfg.duplex.loop_forever = False
        if not cfg.runtime.mock_text_inputs:
            cfg.runtime.mock_text_inputs = ["你好，介绍一下你自己。", "退出"]
    if args.once:
        cfg.duplex.loop_forever = False

    wake = create_wake_client(cfg.wake, timeout=cfg.runtime.health_timeout_sec)
    stt = create_stt_client(cfg.stt, mock_inputs=cfg.runtime.mock_text_inputs, timeout=cfg.runtime.health_timeout_sec)
    tts = create_tts_client(cfg.tts, timeout=cfg.runtime.health_timeout_sec)
    lm = create_llm_client(cfg)
    audio = _create_audio_sink(cfg, args)
    session = VoiceSession(
        duplex=cfg.duplex,
        wake=wake,
        stt=stt,
        tts=tts,
        llm=lm,
        audio=audio,
        wake_trigger_words=cfg.wake.trigger_words,
        runtime=cfg.runtime,
    )

    print("ChatCaht 已启动。按 Ctrl+C 停止。")
    try:
        await session.run()
    finally:
        await lm.close()
    print(
        "会话结束: "
        f"conversations={session.stats.conversations}, wake={session.stats.wake_events}, "
        f"user={session.stats.user_turns}, assistant={session.stats.assistant_turns}, "
        f"interrupts={session.stats.interruptions}"
    )
    return 0


async def _run_daemon(cfg: Config, args: argparse.Namespace) -> int:
    if args.no_services:
        cfg.supervisor.manage_services = False
    # 守护模式必须循环运行、随时可唤醒
    cfg.duplex.loop_forever = True
    if cfg.duplex.start_mode not in {"wake", "manual"}:
        cfg.duplex.start_mode = "wake"

    supervisor = Supervisor(cfg, audio_factory=lambda: _create_audio_sink(cfg, args))
    print("ChatCaht 守护模式已启动：随时唤醒，自动恢复。按 Ctrl+C 停止。")
    try:
        return await supervisor.run()
    except asyncio.CancelledError:
        supervisor.request_stop()
        raise


def _create_audio_sink(cfg: Config, args: argparse.Namespace):
    if getattr(args, "no_audio", False):
        return NullAudioSink()
    wav_path = getattr(args, "save_wav", None)
    if wav_path is None and cfg.tts.save_last_response_wav:
        wav_path = str(Path(cfg.paths.output_dir) / "last_response.wav")
    if wav_path:
        return WaveFileSink(wav_path)
    try:
        return SoundDeviceSink()
    except Exception:
        return NullAudioSink()


async def _text(cfg: Config, prompt: str) -> int:
    lm = create_llm_client(cfg)
    messages = [
        {"role": "system", "content": cfg.duplex.system_prompt},
        {"role": "user", "content": prompt},
    ]

    async def print_status(event: dict) -> None:
        announce = str(event.get("announce") or "").strip()
        if announce:
            print(f"\n[状态] {announce}", flush=True)

    kwargs = {"on_status": print_status} if getattr(lm, "supports_status_events", False) else {}
    try:
        async for chunk in lm.stream_chat(messages, **kwargs):
            print(chunk, end="", flush=True)
        print()
    finally:
        await lm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
