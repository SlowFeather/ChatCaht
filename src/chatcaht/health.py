from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import psutil
import sounddevice as sd
import yaml

from .adapters.llm import create_llm_client
from .adapters.stt import create_stt_client
from .adapters.tts import create_tts_client
from .adapters.wake import create_wake_client
from .assets import verify_manifest
from .audio_runtime import AudioRuntimeClient
from .config import Config
from .models import HealthCheck, ServiceProbe, ServiceState


async def run_health_checks(cfg: Config) -> list[HealthCheck]:
    timeout = cfg.runtime.health_timeout_sec
    wake = create_wake_client(cfg.wake, timeout=timeout)
    stt = create_stt_client(cfg.stt, mock_inputs=cfg.runtime.mock_text_inputs, timeout=timeout)
    tts = create_tts_client(cfg.tts, timeout=timeout)
    lm = create_llm_client(cfg)
    try:
        awaitables = [
            _check_probe("wake", wake.probe()),
            _check_probe("stt", stt.probe()),
            _check_probe("tts", tts.probe()),
        ]
        llm_probe = getattr(lm, "probe", None)
        if llm_probe is not None:
            awaitables.append(_check_probe(f"llm({cfg.llm.provider})", llm_probe()))
        else:
            awaitables.append(_check(f"llm({cfg.llm.provider})", lm.health()))
        if cfg.audio.mode == "unified_required":
            awaitables.insert(0, _check_probe("audio(aec)", AudioRuntimeClient(cfg.audio, timeout=timeout).probe()))
        checks = await asyncio.gather(*awaitables)
    finally:
        await lm.close()
    return list(checks)


async def run_deep_health_checks(cfg: Config, *, manifest: str | Path) -> list[HealthCheck]:
    basic = await run_health_checks(cfg)
    assets, cosy, lmstudio, soundcard, ports = await asyncio.gather(
        asyncio.to_thread(_asset_checks, manifest),
        asyncio.to_thread(_cosyvoice_environment_check, cfg),
        asyncio.to_thread(_lmstudio_model_check, cfg),
        asyncio.to_thread(_soundcard_check, cfg),
        asyncio.to_thread(_port_checks, cfg, basic),
    )
    return [*basic, *assets, cosy, lmstudio, soundcard, *ports]


async def _check_probe(name: str, awaitable) -> HealthCheck:
    try:
        probe: ServiceProbe = await awaitable
        return HealthCheck(name=name, ok=probe.ready, detail=probe.detail, state=probe.state)
    except Exception as exc:
        return HealthCheck(name=name, ok=False, detail=str(exc), state=ServiceState.FAILED)


async def _check(name: str, awaitable) -> HealthCheck:
    try:
        ok, detail = await awaitable
        state = ServiceState.READY if ok else ServiceState.FAILED
        return HealthCheck(name=name, ok=ok, detail=detail, state=state)
    except Exception as exc:
        return HealthCheck(name=name, ok=False, detail=str(exc), state=ServiceState.FAILED)


def _asset_checks(manifest: str | Path) -> list[HealthCheck]:
    try:
        results = verify_manifest(manifest)
    except Exception as exc:
        return [HealthCheck("assets", False, str(exc), ServiceState.FAILED)]
    return [
        HealthCheck(
            name=f"asset:{result.component}:{result.path.name}",
            ok=result.ok,
            detail=f"{result.detail} path={result.path}",
            state=ServiceState.READY if result.ok else ServiceState.FAILED,
        )
        for result in results
    ]


def _cosyvoice_environment_check(cfg: Config) -> HealthCheck:
    gvoice_dir = Path(cfg.services.gvoice_dir).expanduser().resolve()
    gvoice_config = _load_yaml(_resolve_from(gvoice_dir, cfg.services.gvoice_config))
    cosy_cfg = ((gvoice_config.get("tts") or {}).get("cosyvoice3") or {})
    sidecar_dir = _resolve_from(gvoice_dir, str(cosy_cfg.get("sidecar_dir") or "sidecars/cosyvoice3"))
    sidecar_config = _load_yaml(_existing_config(sidecar_dir))
    fp16_requested = bool(((sidecar_config.get("model") or {}).get("fp16", True)))
    script = r'''
import json, sys
from pathlib import Path
repo = Path("vendor/CosyVoice").resolve()
sys.path[:0] = [str(repo / "third_party" / "Matcha-TTS"), str(repo)]
import torch, torchaudio, pyarrow, lightning, pkg_resources
import cosyvoice.flow.flow_matching
import cosyvoice.dataset.processor
cuda = torch.cuda.is_available()
fp16 = False
fp16_error = None
if cuda:
    try:
        tensor = torch.ones((8, 8), device="cuda", dtype=torch.float16)
        float((tensor @ tensor).sum().item())
        fp16 = True
    except Exception as exc:
        fp16_error = str(exc)
print(json.dumps({
    "cuda": cuda,
    "fp16": fp16,
    "fp16_error": fp16_error,
    "device": torch.cuda.get_device_name(0) if cuda else None,
    "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
    "torch": torch.__version__,
}))
'''
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    try:
        completed = subprocess.run(
            [cfg.services.uv_executable, "run", "python", "-c", script],
            cwd=sidecar_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except Exception as exc:
        return HealthCheck("cosyvoice(cuda/fp16/imports)", False, str(exc), ServiceState.FAILED)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        return HealthCheck("cosyvoice(cuda/fp16/imports)", False, detail, ServiceState.FAILED)
    try:
        info = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return HealthCheck("cosyvoice(cuda/fp16/imports)", False, f"invalid probe output: {exc}", ServiceState.FAILED)
    ok = bool(info.get("cuda") and info.get("fp16")) if fp16_requested else True
    detail = (
        f"imports=ok fp16_requested={fp16_requested} cuda={info.get('cuda')} fp16={info.get('fp16')} "
        f"device={info.get('device')} capability={info.get('capability')} torch={info.get('torch')}"
    )
    if info.get("fp16_error"):
        detail += f" fp16_error={info['fp16_error']}"
    return HealthCheck(
        "cosyvoice(cuda/fp16/imports)",
        ok,
        detail,
        ServiceState.READY if ok else ServiceState.FAILED,
    )


def _lmstudio_model_check(cfg: Config) -> HealthCheck:
    if cfg.llm.provider == "lollama":
        lollama_dir = Path(cfg.services.lollama_dir).expanduser().resolve()
        data = _load_yaml(_resolve_from(lollama_dir, cfg.services.lollama_config))
        upstream = data.get("upstream") or {}
        base_url = str(upstream.get("base_url") or cfg.openai.base_url)
        model = str(upstream.get("model") or cfg.openai.model)
        api_key = os.environ.get("LOLLAMA_UPSTREAM_API_KEY") or str(upstream.get("api_key") or "")
    else:
        base_url = cfg.openai.base_url
        model = cfg.openai.model
        api_key = os.environ.get("CHATCAHT_OPENAI_API_KEY") or cfg.openai.api_key
    request = Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    try:
        with urlopen(request, timeout=cfg.runtime.health_timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))
        loaded = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
    except Exception as exc:
        return HealthCheck("lmstudio(model)", False, str(exc), ServiceState.FAILED)
    ok = model in loaded
    detail = f"required={model} loaded={sorted(loaded)}"
    return HealthCheck("lmstudio(model)", ok, detail, ServiceState.READY if ok else ServiceState.FAILED)


def _soundcard_check(cfg: Config) -> HealthCheck:
    try:
        input_device = sd.query_devices(cfg.audio.input_device or None, "input")
        output_device = sd.query_devices(cfg.audio.output_device or None, "output")
        input_channels = int(input_device.get("max_input_channels") or 0)
        output_channels = int(output_device.get("max_output_channels") or 0)
        ok = input_channels > 0 and output_channels > 0
        detail = (
            f"input={input_device.get('name')} channels={input_channels}; "
            f"output={output_device.get('name')} channels={output_channels}"
        )
    except Exception as exc:
        return HealthCheck("soundcard", False, str(exc), ServiceState.FAILED)
    return HealthCheck("soundcard", ok, detail, ServiceState.READY if ok else ServiceState.FAILED)


def _port_checks(cfg: Config, basic: list[HealthCheck]) -> list[HealthCheck]:
    endpoints = {
        "audio": cfg.audio.url,
        "wake": cfg.wake.url,
        "stt": cfg.stt.url,
        "tts": cfg.tts.url,
        "lmstudio": cfg.openai.base_url,
    }
    if cfg.llm.provider == "lollama":
        endpoints["llm"] = cfg.lollama.url
    listeners: dict[int, list[int]] = {}
    for connection in psutil.net_connections(kind="tcp"):
        if connection.status != psutil.CONN_LISTEN or not connection.laddr:
            continue
        listeners.setdefault(int(connection.laddr.port), []).append(int(connection.pid or 0))
    health_by_name = {check.name.split("(", 1)[0]: check.ok for check in basic}
    checks: list[HealthCheck] = []
    for name, url in endpoints.items():
        port = urlsplit(url).port
        if port is None:
            port = 443 if urlsplit(url).scheme in {"https", "wss"} else 80
        pids = sorted(set(listeners.get(port, [])))
        owners = []
        for pid in pids:
            try:
                owners.append(f"{pid}:{psutil.Process(pid).name()}")
            except (psutil.Error, OSError):
                owners.append(str(pid))
        occupied = bool(pids)
        service_ok = health_by_name.get(name, occupied)
        ok = occupied and service_ok
        detail = f"port={port} listeners={owners or 'none'}"
        if occupied and not service_ok:
            detail += " but configured service is not READY"
        checks.append(HealthCheck(f"port:{name}", ok, detail, ServiceState.READY if ok else ServiceState.FAILED))
    return checks


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return data


def _resolve_from(base: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _existing_config(project_dir: Path) -> Path:
    for name in ("configs/config.yaml", "configs/config.example.yaml"):
        candidate = project_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no config.yaml or config.example.yaml under {project_dir}")
