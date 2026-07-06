"""ChatCaht 项目监控看板 —— 独立只读子项目。

只读取 ChatCaht 的 configs/config.yaml、artifacts/run/*.pid.json 和
artifacts/logs/*.log，并用与 chatcaht 适配器相同的 WebSocket ping 协议做
健康检查；不 import chatcaht 包、不写任何主项目文件。

启动（在 ChatCaht 根目录）::

    uv run python dashboard/server.py            # 默认 http://127.0.0.1:8899
    uv run python dashboard/server.py --port 9000

依赖：ChatCaht 自身 venv 里已有的 pyyaml + websockets，其余全部为标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# 各项目日志格式不同，统一归一化为 (ts, level, logger, msg)：
#   A) ChatCaht / LoLLama : 2026-07-05 22:13:35,751 INFO lollama.service.server: msg
#   B) SpText / GVoice    : 2026-07-05 22:13:35 INFO [sptext.service.server] msg
#   C) WakeUp (仅时分秒)   : 22:13:35 [INFO] wakeup.service.server: msg
LOG_PATTERNS = (
    re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.]\d{3} (\w+) ([\w.]+): (.*)$"),
    re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) \[([\w.]+)\] (.*)$"),
    re.compile(r"^(\d{2}:\d{2}:\d{2}) \[(\w+)\] ([\w.]+): (.*)$"),
)


def match_log_line(line: str) -> tuple[str, str, str, str] | None:
    for pattern in LOG_PATTERNS:
        m = pattern.match(line)
        if m:
            return m.group(1), m.group(2), m.group(3), m.group(4)
    return None
METRIC_PATTERNS = {
    "llm_first_token": re.compile(r"llm stream done .*?first_token=([\d.]+)s"),
    "llm_total": re.compile(r"llm stream done .*?total=([\d.]+)s"),
    "lollama_elapsed": re.compile(r"lollama chat done .*?elapsed=([\d.]+)s"),
    "tts_elapsed": re.compile(r"tts synthesize done .*?elapsed=([\d.]+)s"),
}


def load_config() -> dict:
    for name in ("config.yaml", "config.example.yaml"):
        path = ROOT / "configs" / name
        if path.exists():
            with path.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
    return {}


def _port_of(url: str) -> int | None:
    try:
        return urlparse(url).port
    except Exception:
        return None


def build_projects(cfg: dict) -> list[dict]:
    """静态描述每个被监控项目：健康检查方式、日志、pid 文件、源码目录。"""
    services = cfg.get("services", {})
    openai_cfg = cfg.get("openai", {})
    provider = (cfg.get("llm") or {}).get("provider", "openai")
    wake_url = (cfg.get("wake") or {}).get("url", "")
    stt_url = (cfg.get("stt") or {}).get("url", "")
    tts_url = (cfg.get("tts") or {}).get("url", "")
    lollama_url = (cfg.get("lollama") or {}).get("url", "")
    base_url = openai_cfg.get("base_url", "")
    projects = [
        {
            "key": "chatcaht",
            "name": "ChatCaht",
            "role": "全双工语音编排器",
            "check": "log",
            "url": "",
            "port": None,
            "pid_file": None,
            "log": "chatcaht.log",
            "dir": str(ROOT),
        },
        {
            "key": "wake",
            "name": "WakeUp",
            "role": "唤醒词检测",
            "check": "ws_greet",  # 服务端先发欢迎帧，再 ping/pong
            "url": wake_url,
            "port": _port_of(wake_url),
            "pid_file": "wakeup.pid.json",
            "log": "wakeup.service.log",
            "own_log": "wakeup:wakeup.log",  # serve 模式写 artifacts/logs/wakeup.log
            "dir": services.get("wakeup_dir", "../WakeUp"),
        },
        {
            "key": "stt",
            "name": "SpText",
            "role": "实时语音识别 ASR",
            "check": "ws_status",
            "url": stt_url,
            "port": _port_of(stt_url),
            "pid_file": "sptext.pid.json",
            "log": "sptext.service.log",
            "own_log": "sptext:sptext.log",
            "dir": services.get("sptext_dir", "../SpText"),
        },
        {
            "key": "llm",
            "name": "LoLLama",
            "role": "智能体（5 层记忆 + 工具）",
            "check": "ws_status" if provider == "lollama" else "off",
            "url": lollama_url,
            "port": _port_of(lollama_url),
            "pid_file": "lollama.pid.json",
            "log": "lollama.service.log",
            "own_log": "lollama:lollama.log",
            "dir": services.get("lollama_dir", "../LoLLama"),
        },
        {
            "key": "lmstudio",
            "name": "LM Studio",
            "role": f"模型后端 · {openai_cfg.get('model', '?')}",
            "check": "http_models",
            "url": base_url,
            "port": _port_of(base_url),
            "pid_file": None,
            "log": None,
            "dir": "",
        },
        {
            "key": "tts",
            "name": "GVoice",
            "role": "流式中文 TTS",
            "check": "ws_ping",
            "url": tts_url,
            "port": _port_of(tts_url),
            "pid_file": "gvoice.pid.json",
            "log": "gvoice.service.log",
            "own_log": "gvoice:gvoice.log",
            "dir": services.get("gvoice_dir", "../GVoice"),
        },
    ]
    return projects


# ---------------------------------------------------------------- 健康检查


# status 帧里值得上卡片的字段（各服务只会有其中一部分）
INFO_FIELDS = (
    "listening",
    "ready",
    "backend",
    "model",
    "upstream",
    "clients",
    "requests_served",
    "uptime_seconds",
    "memory",
    "tools",
)


def _pick_info(frame: dict) -> dict:
    return {k: frame[k] for k in INFO_FIELDS if k in frame and frame[k] is not None}


def ws_health(url: str, *, recv_first: bool, query: str = "ping", timeout: float = 3.0) -> dict:
    """连接服务并发一条 query 命令。

    query="status" 时（WakeUp/SpText/LoLLama 均支持）把回帧里的运行信息
    （listening/ready/model/记忆统计/uptime 等）带回卡片；GVoice 只认 ping，
    未知命令会在它日志里刷 WARNING，所以对 tts 保持 query="ping"。
    """
    from websockets.sync.client import connect

    info: dict = {}
    started = time.perf_counter()
    try:
        with connect(url, open_timeout=timeout, close_timeout=1.0, max_size=None) as ws:
            if recv_first:  # WakeUp 连接即推一帧 status，本身就带 listening/model
                greeting = ws.recv(timeout=timeout)
                if isinstance(greeting, str):
                    try:
                        info.update(_pick_info(json.loads(greeting)))
                    except Exception:
                        pass
            ws.send(json.dumps({"type": query}))
            raw = ws.recv(timeout=timeout)
        latency = round((time.perf_counter() - started) * 1000)
        detail = "reachable"
        if isinstance(raw, (bytes, bytearray)):
            detail = "reachable (binary reply)"
        else:
            try:
                frame = json.loads(raw)
                detail = f"reply type={frame.get('type')}"
                info.update(_pick_info(frame))
            except Exception:
                pass
        return {"ok": True, "latency_ms": latency, "detail": detail, "info": info}
    except Exception as exc:
        return {"ok": False, "latency_ms": None, "detail": f"{type(exc).__name__}: {exc}", "info": info}


def http_models_health(base_url: str, api_key: str, timeout: float = 3.0) -> dict:
    url = base_url.rstrip("/") + "/models"
    started = time.perf_counter()
    try:
        req = Request(url, headers={"Authorization": f"Bearer {api_key or 'none'}"})
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
        latency = round((time.perf_counter() - started) * 1000)
        models = body.get("data", [])
        return {"ok": True, "latency_ms": latency, "detail": f"{len(models)} model(s) loaded"}
    except Exception as exc:
        return {"ok": False, "latency_ms": None, "detail": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- 进程与日志


def read_pid(pid_file: str | None) -> int | None:
    if not pid_file:
        return None
    path = ROOT / "artifacts" / "run" / pid_file
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["pid"])
    except Exception:
        return None


def running_pids() -> set[int]:
    """一次 tasklist 拿到全部存活 PID，避免每个服务各起一个子进程。"""
    if os.name != "nt":
        return set()
    try:
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        pids = set()
        for line in out.splitlines():
            parts = line.split('","')
            if len(parts) > 1:
                try:
                    pids.add(int(parts[1].strip('"')))
                except ValueError:
                    pass
        return pids
    except Exception:
        return set()


def pid_alive(pid: int, alive_set: set[int]) -> bool:
    if os.name == "nt":
        return pid in alive_set
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def tail_text(path: Path, max_bytes: int = 1_000_000) -> list[str]:
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # 丢掉可能被截断的首行
    return lines


def parse_entries(lines: list[str]) -> list[dict]:
    """把原始行聚合成日志条目；无时间戳前缀的行（如 traceback）并入上一条。"""
    entries: list[dict] = []
    for line in lines:
        parsed = match_log_line(line)
        if parsed:
            ts, level, logger_name, msg = parsed
            entries.append({"ts": ts, "level": level, "logger": logger_name, "msg": msg, "extra": []})
        elif entries:
            entries[-1]["extra"].append(line)
        elif line.strip():
            entries.append({"ts": "", "level": "RAW", "logger": "", "msg": line, "extra": []})
    return entries


def log_registry() -> dict[str, Path]:
    """name → path 白名单。ChatCaht 托管日志用裸文件名；各项目自有日志加 '项目:' 前缀。

    托管日志（artifacts/logs/*.service.log）只在服务由 ChatCaht supervisor 拉起时
    更新；服务独立运行时要看它们各自 artifacts/logs 下的自有日志。
    """
    services = load_config().get("services", {})
    registry: dict[str, Path] = {}
    chat_logs = ROOT / "artifacts" / "logs"
    if chat_logs.exists():
        for path in chat_logs.glob("*.log"):
            registry[path.name] = path
    for label, dir_key in (
        ("wakeup", "wakeup_dir"),
        ("sptext", "sptext_dir"),
        ("gvoice", "gvoice_dir"),
        ("lollama", "lollama_dir"),
    ):
        raw = services.get(dir_key)
        if not raw:
            continue
        base = Path(raw)
        if not base.is_absolute():
            base = ROOT / base
        log_dir = base.resolve() / "artifacts" / "logs"
        if log_dir.exists():
            for path in log_dir.glob("*.log"):
                registry[f"{label}:{path.name}"] = path
    return registry


def log_summary(log_name: str | None, registry: dict[str, Path] | None = None) -> dict:
    if not log_name:
        return {"file": None}
    if registry is not None:
        path = registry.get(log_name)
    else:
        path = ROOT / "artifacts" / "logs" / log_name
    if path is None or not path.exists():
        return {"file": log_name, "exists": False}
    stat = path.stat()
    entries = parse_entries(tail_text(path, max_bytes=200_000))
    errors = [e for e in entries if e["level"] in ("ERROR", "CRITICAL")]
    warnings = [e for e in entries if e["level"] == "WARNING"]
    last = entries[-1] if entries else None
    return {
        "file": log_name,
        "exists": True,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "recent_errors": len(errors),
        "recent_warnings": len(warnings),
        "last_error": errors[-1]["msg"][:200] if errors else None,
        "last_line": (last["msg"][:160] if last else None),
        "last_ts": (last["ts"] if last else None),
    }


# ---------------------------------------------------------------- 状态聚合（带短缓存）

_cache_lock = threading.Lock()
_cache: dict = {"ts": 0.0, "data": None}
CACHE_TTL = 2.5


def collect_status() -> dict:
    with _cache_lock:
        if _cache["data"] is not None and time.time() - _cache["ts"] < CACHE_TTL:
            return _cache["data"]

    cfg = load_config()
    projects = build_projects(cfg)
    alive = running_pids()
    openai_cfg = cfg.get("openai", {})

    def check(project: dict) -> dict:
        kind = project["check"]
        if kind == "ws_greet":
            return ws_health(project["url"], recv_first=True)  # 欢迎帧即 status
        if kind == "ws_status":
            return ws_health(project["url"], recv_first=False, query="status")
        if kind == "ws_ping":
            return ws_health(project["url"], recv_first=False)
        if kind == "http_models":
            return http_models_health(project["url"], openai_cfg.get("api_key", ""))
        if kind == "off":
            return {"ok": None, "latency_ms": None, "detail": "当前 provider 未启用"}
        return {"ok": None, "latency_ms": None, "detail": "由日志活动判断"}

    with ThreadPoolExecutor(max_workers=len(projects)) as pool:
        healths = list(pool.map(check, projects))

    registry = log_registry()
    now = time.time()
    result = []
    for project, health in zip(projects, healths):
        pid = read_pid(project["pid_file"])
        process = None
        if project["pid_file"]:
            process = "running" if (pid and pid_alive(pid, alive)) else "stopped"
        # 托管日志 vs 项目自有日志：用更新鲜的那份做卡片摘要
        candidates = [
            log_summary(name, registry)
            for name in (project["log"], project.get("own_log"))
            if name
        ]
        candidates = [c for c in candidates if c.get("exists")]
        log = max(candidates, key=lambda c: c.get("mtime") or 0) if candidates else log_summary(project["log"], registry)
        state = _project_state(project, health, process, log, now)
        result.append(
            {
                "key": project["key"],
                "name": project["name"],
                "role": project["role"],
                "url": project["url"],
                "port": project["port"],
                "dir": project["dir"],
                "pid": pid,
                "process": process,
                "health": health,
                "log": log,
                "state": state,
            }
        )

    payload = {
        "generated_at": now,
        "provider": (cfg.get("llm") or {}).get("provider", "openai"),
        "model": openai_cfg.get("model", ""),
        "projects": result,
    }
    with _cache_lock:
        _cache.update(ts=time.time(), data=payload)
    return payload


def _project_state(project: dict, health: dict, process: str | None, log: dict, now: float) -> str:
    """归一化为 4 档：online / degraded / offline / idle。"""
    if project["check"] == "log":
        mtime = log.get("mtime")
        if mtime and now - mtime < 120:
            return "online"
        return "idle"
    if project["check"] == "off":
        return "idle"
    if health["ok"]:
        return "degraded" if log.get("recent_errors") else "online"
    if process == "running":
        return "degraded"  # 进程在但健康检查不通过
    return "offline"


# ---------------------------------------------------------------- 指标解析


def collect_metrics() -> dict:
    path = ROOT / "artifacts" / "logs" / "chatcaht.log"
    series: dict[str, list] = {key: [] for key in METRIC_PATTERNS}
    if not path.exists():
        return {"series": series}
    for line in tail_text(path, max_bytes=2_000_000):
        parsed = match_log_line(line)
        if not parsed:
            continue
        ts_str, _level, _logger, msg = parsed
        for key, pattern in METRIC_PATTERNS.items():
            hit = pattern.search(msg)
            if hit:
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    continue
                series[key].append({"t": ts, "v": float(hit.group(1))})
    for key in series:
        series[key] = series[key][-120:]
    return {"series": series}


# ---------------------------------------------------------------- 用户记忆

# 与 LoLLama memory/manager.py 的 LAYER_LABELS 保持一致
MEMORY_LAYER_LABELS = {"episodic": "情景", "semantic": "事实", "procedural": "偏好", "core": "画像"}


def _lollama_root() -> Path | None:
    raw = load_config().get("services", {}).get("lollama_dir")
    if not raw:
        return None
    base = Path(raw)
    if not base.is_absolute():
        base = ROOT / base
    return base.resolve()


def _lollama_config() -> dict:
    root = _lollama_root()
    if root is None:
        return {}
    for name in ("config.yaml", "config.example.yaml"):
        path = root / "configs" / name
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
            except Exception:
                return {}
    return {}


def collect_memory() -> dict:
    """读取 LoLLama 的记忆持久化文件（每个 user_id 一个 JSON，4 层）。

    只读展示；有效强度按 LoLLama 的公式现算：strength × 0.5^(闲置小时/半衰期)。
    """
    root = _lollama_root()
    if root is None:
        return {"users": [], "error": "config 未配置 lollama_dir"}
    lcfg = _lollama_config()
    memory_dir = (lcfg.get("paths") or {}).get("memory_dir", "artifacts/memory")
    mem_path = root / memory_dir
    layers_cfg = ((lcfg.get("memory") or {}).get("layers") or {})

    def half_life(layer: str) -> float:
        try:
            return float((layers_cfg.get(layer) or {}).get("half_life_hours", 0) or 0)
        except (TypeError, ValueError):
            return 0.0

    now = time.time()
    users = []
    if mem_path.exists():
        for file in sorted(mem_path.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                users.append({"user_id": file.stem, "error": f"{type(exc).__name__}: {exc}"})
                continue
            layers = {}
            for layer, label in MEMORY_LAYER_LABELS.items():
                items = []
                for item in data.get(layer) or []:
                    strength = float(item.get("strength", 0) or 0)
                    hl = half_life(layer)
                    if hl > 0:
                        idle_hours = max(0.0, (now - float(item.get("last_accessed", now) or now)) / 3600.0)
                        effective = strength * (0.5 ** (idle_hours / hl))
                    else:
                        effective = strength
                    items.append(
                        {
                            "text": item.get("text", ""),
                            "importance": item.get("importance"),
                            "strength": round(strength, 3),
                            "effective": round(effective, 3),
                            "hits": item.get("hits", 0),
                            "created_at": item.get("created_at"),
                            "last_accessed": item.get("last_accessed"),
                            "source": item.get("source", ""),
                        }
                    )
                items.sort(key=lambda x: x["effective"], reverse=True)
                layers[layer] = {"label": label, "count": len(items), "items": items}
            users.append(
                {
                    "user_id": file.stem,
                    "file": str(file),
                    "mtime": file.stat().st_mtime,
                    "total": sum(l["count"] for l in layers.values()),
                    "layers": layers,
                }
            )
    return {"users": users, "memory_dir": str(mem_path)}


# ---------------------------------------------------------------- 日志查询


def query_logs(name: str, lines: int, level: str, search: str) -> dict:
    path = log_registry().get(name or "")
    if path is None or not path.exists():
        return {"name": name, "entries": [], "error": "log not found"}
    entries = parse_entries(tail_text(path))
    if level and level != "ALL":
        order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
        floor = order.get(level, 0)
        entries = [e for e in entries if order.get(e["level"], 1) >= floor]
    if search:
        needle = search.lower()
        entries = [
            e
            for e in entries
            if needle in e["msg"].lower()
            or needle in e["logger"].lower()
            or any(needle in x.lower() for x in e["extra"])
        ]
    entries = entries[-max(10, min(lines, 2000)) :]
    stat = path.stat()
    return {
        "name": name,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "count": len(entries),
        "entries": entries,
    }


def list_logs() -> list[dict]:
    items = []
    for name, path in sorted(log_registry().items()):
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "name": name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "source": "project" if ":" in name else "chatcaht",
            }
        )
    return items


# ---------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    server_version = "ChatCahtDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            elif route == "/api/status":
                self._send_json(collect_status())
            elif route == "/api/metrics":
                self._send_json(collect_metrics())
            elif route == "/api/logs":
                qs = parse_qs(parsed.query)
                self._send_json(
                    query_logs(
                        name=qs.get("name", [""])[0],
                        lines=int(qs.get("lines", ["300"])[0]),
                        level=qs.get("level", ["ALL"])[0].upper(),
                        search=qs.get("q", [""])[0],
                    )
                )
            elif route == "/api/logfiles":
                self._send_json({"logs": list_logs()})
            elif route == "/api/memory":
                self._send_json(collect_memory())
            else:
                self.send_error(404)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        return  # 保持控制台安静；请求处理异常仍会正常打印 traceback


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatCaht 项目监控看板")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ChatCaht Dashboard: http://{args.host}:{args.port}  (Ctrl+C 停止)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
