from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    # Windows 控制台可能是 GBK 等本地编码，转写文本里超出码表的字符会让
    # StreamHandler 反复报 UnicodeEncodeError；降级为转义而不是刷错误堆栈。
    try:
        sys.stderr.reconfigure(errors="backslashreplace")
    except Exception:
        pass
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
