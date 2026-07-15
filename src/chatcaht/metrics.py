from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class MetricsRecorder:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        metric: str,
        *,
        value_ms: float | None = None,
        session_id: str | None = None,
        turn_id: int | None = None,
        **detail: Any,
    ) -> None:
        if self.path is None:
            return
        numeric_value = None if value_ms is None else max(0.0, float(value_ms))
        event = {
            "ts": time.time(),
            "metric": metric,
            "value_ms": None if numeric_value is None else round(numeric_value, 3),
            "session_id": session_id,
            "turn_id": turn_id,
            **detail,
        }
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
