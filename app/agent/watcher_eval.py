"""능동 안내(watcher) 오탐 평가용 로깅.

watcher가 "N번 곧 추월"이라 안내할 때마다 그 '예측'을 jsonl 한 줄로 남긴다.
나중에 scripts/eval_watcher.py 가 이 로그 + 세션 실제 순위를 대조해
'예측 시각 이후 실제로 추월(순위 상승)이 일어났는지'를 판정하고
오탐률·Precision@K·분당 오탐 수를 집계한다. (여기선 기록만 — 판정/집계는 오프라인)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)


def log_prediction(session_key: int, at_time: str | None,
                   driver_number: int, probability: float) -> None:
    """안내 1건을 jsonl로 적재. 실패해도 안내 흐름을 막지 않는다(조용히 무시)."""
    if not settings.watcher_eval_enabled:
        return
    try:
        path = Path(settings.watcher_eval_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "logged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_key": session_key,
            "at_time": at_time,          # 예측 기준 리플레이 시각(cutoff)
            "driver_number": driver_number,
            "probability": round(float(probability), 4),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("[watcher_eval] 예측 로그 적재 실패(무시)")
