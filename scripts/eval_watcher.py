"""능동 안내(watcher) 오탐 평가 — 오프라인 집계.

app/agent/watcher_eval.py 가 남긴 예측 로그(jsonl)를 읽어, 각 예측이
'예측 시각 이후 window초 안에 실제 추월(순위 상승)로 이어졌는지'를 세션 순위 데이터와 대조한다.
출력: 안내 수, 적중/오탐, Precision, Precision@K, 분당 오탐 수(False Alerts/min).

실행:  python -m scripts.eval_watcher [--window 30] [--log logs/watcher_eval.jsonl] [--k 5]
주의:  pit로 인한 순위변화도 '추월'로 포함될 수 있음(초기 지표). 트랙 배틀만 보려면 향후 pit 제외.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.data import openf1


def _parse(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def _pos_at(rows: list[dict], driver: int, t: datetime):
    """driver의 t 시각 이하 '가장 최근' position(없으면 None)."""
    best_dt, best_pos = None, None
    for r in rows:
        if r.get("driver_number") != driver:
            continue
        dt, pos = _parse(r.get("date")), r.get("position")
        if dt is None or pos is None or dt > t:
            continue
        if best_dt is None or dt > best_dt:
            best_dt, best_pos = dt, pos
    return best_pos


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/watcher_eval.jsonl")
    ap.add_argument("--window", type=float, default=30.0, help="예측 후 추월 확인 창(초)")
    ap.add_argument("--k", type=int, default=5, help="Precision@K")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"로그 없음: {path}  (능동 안내가 한 번도 안 울렸거나 watcher_eval_enabled=false)")
        return
    preds = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    preds = [p for p in preds if p.get("at_time") and p.get("session_key") is not None]
    if not preds:
        print("평가할 예측이 없어요(at_time 있는 로그가 없음).")
        return

    # 세션별 순위 데이터 한 번씩만 조회
    positions: dict[int, list[dict]] = {}
    for s in {p["session_key"] for p in preds}:
        positions[s] = await openf1.get_positions(s)

    judged: list[tuple[float, bool]] = []   # (probability, hit)
    undecided = 0
    for p in preds:
        t0 = _parse(p["at_time"])
        if t0 is None:
            undecided += 1
            continue
        rows = positions.get(p["session_key"], [])
        p0 = _pos_at(rows, p["driver_number"], t0)
        p1 = _pos_at(rows, p["driver_number"], t0 + timedelta(seconds=args.window))
        if p0 is None or p1 is None:
            undecided += 1
            continue
        judged.append((float(p.get("probability", 0.0)), p1 < p0))   # 순위 숫자↓ = 상승 = 추월

    total = len(judged)
    hits = sum(1 for _, h in judged if h)
    misses = total - hits

    # 분당 오탐: 관측된 at_time 범위(분) 기준
    times = sorted(t for t in (_parse(p["at_time"]) for p in preds) if t)
    span_min = (times[-1] - times[0]).total_seconds() / 60.0 if len(times) > 1 else 0.0

    # Precision@K: 확률 상위 K개 중 적중률
    topk = sorted(judged, key=lambda x: x[0], reverse=True)[: args.k]

    print("=== 능동 안내 오탐 평가 ===")
    print(f"안내 수: {len(preds)}  | 판정가능: {total}  | 판정불가(데이터부족): {undecided}")
    print(f"적중(실제 추월): {hits}  | 오탐(추월 없음): {misses}")
    if total:
        print(f"Precision(적중/판정): {hits / total:.3f}")
    if topk:
        print(f"Precision@{args.k}: {sum(1 for _, h in topk if h) / len(topk):.3f}")
    if span_min > 0:
        print(f"분당 오탐(False Alerts/min): {misses / span_min:.2f}  (관측 {span_min:.1f}분)")
    print(f"기준: window={args.window}s, 순위 상승=추월로 판정(pit 포함)")


if __name__ == "__main__":
    asyncio.run(main())
