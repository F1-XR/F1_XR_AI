"""Evaluate the deterministic battle decision policy on historical intervals.

This does not make the rule "learned"; it makes it measurable. Given OpenF1
intervals JSON, the script walks battle windows, runs the same policy used by
the runtime agent, then measures future gap outcomes.

Examples:
  python -m scripts.eval_decision_policy --intervals data/intervals.json
  python -m scripts.eval_decision_policy --intervals data/intervals.json --json-out decision_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.decision_policy import recommend_battle_policy
from app.ml.state_estimator import GapEstimator


def _to_seconds(iso: str) -> float | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interp(series: list[tuple[float, float]], t: float) -> float | None:
    if not series or t < series[0][0] or t > series[-1][0]:
        return None
    return float(np.interp(t, [x for x, _ in series], [y for _, y in series]))


def _trend(series: list[tuple[float, float]], idx: int, lookback: float = 3.0) -> str:
    now_t, now_gap = series[idx]
    for j in range(idx - 1, -1, -1):
        t, gap = series[j]
        if now_t - t >= lookback:
            if now_gap < gap - 0.05:
                return "closing"
            if now_gap > gap + 0.05:
                return "opening"
            return "stable"
    return "stable"


def _kalman_prediction(series: list[tuple[float, float]], idx: int, horizon: float) -> tuple[float | None, float | None]:
    now_t = series[idx][0]
    est = GapEstimator(fuse_speed_delta=False)
    for t, gap in series[: idx + 1]:
        est.observe(t - now_t, gap)
    if not est.ready:
        return None, None
    mean, std = est.predict(horizon)
    if math.isnan(mean):
        return None, None
    return round(mean, 4), round(std, 4)


def _load_intervals(path: Path) -> dict[int, list[tuple[float, float]]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_driver: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        driver = row.get("driver_number")
        t = _to_seconds(row.get("date"))
        gap = _num(row.get("interval"))
        if driver is None or t is None or gap is None:
            continue
        by_driver[int(driver)].append((t, gap))
    return {driver: sorted(series) for driver, series in by_driver.items()}


def evaluate(
    intervals_path: Path,
    *,
    predict_horizon: float = 3.0,
    outcome_horizon: float = 30.0,
    max_gap: float = 3.0,
    stride: int = 3,
) -> dict[str, Any]:
    by_driver = _load_intervals(intervals_path)
    rows: list[dict[str, Any]] = []

    for driver, series in by_driver.items():
        if len(series) < 8:
            continue
        for idx in range(6, len(series), max(1, stride)):
            now_t, gap = series[idx]
            if gap > max_gap:
                continue
            future_gap = _interp(series, now_t + outcome_horizon)
            if future_gap is None:
                continue
            predicted_gap, predicted_std = _kalman_prediction(series, idx, predict_horizon)
            trend = _trend(series, idx)
            policy = recommend_battle_policy(
                gap_seconds=gap,
                predicted_gap_seconds=predicted_gap,
                predicted_gap_std_seconds=predicted_std,
                trend=trend,
                drs=gap <= 1.0,  # intervals-only proxy: DRS eligible, not actual activation.
                fusion_used=False,
                overtake_probability=None,
            )
            rows.append({
                "driver_number": driver,
                "t": now_t,
                "gap_seconds": gap,
                "future_gap_seconds": future_gap,
                "future_gap_delta": future_gap - gap,
                "gap_closed": future_gap <= gap - 0.10,
                "near_attack_window": future_gap <= 1.0,
                **{k: policy[k] for k in ("action", "confidence", "risk")},
            })

    by_action: dict[str, dict[str, Any]] = {}
    for action in sorted({row["action"] for row in rows}):
        subset = [row for row in rows if row["action"] == action]
        deltas = [row["future_gap_delta"] for row in subset]
        by_action[action] = {
            "n": len(subset),
            "gap_closed_rate": sum(row["gap_closed"] for row in subset) / len(subset) if subset else None,
            "near_attack_window_rate": sum(row["near_attack_window"] for row in subset) / len(subset) if subset else None,
            "mean_future_gap_delta": float(np.mean(deltas)) if deltas else None,
            "median_future_gap_delta": float(np.median(deltas)) if deltas else None,
        }

    return {
        "mode": "intervals_only",
        "intervals_path": str(intervals_path),
        "predict_horizon_sec": predict_horizon,
        "outcome_horizon_sec": outcome_horizon,
        "max_gap_sec": max_gap,
        "sample_count": len(rows),
        "by_action": by_action,
        "note": (
            "DRS is proxied as gap<=1.0 because intervals JSON alone has no car_data. "
            "Use this as a lightweight policy sanity check; richer evaluation can add car_data/positions."
        ),
    }


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Battle Decision Policy Evaluation",
        "",
        f"- mode: `{result['mode']}`",
        f"- samples: {result['sample_count']:,}",
        f"- prediction horizon: {result['predict_horizon_sec']}s",
        f"- outcome horizon: {result['outcome_horizon_sec']}s",
        "",
        "| action | samples | gap closed rate | near attack-window rate | mean future gap delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for action, block in result["by_action"].items():
        delta = block["mean_future_gap_delta"]
        lines.append(
            f"| {action} | {block['n']:,} | {_fmt_pct(block['gap_closed_rate'])} | "
            f"{_fmt_pct(block['near_attack_window_rate'])} | "
            f"{'-' if delta is None else f'{delta:+.3f}s'} |"
        )
    lines.extend(["", result["note"]])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", required=True, help="OpenF1 intervals JSON path.")
    parser.add_argument("--predict-horizon", type=float, default=3.0)
    parser.add_argument("--outcome-horizon", type=float, default=30.0)
    parser.add_argument("--max-gap", type=float, default=3.0)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()

    result = evaluate(
        Path(args.intervals),
        predict_horizon=args.predict_horizon,
        outcome_horizon=args.outcome_horizon,
        max_gap=args.max_gap,
        stride=args.stride,
    )
    print(to_markdown(result))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {out}")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(result), encoding="utf-8")
        print(f"[markdown] {out}")


if __name__ == "__main__":
    main()
