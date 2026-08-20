"""Evaluate the Kalman gap estimator against a linear extrapolation baseline.

This script is meant to create an AI debug/evaluation artifact, not a product UI.
Use it to prove that Feature 1-A improves short-horizon gap prediction under noisy
telemetry and to capture uncertainty coverage.

Examples:
  python -m scripts.eval_state_estimator
  python -m scripts.eval_state_estimator --json-out eval/state_estimator_benchmark.json
  python -m scripts.eval_state_estimator --markdown-out eval/state_estimator_benchmark.md
  python -m scripts.eval_state_estimator --intervals path/to/intervals.json --driver 44
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.state_estimator import GapEstimator, linear_extrapolate


def _to_seconds(iso: str) -> float | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _metrics(errors: list[float]) -> dict[str, float | int | None]:
    if not errors:
        return {"n": 0, "mae": None, "rmse": None}
    arr = np.array(errors, dtype=float)
    return {
        "n": int(len(arr)),
        "mae": float(arr.mean()),
        "rmse": float(math.sqrt((arr ** 2).mean())),
    }


def _result_block(
    tag: str,
    kf_err: list[float],
    lin_err: list[float],
    cov: int,
    tot: int,
    horizon: float,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kf = _metrics(kf_err)
    lin = _metrics(lin_err)
    improvement = None
    if kf["mae"] is not None and lin["mae"] not in (None, 0):
        improvement = float(100 * (1 - float(kf["mae"]) / float(lin["mae"])))
    return {
        "tag": tag,
        "horizon_sec": horizon,
        "kalman": kf,
        "linear": lin,
        "mae_improvement_pct": improvement,
        "coverage_95_pct": float(100 * cov / tot) if tot else None,
        "coverage_samples": int(tot),
        "meta": meta or {},
    }


def _interp(series: list[tuple[float, float]], t: float) -> float | None:
    xs = [x for x, _ in series]
    if t < xs[0] or t > xs[-1]:
        return None
    ys = [y for _, y in series]
    return float(np.interp(t, xs, ys))


def eval_intervals(
    path: Path,
    driver: int | None,
    horizon: float,
    q: float,
    rvar: float,
) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_driver: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        driver_number = row.get("driver_number")
        t = _to_seconds(row.get("date"))
        gap = _num(row.get("interval"))
        if driver_number is None or t is None or gap is None:
            continue
        by_driver.setdefault(int(driver_number), []).append((t, gap))

    drivers = [driver] if driver is not None else sorted(by_driver)
    kf_err: list[float] = []
    lin_err: list[float] = []
    cov = 0
    tot = 0

    for driver_number in drivers:
        series = sorted(by_driver.get(driver_number, []))
        for i in range(6, len(series)):
            t_now, gap_now = series[i]
            future = _interp(series, t_now + horizon)
            if future is None or gap_now > 3.0:
                continue

            times = [t for t, _ in series[: i + 1]]
            gaps = [g for _, g in series[: i + 1]]
            est = GapEstimator(q_accel=q, gap_meas_var=rvar, fuse_speed_delta=False)
            for t, gap in zip(times, gaps):
                est.observe(t, gap)

            mean, std = est.predict(horizon)
            if math.isnan(mean):
                continue
            kf_err.append(abs(mean - future))
            if std > 0:
                tot += 1
                if abs(mean - future) <= 1.96 * std:
                    cov += 1

            baseline = linear_extrapolate(times, gaps, horizon)
            if baseline is not None:
                lin_err.append(abs(baseline - future))

    result = _result_block(
        "openf1_intervals",
        kf_err,
        lin_err,
        cov,
        tot,
        horizon,
        meta={"path": str(path), "driver": driver, "q_accel": q, "gap_meas_var": rvar},
    )
    _report("OpenF1 intervals", result)
    return result


def _make_synthetic(rng: np.random.Generator, n: int, meas_std: float, accel_std: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data = []
    for _ in range(n):
        ts = [0.0]
        while ts[-1] < 12.0:
            ts.append(ts[-1] + float(rng.uniform(0.25, 0.9)))
        ts_arr = np.array(ts)
        gap = float(rng.uniform(0.5, 1.6))
        rate = float(rng.uniform(-0.18, 0.05))
        true_gaps = []
        prev = ts_arr[0]
        for t in ts_arr:
            dt = t - prev
            prev = t
            rate += float(rng.normal(0, accel_std)) * math.sqrt(max(dt, 1e-3))
            gap = max(0.0, gap + rate * dt)
            true_gaps.append(gap)
        true_arr = np.array(true_gaps)
        meas_arr = true_arr + rng.normal(0, meas_std, size=len(true_arr))
        data.append((ts_arr, true_arr, meas_arr))
    return data


def _eval_synthetic_case(
    data: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    q: float,
    rvar: float,
    horizon: float,
) -> tuple[list[float], list[float], int, int]:
    kf_err: list[float] = []
    lin_err: list[float] = []
    cov = 0
    tot = 0

    for ts, true_gaps, measured_gaps in data:
        for i in range(6, len(ts)):
            target_t = ts[i] + horizon
            if target_t > ts[-1]:
                break
            future = float(np.interp(target_t, ts, true_gaps))
            times = list(ts[: i + 1])
            gaps = list(measured_gaps[: i + 1])

            est = GapEstimator(q_accel=q, gap_meas_var=rvar, fuse_speed_delta=False)
            for t, gap in zip(times, gaps):
                est.observe(float(t), float(gap))
            mean, std = est.predict(horizon)

            kf_err.append(abs(mean - future))
            if std > 0:
                tot += 1
                if abs(mean - future) <= 1.96 * std:
                    cov += 1

            baseline = linear_extrapolate(times, gaps, horizon)
            if baseline is not None:
                lin_err.append(abs(baseline - future))

    return kf_err, lin_err, cov, tot


def eval_synthetic(horizon: float, q: float) -> dict[str, Any]:
    rng = np.random.default_rng(11)
    regimes = [
        ("clean_noise_std_0_05", 0.05, 0.02, 0.0025),
        ("realistic_telemetry_std_0_12", 0.12, 0.03, 0.02),
        ("high_noise_low_maneuver_std_0_15", 0.15, 0.008, 0.02),
    ]

    print(f"Synthetic benchmark: Kalman vs linear baseline, horizon={horizon}s\n")
    print(f"| {'case':<34} | {'KF MAE':>8} | {'LIN MAE':>8} | {'improve':>8} | {'95% cov':>8} |")
    print(f"| {'-' * 34} | {'-' * 8} | {'-' * 8} | {'-' * 8} | {'-' * 8} |")

    results = []
    for name, meas_std, accel_std, gap_meas_var in regimes:
        data = _make_synthetic(rng, 400, meas_std, accel_std)
        kf_err, lin_err, cov, tot = _eval_synthetic_case(data, q, gap_meas_var, horizon)
        block = _result_block(
            name,
            kf_err,
            lin_err,
            cov,
            tot,
            horizon,
            meta={
                "measurement_std": meas_std,
                "accel_std": accel_std,
                "gap_meas_var": gap_meas_var,
                "q_accel": q,
            },
        )
        results.append(block)
        _print_table_row(block)

    print("\nInterpretation: use this as an AI evaluation artifact. Product XR should keep the final Kalman prediction and uncertainty only.")
    return {"mode": "synthetic", "horizon_sec": horizon, "results": results}


def _print_table_row(block: dict[str, Any]) -> None:
    kalman = block["kalman"]
    linear = block["linear"]
    improvement = block.get("mae_improvement_pct")
    coverage = block.get("coverage_95_pct")
    print(
        f"| {block['tag']:<34} | "
        f"{kalman['mae'] if kalman['mae'] is not None else float('nan'):8.4f} | "
        f"{linear['mae'] if linear['mae'] is not None else float('nan'):8.4f} | "
        f"{improvement if improvement is not None else float('nan'):+7.1f}% | "
        f"{coverage if coverage is not None else float('nan'):7.1f}% |"
    )


def _report(tag: str, result: dict[str, Any]) -> None:
    print(f"=== {tag} state-estimator benchmark ===")
    if result["kalman"]["n"] == 0:
        print("No evaluable battle windows found. Need continuous gap<3s observations.")
        return
    _print_table_row(result)


def _markdown(result: dict[str, Any]) -> str:
    if result.get("mode") == "synthetic":
        blocks = result.get("results", [])
        title = "Synthetic Linear vs Kalman Benchmark"
        horizon = result.get("horizon_sec")
    else:
        blocks = [result]
        title = "OpenF1 Intervals Linear vs Kalman Benchmark"
        horizon = result.get("horizon_sec")

    lines = [
        f"# {title}",
        "",
        f"- horizon: {horizon}s",
        "- baseline: linear extrapolation over the recent gap window",
        "- estimator: constant-velocity Kalman gap estimator with uncertainty",
        "",
        "| case | samples | Kalman MAE | Linear MAE | MAE improvement | 95% coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for block in blocks:
        kalman = block["kalman"]
        linear = block["linear"]
        improvement = block.get("mae_improvement_pct")
        coverage = block.get("coverage_95_pct")
        lines.append(
            "| {case} | {n:,} | {kf} | {lin} | {imp} | {cov} |".format(
                case=block["tag"],
                n=kalman["n"],
                kf="-" if kalman["mae"] is None else f"{kalman['mae']:.4f}s",
                lin="-" if linear["mae"] is None else f"{linear['mae']:.4f}s",
                imp="-" if improvement is None else f"{improvement:+.1f}%",
                cov="-" if coverage is None else f"{coverage:.1f}%",
            )
        )
    lines.extend(
        [
            "",
            "Recommended use: keep this as a debug/evaluation capture for the AI portfolio. In XR, show only the Kalman prediction and uncertainty band unless an explicit debug overlay is needed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", default=None, help="OpenF1 intervals JSON path.")
    parser.add_argument("--driver", type=int, default=None, help="Evaluate one driver number only.")
    parser.add_argument("--horizon", type=float, default=3.0, help="Prediction horizon in seconds.")
    parser.add_argument("--q", type=float, default=0.0008, help="Kalman process noise q_accel.")
    parser.add_argument("--rvar", type=float, default=0.01, help="Gap measurement variance for intervals mode.")
    parser.add_argument("--json-out", default=None, help="Write benchmark metrics to a JSON file.")
    parser.add_argument("--markdown-out", default=None, help="Write a compact markdown summary.")
    args = parser.parse_args()

    if args.intervals:
        path = Path(args.intervals)
        if not path.exists():
            raise SystemExit(f"intervals file not found: {path}")
        result = eval_intervals(path, args.driver, args.horizon, args.q, args.rvar)
    else:
        result = eval_synthetic(args.horizon, args.q)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] {out}")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown(result), encoding="utf-8")
        print(f"[markdown] {out}")


if __name__ == "__main__":
    main()
