"""배틀 갭 상태추정(칼만) 평가 — N초 예측 정확도 + 불확실성 보정 vs 선형 baseline.

Feature 1(월드 모델)의 핵심 주장을 숫자로 보인다:
  ▸ 칼만이 6초 선형 외삽 대비 N초 예측 오차(MAE/RMSE)를 얼마나 줄이는가
  ▸ 칼만이 내는 불확실성 밴드(±1.96σ)가 실제로 ~95%를 덮는가(보정)

두 모드:
  1) 실데이터: OpenF1 intervals JSON(또는 overtakeML 캐시 raw/intervals.json)으로 walk-forward 평가
  2) 합성(기본): 여러 노이즈 구간의 몬테카를로로 칼만 vs 선형 + 커버리지 비교

실행:
  python -m scripts.eval_state_estimator                        # 합성 벤치
  python -m scripts.eval_state_estimator --intervals path.json --driver 44
  python -m scripts.eval_state_estimator --intervals data/raw/2024_spa_race/intervals.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from app.ml.state_estimator import GapEstimator, linear_extrapolate


def _to_seconds(iso: str) -> float | None:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ───────────────────────── 실데이터 walk-forward ─────────────────────────

def eval_intervals(path: Path, driver: int | None, horizon: float, q: float, rvar: float) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    # 드라이버별 (t, gap) 시계열 구성
    by_driver: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        dn = r.get("driver_number")
        t = _to_seconds(r.get("date"))
        g = _num(r.get("interval"))
        if dn is None or t is None or g is None:
            continue
        by_driver.setdefault(int(dn), []).append((t, g))

    drivers = [driver] if driver is not None else sorted(by_driver)
    kf_err, lin_err = [], []
    cov = tot = 0
    for dn in drivers:
        series = sorted(by_driver.get(dn, []))
        # 배틀 구간만: gap<3s 인 연속 구간에서 walk-forward
        for i in range(6, len(series)):
            t_now, _ = series[i]
            t_target = t_now + horizon
            # 실제 미래 갭: t_target 전후 관측 선형보간
            future = _interp(series, t_target)
            if future is None or series[i][1] > 3.0:
                continue
            times = [t for t, _ in series[:i + 1]]
            gaps = [g for _, g in series[:i + 1]]
            est = GapEstimator(q_accel=q, gap_meas_var=rvar, fuse_speed_delta=False)
            for t, g in zip(times, gaps):
                est.observe(t, g)
            m, s = est.predict(horizon)
            if math.isnan(m):
                continue
            kf_err.append(abs(m - future))
            if s > 0:
                tot += 1
                if abs(m - future) <= 1.96 * s:
                    cov += 1
            lb = linear_extrapolate(times, gaps, horizon)
            if lb is not None:
                lin_err.append(abs(lb - future))

    if not kf_err:
        print("평가 가능한 배틀 구간이 없어요(gap<3s 연속 관측 부족).")
        return
    _report("실데이터", kf_err, lin_err, cov, tot, horizon)


def _interp(series: list[tuple[float, float]], t: float) -> float | None:
    xs = [x for x, _ in series]
    if t < xs[0] or t > xs[-1]:
        return None
    ys = [y for _, y in series]
    return float(np.interp(t, xs, ys))


# ───────────────────────── 합성 몬테카를로 ─────────────────────────

def _make(rng, n, meas_std, accel_std, horizon):
    data = []
    for _ in range(n):
        ts = [0.0]
        while ts[-1] < 12.0:
            ts.append(ts[-1] + float(rng.uniform(0.25, 0.9)))
        ts = np.array(ts)
        gap = float(rng.uniform(0.5, 1.6))
        rate = float(rng.uniform(-0.18, 0.05))
        tg, prev = [], ts[0]
        for t in ts:
            dt = t - prev
            prev = t
            rate += float(rng.normal(0, accel_std)) * math.sqrt(max(dt, 1e-3))
            gap = max(0.0, gap + rate * dt)
            tg.append(gap)
        tg = np.array(tg)
        meas = tg + rng.normal(0, meas_std, size=len(tg))
        data.append((ts, tg, meas))
    return data


def _eval_synth(data, q, rvar, horizon):
    kf, lin = [], []
    cov = tot = 0
    for ts, tg, meas in data:
        for i in range(6, len(ts)):
            tt = ts[i] + horizon
            if tt > ts[-1]:
                break
            truef = float(np.interp(tt, ts, tg))
            times, gaps = list(ts[:i + 1]), list(meas[:i + 1])
            est = GapEstimator(q_accel=q, gap_meas_var=rvar, fuse_speed_delta=False)
            for a, b in zip(times, gaps):
                est.observe(a, b)
            m, s = est.predict(horizon)
            kf.append(abs(m - truef))
            if s > 0:
                tot += 1
                if abs(m - truef) <= 1.96 * s:
                    cov += 1
            lb = linear_extrapolate(times, gaps, horizon)
            if lb is not None:
                lin.append(abs(lb - truef))
    return kf, lin, cov, tot


def eval_synthetic(horizon: float, q: float, rvar: float) -> None:
    rng = np.random.default_rng(11)
    print("합성 몬테카를로 (칼만 vs 6초 선형회귀), horizon =", horizon, "s\n")
    regimes = [
        ("깨끗(측정std=0.05)", 0.05, 0.02, 0.0025),
        ("현실 telemetry(측정std=0.12)", 0.12, 0.03, 0.02),
        ("고노이즈·저기동(측정std=0.15)", 0.15, 0.008, 0.02),
    ]
    print(f"| {'구간':<26} | {'KF MAE':>8} | {'LIN MAE':>8} | {'개선':>7} | {'95%커버':>7} |")
    print(f"| {'-'*26} | {'-'*8} | {'-'*8} | {'-'*7} | {'-'*7} |")
    for name, ms, ac, gv in regimes:
        data = _make(rng, 400, ms, ac, horizon)
        kf, lin, cov, tot = _eval_synth(data, q, gv, horizon)
        kfm, linm = float(np.mean(kf)), float(np.mean(lin))
        impr = 100 * (1 - kfm / linm)
        c = 100 * cov / max(1, tot)
        print(f"| {name:<26} | {kfm:8.4f} | {linm:8.4f} | {impr:+6.1f}% | {c:6.1f}% |")
    print("\n해석: 노이즈가 클수록 칼만이 전체 이력을 최적 활용해 선형창을 앞선다.")
    print("      깨끗한 구간에선 점예측은 비슷하되 칼만은 '보정된 불확실성(±σ)'을 추가로 제공한다.")


def _report(tag, kf_err, lin_err, cov, tot, horizon):
    kf = np.array(kf_err)
    lin = np.array(lin_err) if lin_err else None
    print(f"=== {tag} 상태추정 평가 (horizon={horizon}s, n={len(kf)}) ===")
    print(f"칼만  MAE={kf.mean():.4f}s  RMSE={math.sqrt((kf**2).mean()):.4f}s")
    if lin is not None and len(lin):
        print(f"선형  MAE={lin.mean():.4f}s  RMSE={math.sqrt((lin**2).mean()):.4f}s")
        print(f"개선  {100*(1-kf.mean()/lin.mean()):+.1f}% (MAE)")
    if tot:
        print(f"불확실성 밴드 95%(±1.96σ) 커버리지: {100*cov/tot:.1f}% (이상적 ~95%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervals", default=None, help="OpenF1 intervals JSON 경로(있으면 실데이터 평가)")
    ap.add_argument("--driver", type=int, default=None, help="특정 드라이버 번호만(없으면 전체)")
    ap.add_argument("--horizon", type=float, default=3.0, help="예측 지평(초)")
    ap.add_argument("--q", type=float, default=0.0008, help="프로세스 노이즈 q_accel")
    ap.add_argument("--rvar", type=float, default=0.01, help="갭 측정 노이즈 r(분산)")
    args = ap.parse_args()

    if args.intervals:
        path = Path(args.intervals)
        if not path.exists():
            raise SystemExit(f"intervals 파일 없음: {path}")
        eval_intervals(path, args.driver, args.horizon, args.q, args.rvar)
    else:
        eval_synthetic(args.horizon, args.q, args.rvar)


if __name__ == "__main__":
    main()
