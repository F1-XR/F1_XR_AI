"""실시간 상태 추정 — 배틀 갭 칼만 필터(월드 모델 첫 조각).

역할:
    노이즈 낀·불규칙 샘플링의 갭(intervals) 관측을 융합해 '진짜 상태'를 추정하고
    N초 뒤를 **불확실성(±std)과 함께** 예측한다. 기존 선형 외삽(자 대고 직선)을
    원칙적인 필터로 대체한다.

상태(state):  x = [gap, gap_rate]   (gap[초], gap_rate[초/초] = 초당 좁혀지는 속도, 음수=접근)
모델(motion): 등속(constant-velocity) — gap(t+dt) = gap + gap_rate*dt
관측(measure):
    A) gap        : intervals 의 갭 (state[0] 직접 관측)
    B) gap_rate   : speed_delta(상대속도) 를 갭 변화율로 환산한 '두 번째 센서'(다신호 융합)

출력:
    현재 추정 gap / gap_rate / 불확실성(std), 그리고 predict(horizon) -> (mean, std).
    이 std 가 Battle Lens 불확실성 밴드(Feature 5) 와 보정 평가(Feature 2)로 이어진다.

의존성: numpy 만. 앱은 이미 numpy 를 쓴다(lightgbm/predict).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ── B) speed_delta(상대속도, km/h) → gap_rate(초/초) 환산 게인 ──
#   정확 변환엔 트랙 거리·기준속도가 필요하다. 여기선 "상대속도가 클수록 갭이 빨리 준다"는
#   물리 방향만 살린 스케일 게인(휴리스틱)으로 두 번째 센서를 만든다(관측 노이즈 크게).
#   부호: subject가 앞차보다 빠르면(speed_delta>0) 갭은 줄어든다 → gap_rate < 0.
_SPEED_DELTA_TO_GAP_RATE = -0.004     # (초/초) per (km/h). 데이터로 튜닝 가능.
_DEFAULT_REF_SPEED_KMH = 220.0        # 참고: 갭_rate ≈ -Δspeed/ref 도 가능(대안식)


@dataclass
class KalmanCV1D:
    """1D 등속 칼만 필터. 상태 [위치, 속도]. 여기서 위치=gap, 속도=gap_rate.

    q_accel : 프로세스 노이즈(가속도 분산). 클수록 모델을 덜 믿고 측정을 빨리 따라감.
    """

    q_accel: float = 0.05
    x: np.ndarray = field(default_factory=lambda: np.zeros(2))
    P: np.ndarray = field(default_factory=lambda: np.diag([1.0, 1.0]).astype(float))
    initialized: bool = False

    def reset(self, gap0: float, rate0: float = 0.0,
              gap_var: float = 0.25, rate_var: float = 1.0) -> None:
        self.x = np.array([gap0, rate0], dtype=float)
        self.P = np.diag([gap_var, rate_var]).astype(float)
        self.initialized = True

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]])

    def _Q(self, dt: float) -> np.ndarray:
        # 이산 백색 가속 모델(discrete white-noise acceleration)
        q = self.q_accel
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([[dt4 / 4.0, dt3 / 2.0],
                             [dt3 / 2.0, dt2]])

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update_gap(self, z: float, r: float = 0.05) -> None:
        """갭 측정(state[0]) 보정."""
        self._update(z, np.array([1.0, 0.0]), r)

    def update_rate(self, z: float, r: float = 1.0) -> None:
        """갭 변화율 측정(state[1]) 보정 — B) speed_delta 융합용."""
        self._update(z, np.array([0.0, 1.0]), r)

    def _update(self, z: float, H: np.ndarray, r: float) -> None:
        """스칼라 측정 z, 측정행렬 H(1D, shape (2,)) 로 보정."""
        if not self.initialized:
            return
        y = float(z - H @ self.x)              # innovation (스칼라)
        S = float(H @ self.P @ H) + r          # innovation covariance (스칼라)
        if S <= 1e-12:
            return
        K = (self.P @ H) / S                   # Kalman gain (2,)
        self.x = self.x + K * y
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def predict_ahead(self, horizon: float) -> tuple[float, float]:
        """현재 상태에서 horizon 초 뒤 gap 의 (평균, 표준편차)."""
        F = self._F(horizon)
        xf = F @ self.x
        Pf = F @ self.P @ F.T + self._Q(horizon)
        return float(xf[0]), float(math.sqrt(max(0.0, Pf[0, 0])))


@dataclass
class GapEstimator:
    """배틀 갭 상태추정기. 시간순 갭 관측을 넣으면 매끈한 추정 + 불확실 예측을 준다.

    사용:
        est = GapEstimator()
        est.observe(t0, gap0); est.observe(t1, gap1); ...
        est.observe_speed_delta(t1, speed_delta_kmh)      # (B) 선택: 다신호 융합
        mean, std = est.predict(horizon=3.0)              # 3초 뒤 갭 (±std)
    """

    # 아래 노이즈 기본값은 합성/현실 telemetry에서 튜닝한 값이다(scripts/eval_state_estimator.py로
    # 실데이터에 재튜닝 권장). 고노이즈 구간에서 6초 선형회귀 대비 MAE ~16% 개선 + 95% 밴드 보정.
    q_accel: float = 0.0008           # 프로세스 노이즈(가속도). 작을수록 CV를 더 믿어 매끈(강건).
    gap_meas_var: float = 0.01        # 갭 측정 노이즈 r (std ~0.1s 상당). 데이터에 맞춰 튜닝.
    rate_meas_var: float = 0.8        # speed_delta 유도 rate 노이즈 r (약한 센서 → 크게)
    fuse_speed_delta: bool = True     # (B) speed_delta 융합 on/off
    kf: KalmanCV1D = field(default_factory=KalmanCV1D)
    _last_t: float | None = None

    def __post_init__(self) -> None:
        self.kf.q_accel = self.q_accel

    # ── 관측 입력 ──
    def observe(self, t: float, gap: float) -> None:
        """t(초), gap(초) 갭 관측 1건. 시간은 임의 간격(불규칙) 허용."""
        if gap is None or not math.isfinite(gap):
            return
        if not self.kf.initialized:
            self.kf.reset(gap0=float(gap))
            self._last_t = t
            return
        dt = t - self._last_t if self._last_t is not None else 0.0
        if dt < 0:
            return                       # 과거로 역행하는 관측은 무시
        self.kf.predict(dt)
        self.kf.update_gap(float(gap), r=self.gap_meas_var)
        self._last_t = t

    def observe_speed_delta(self, t: float, speed_delta_kmh: float) -> None:
        """(B) 상대속도(subject - 앞차, km/h)를 gap_rate 관측으로 융합.
        같은 t 의 gap 관측 '뒤'에 부르는 것을 권장(같은 시각 보정)."""
        if not self.fuse_speed_delta or speed_delta_kmh is None or not math.isfinite(speed_delta_kmh):
            return
        if not self.kf.initialized:
            return
        # 같은 시각이면 predict 없이 rate 만 보정. 앞선 dt 는 observe 에서 이미 처리됨.
        rate_meas = _SPEED_DELTA_TO_GAP_RATE * float(speed_delta_kmh)
        self.kf.update_rate(rate_meas, r=self.rate_meas_var)

    # ── 상태/예측 조회 ──
    @property
    def ready(self) -> bool:
        return self.kf.initialized

    @property
    def gap(self) -> float:
        return float(self.kf.x[0])

    @property
    def gap_rate(self) -> float:
        return float(self.kf.x[1])

    @property
    def gap_std(self) -> float:
        return float(math.sqrt(max(0.0, self.kf.P[0, 0])))

    def predict(self, horizon: float = 3.0, clamp_nonneg: bool = True) -> tuple[float, float]:
        """horizon 초 뒤 갭 (평균, 표준편차). 평균은 0 미만이면 0으로 클램프(실제 갭은 음수 불가)."""
        if not self.kf.initialized:
            return float("nan"), float("nan")
        mean, std = self.kf.predict_ahead(horizon)
        if clamp_nonneg:
            mean = max(0.0, mean)
        return mean, std


def estimate_from_series(
    times: list[float],
    gaps: list[float],
    speed_deltas: list[float] | None = None,
    horizon: float = 3.0,
    **kwargs,
) -> dict:
    """시간순 시계열을 한 번에 넣어 최종 상태 + horizon 예측을 얻는 편의 함수.

    times/gaps 는 같은 길이. speed_deltas 를 주면 (B) 다신호 융합.
    반환: {gap, gap_rate, gap_std, pred_mean, pred_std, n}
    """
    est = GapEstimator(**kwargs)
    n = 0
    for i, (t, g) in enumerate(zip(times, gaps)):
        if g is None or not math.isfinite(g):
            continue
        est.observe(t, g)
        if speed_deltas is not None and i < len(speed_deltas):
            sd = speed_deltas[i]
            if sd is not None and math.isfinite(sd):
                est.observe_speed_delta(t, sd)
        n += 1
    if not est.ready:
        return {"ready": False, "n": n}
    mean, std = est.predict(horizon)
    return {
        "ready": True, "n": n,
        "gap": round(est.gap, 4),
        "gap_rate": round(est.gap_rate, 4),
        "gap_std": round(est.gap_std, 4),
        "pred_mean": round(mean, 4),
        "pred_std": round(std, 4),
        "horizon": horizon,
    }


def linear_extrapolate(times: list[float], gaps: list[float], horizon: float = 3.0,
                       lookback: float = 6.0, clamp_delta: float = 2.0) -> float | None:
    """기존 show_battle_context 의 선형 외삽 baseline(비교용).
    최근 lookback 초의 기울기로 horizon 초 뒤 갭을 외삽(±clamp_delta 제한)."""
    pts = []
    if not times:
        return None
    t_end = times[-1]
    for t, g in zip(times, gaps):
        if g is None or not math.isfinite(g):
            continue
        dt = t - t_end
        if dt >= -lookback:
            pts.append((dt, g))
    if len(pts) < 2:
        return None
    n = len(pts)
    sx = sum(t for t, _ in pts); sy = sum(g for _, g in pts)
    sxx = sum(t * t for t, _ in pts); sxy = sum(t * g for t, g in pts)
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if abs(denom) > 1e-9 else 0.0
    gap_now = pts[-1][1]
    delta = max(-clamp_delta, min(clamp_delta, slope * horizon))
    return max(0.0, gap_now + delta)
