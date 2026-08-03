"""트랙 기준선 — '좌표 → 진행률(track_progress)' 변환용.

원리(파이프라인 _build_track_reference와 동일):
  깨끗한 한 바퀴(lap>1·pit-out 아님)의 location 좌표를 이어 트랙 모양을 만들고,
  누적 거리(0~1)를 progress로 둔다. 이후 어떤 좌표든 이 모양에 '최근접 투영'하면 진행률이 나온다.

비용 설계:
  - 트랙 모양(기준선) = '사실'이라 **경기(session)당 1회만** 만들어 캐시(재예측 아님, 취지 훼손 없음).
  - 진행률 계산 = 저장된 모양에 좌표 1개 투영(numpy argmin) → 매우 가벼움.
  - 추월 '확률'은 여기서 하지 않는다(그건 항상 실시간).
sklearn KDTree 대신 numpy 브루트포스 최근접(기준점 ~1000개라 충분히 빠름) — 의존성 최소화.
"""
from __future__ import annotations

import datetime as _dt

from ..data import openf1

# session_key -> {"xy": ndarray(N,2), "progress": ndarray(N)} | None(구성 실패)
_cache: dict[int, dict | None] = {}
_MAX_LAP_CANDIDATES = 40   # 기준선용 깨끗한 랩 시도 상한(비용 제한)


def _iso_plus(iso: str, seconds: float) -> str | None:
    try:
        t = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (t + _dt.timedelta(seconds=seconds)).isoformat()
    except (ValueError, AttributeError):
        return None


async def get_reference(session_key: int):
    """이 경기의 트랙 기준선(캐시). 없으면 1회 구성. 구성 실패면 None(→track_progress 결측)."""
    if session_key in _cache:
        return _cache[session_key]
    ref = await _build(session_key)
    _cache[session_key] = ref
    return ref


async def _build(session_key: int):
    import numpy as np

    laps = await openf1.get_laps(session_key)   # 전 드라이버
    cands = [
        l for l in laps
        if l.get("date_start") and l.get("lap_duration")
        and (l.get("lap_number") or 0) > 1 and not l.get("is_pit_out_lap")
    ]
    cands.sort(key=lambda l: l["date_start"])

    for l in cands[:_MAX_LAP_CANDIDATES]:
        dn = l.get("driver_number")
        start = l["date_start"]
        end = _iso_plus(start, float(l["lap_duration"]))
        if dn is None or end is None:
            continue
        pts = await openf1.get_location_window(session_key, dn, start, end)
        xy = [(p.get("x"), p.get("y")) for p in pts
              if p.get("x") is not None and p.get("y") is not None]
        if len(xy) < 200:
            continue
        arr = np.array(xy, dtype=float)
        if len(arr) > 1200:                          # 과밀 좌표 다운샘플
            arr = arr[np.linspace(0, len(arr) - 1, 1200).astype(int)]
        keep = np.r_[True, np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-6]
        arr = arr[keep]                              # 중복점 제거
        if len(arr) < 50:
            continue
        dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(arr, axis=0), axis=1))]
        if dist[-1] <= 0:
            continue
        return {"xy": arr, "progress": dist / dist[-1]}

    return None   # 깨끗한 랩 못 찾음(사고 많은 경기 등) → track_progress는 결측 폴백


def project(ref: dict, x: float, y: float) -> float:
    """좌표 (x,y)를 기준선에 최근접 투영 → 진행률(0~1)."""
    import numpy as np
    d2 = ((ref["xy"] - np.array([x, y], dtype=float)) ** 2).sum(axis=1)
    return float(ref["progress"][int(d2.argmin())])
