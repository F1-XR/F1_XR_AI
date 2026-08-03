"""단일시점 피처 빌더 — (session_key, at_time, driver) → 추월 예측 모델 입력 피처.

원칙:
  - **at_time 이하 데이터만** 사용한다(스포일러 방지). 미래 값은 절대 넣지 않는다.
  - 계산 가능한 피처만 dict로 채우고, 나머지는 predict가 계약의 결측값(-1.0)으로 메운다.
    (모델이 fillna(-1.0)로 학습돼 결측 피처를 그대로 처리한다 → 단계적 확장 가능.)

커버 현황(car_data + location 배선 반영, 18/26):
  계산: season, gap_ahead, gap_trend, position, is_lap1, tyre_age, sector,
        air_temperature, track_temperature, humidity, rainfall,
        speed, drs_active, speed_delta,                         (← car_data 시간창)
        track_progress, track_progress_sin/cos, segment          (← location 기준선 투영)
  결측(-1.0로 채움, 중요도 낮음): tyre_age_delta, position_delta, same_lap,
        restart_phase, circuit_key·circuit_type_code, weather_regime_code, drs_range.
실측(measure_step4.py): 축소 0.745 → +car/sector 0.780 → +track_progress 0.806 (gap baseline 0.787 넘김).
"""
from __future__ import annotations

import datetime as _dt
import math

from ..data import openf1
from . import track_ref


def _num(v):
    """숫자로 바꿀 수 있으면 float, 아니면 None(예: '+1 LAP')."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _before(rows: list[dict], cutoff: str | None, key: str = "date") -> list[dict]:
    """cutoff(ISO) 이하의 기록만. cutoff 없으면 전체. (ISO 문자열 비교 = 시간 순서)"""
    if not cutoff:
        return [r for r in rows if r.get(key)]
    return [r for r in rows if r.get(key) and r[key] <= cutoff]


def _seconds_between(iso_a: str, iso_b: str) -> float:
    try:
        a = _dt.datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        b = _dt.datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
        return abs((b - a).total_seconds())
    except (ValueError, AttributeError):
        return 0.0


def _iso_minus(iso: str, seconds: float) -> str | None:
    """ISO 시각에서 N초 뺀 ISO 문자열(car_data 시간창 시작점 계산용)."""
    try:
        t = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (t - _dt.timedelta(seconds=seconds)).isoformat()
    except (ValueError, AttributeError):
        return None


async def build_features(session_key: int, at_time: str | None, driver_number: int) -> dict:
    """계산 가능한 피처만 담은 dict를 반환한다(키=피처명). 나머지는 predict가 -1.0으로 채운다."""
    cutoff = at_time
    feats: dict[str, float] = {}

    # season — at_time 연도
    if cutoff and len(cutoff) >= 4 and cutoff[:4].isdigit():
        feats["season"] = float(cutoff[:4])

    # intervals → gap_ahead, gap_trend
    iv = _before(await openf1.get_intervals(session_key, driver_number), cutoff)
    iv.sort(key=lambda r: r["date"])
    if iv:
        g_now = _num(iv[-1].get("interval"))   # 앞차와의 간격(초)
        if g_now is not None:
            feats["gap_ahead"] = g_now
            # gap_trend: 최근값 - 약 3초 전 값 (음수 = 좁혀지는 중 = 추월 압박)
            t_now = iv[-1]["date"]
            for r in reversed(iv[:-1]):
                if _seconds_between(r["date"], t_now) >= 3.0:
                    g_prev = _num(r.get("interval"))
                    if g_prev is not None:
                        feats["gap_trend"] = g_now - g_prev
                    break

    # position → position
    pos = [
        r for r in _before(await openf1.get_positions(session_key), cutoff)
        if r.get("driver_number") == driver_number and r.get("position") is not None
    ]
    pos.sort(key=lambda r: r["date"])
    if pos:
        feats["position"] = float(pos[-1]["position"])

    # laps → current_lap, is_lap1, sector
    laps = [
        r for r in _before(await openf1.get_laps(session_key, driver_number), cutoff, key="date_start")
        if r.get("lap_number") is not None
    ]
    cur_lap_rec = max(laps, key=lambda r: r["lap_number"]) if laps else None
    current_lap = int(cur_lap_rec["lap_number"]) if cur_lap_rec else None
    if current_lap is not None:
        feats["is_lap1"] = 1.0 if current_lap == 1 else 0.0

        # sector: 현재 랩 시작 이후 경과시간 vs 섹터 소요시간(1/2/3)
        s1 = _num(cur_lap_rec.get("duration_sector_1"))
        s2 = _num(cur_lap_rec.get("duration_sector_2"))
        start_iso = cur_lap_rec.get("date_start")
        if s1 is not None and start_iso and cutoff:
            elapsed = _seconds_between(start_iso, cutoff)
            if elapsed < s1:
                feats["sector"] = 1.0
            elif s2 is not None and elapsed < s1 + s2:
                feats["sector"] = 2.0
            else:
                feats["sector"] = 3.0

        # stints → tyre_age (현재 타이어의 사용 랩 수)
        stints = await openf1.get_stints(session_key, driver_number)
        cur = None
        for s in stints:
            ls, le = s.get("lap_start"), s.get("lap_end")
            if ls is not None and ls <= current_lap and (le is None or current_lap <= le):
                cur = s
        if cur is None:   # 폴백: 지금까지 시작된 스틴트 중 마지막
            started = [s for s in stints if s.get("lap_start") is not None and s["lap_start"] <= current_lap]
            cur = max(started, key=lambda s: s["lap_start"]) if started else None
        if cur is not None:
            age = (current_lap - int(cur["lap_start"])) + int(cur.get("tyre_age_at_start") or 0)
            feats["tyre_age"] = float(age)

    # weather → air_temperature, track_temperature, humidity, rainfall (최신 ≤ cutoff)
    weather = _before(await openf1.get_weather(session_key), cutoff)
    weather.sort(key=lambda r: r["date"])
    if weather:
        w = weather[-1]
        for field in ("air_temperature", "track_temperature", "humidity", "rainfall"):
            v = _num(w.get(field))
            if v is not None:
                feats[field] = v

    # car_data(시점 근처 창) → speed, drs_active, speed_delta
    # 세션 전체 car_data는 초대용량이라 [cutoff-6s, cutoff] 창 + 드라이버 1명만 조회.
    start_iso = _iso_minus(cutoff, 6.0) if cutoff else None
    if start_iso:
        car = [c for c in await openf1.get_car_data_window(session_key, driver_number, start_iso, cutoff)
               if c.get("date")]
        car.sort(key=lambda c: c["date"])
        if car:
            now = car[-1]
            sp = _num(now.get("speed"))
            if sp is not None:
                feats["speed"] = sp
                feats["drs_active"] = 1.0 if now.get("drs") in (10, 12, 14) else 0.0
                # speed_delta: 약 3초 전 speed와 차이
                t_now = now["date"]
                for c in reversed(car[:-1]):
                    if _seconds_between(c["date"], t_now) >= 3.0:
                        sp_prev = _num(c.get("speed"))
                        if sp_prev is not None:
                            feats["speed_delta"] = sp - sp_prev
                        break

    # track_progress(위치 기하) → 현재 좌표를 경기 트랙 기준선에 투영
    # 기준선은 경기당 1회 캐시(사실), 진행률은 좌표 1개 투영(가벼움). 추월 확률은 별개(실시간).
    if cutoff:
        ref = await track_ref.get_reference(session_key)
        if ref is not None:
            loc_start = _iso_minus(cutoff, 2.0)
            locs = [p for p in await openf1.get_location_window(session_key, driver_number, loc_start, cutoff)
                    if p.get("x") is not None and p.get("y") is not None and p.get("date")]
            if locs:
                locs.sort(key=lambda p: p["date"])
                now = locs[-1]
                pr = track_ref.project(ref, now["x"], now["y"])
                feats["track_progress"] = pr
                feats["track_progress_sin"] = math.sin(2 * math.pi * pr)
                feats["track_progress_cos"] = math.cos(2 * math.pi * pr)
                feats["segment"] = float(min(int(pr * 30), 29))

    return feats
