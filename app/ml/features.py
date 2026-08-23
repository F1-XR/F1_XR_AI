"""단일시점 피처 빌더 — (session_key, at_time, driver) → 추월 예측 모델 입력 피처.

원칙:
  - **at_time 이하 데이터만** 사용한다(스포일러 방지). 미래 값은 절대 넣지 않는다.
  - 계산 가능한 피처만 dict로 채우고, 나머지는 predict가 계약의 결측값(-1.0)으로 메운다.
    (모델이 fillna(-1.0)로 학습돼 결측 피처를 그대로 처리한다 → 단계적 확장 가능.)

커버 현황(car_data + location + 앞차 상대피처 배선 반영, 최대 26/26):
  계산: season, gap_ahead, gap_trend, position, is_lap1, tyre_age, sector,
        air_temperature, track_temperature, humidity, rainfall,
        speed, drs_active, speed_delta,                         (← 앞차와 같은 시각 car_data)
        track_progress, track_progress_sin/cos, segment          (← location 기준선 투영)
        tyre_age_delta, position_delta, same_lap, drs_range, weather_regime_code,
        circuit_key, circuit_type_code, restart_phase
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


def _current_lap(laps: list[dict], cutoff: str | None) -> tuple[int | None, dict | None]:
    valid = [
        r for r in _before(laps, cutoff, key="date_start")
        if r.get("lap_number") is not None
    ]
    record = max(valid, key=lambda r: r["lap_number"]) if valid else None
    return (int(record["lap_number"]), record) if record else (None, None)


def _tyre_age(stints: list[dict], current_lap: int | None) -> float | None:
    if current_lap is None:
        return None
    current = None
    for stint in stints:
        start, end = stint.get("lap_start"), stint.get("lap_end")
        if start is not None and start <= current_lap and (end is None or current_lap <= end):
            current = stint
    if current is None:
        started = [s for s in stints if s.get("lap_start") is not None and s["lap_start"] <= current_lap]
        current = max(started, key=lambda s: s["lap_start"]) if started else None
    if current is None:
        return None
    return float(
        current_lap - int(current["lap_start"]) + int(current.get("tyre_age_at_start") or 0)
    )


async def build_features(session_key: int, at_time: str | None, driver_number: int) -> dict:
    """계산 가능한 피처만 담은 dict를 반환한다(키=피처명). 나머지는 predict가 -1.0으로 채운다."""
    cutoff = at_time
    feats: dict[str, float] = {}

    # season — at_time 연도
    if cutoff and len(cutoff) >= 4 and cutoff[:4].isdigit():
        feats["season"] = float(cutoff[:4])

    # intervals → current gap plus short temporal approach pattern
    iv = _before(await openf1.get_intervals(session_key, driver_number), cutoff)
    iv.sort(key=lambda r: r["date"])
    if iv:
        g_now = _num(iv[-1].get("interval"))   # 앞차와의 간격(초)
        if g_now is not None:
            feats["gap_ahead"] = g_now
            # 학습 파이프라인의 1초 grid diff(5)와 동일하게 약 5초 변화량을 사용한다.
            t_now = iv[-1]["date"]
            for r in reversed(iv[:-1]):
                if _seconds_between(r["date"], t_now) >= 5.0:
                    g_prev = _num(r.get("interval"))
                    if g_prev is not None:
                        feats["gap_trend"] = g_now - g_prev
                    break
            t_now_dt = _dt.datetime.fromisoformat(t_now.replace("Z", "+00:00"))
            recent_gaps: list[tuple[float, float]] = []
            for row in iv:
                gap = _num(row.get("interval"))
                if gap is None:
                    continue
                row_dt = _dt.datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
                age = (t_now_dt - row_dt).total_seconds()
                if 0 <= age <= 10.5:
                    recent_gaps.append((age, gap))
            for seconds in (3, 5, 10):
                older = [item for item in recent_gaps if item[0] >= seconds]
                if older:
                    _, previous_gap = min(older, key=lambda item: item[0])
                    feats[f"gap_delta_{seconds}s"] = g_now - previous_gap
            gaps_5 = [gap for age, gap in recent_gaps if age <= 5.0]
            gaps_10 = [gap for _, gap in recent_gaps]
            if gaps_5:
                feats["gap_min_5s"] = min(gaps_5)
            if len(gaps_10) >= 3:
                mean_gap = sum(gaps_10) / len(gaps_10)
                feats["gap_std_10s"] = math.sqrt(
                    sum((gap - mean_gap) ** 2 for gap in gaps_10) / (len(gaps_10) - 1)
                )
                chronological = list(reversed(gaps_10))
                feats["closing_ratio_10s"] = sum(
                    now_gap < prev_gap
                    for prev_gap, now_gap in zip(chronological, chronological[1:])
                ) / max(1, len(chronological) - 1)

    # position → position
    pos = [
        r for r in _before(await openf1.get_positions(session_key), cutoff)
        if r.get("driver_number") == driver_number and r.get("position") is not None
    ]
    pos.sort(key=lambda r: r["date"])
    ahead_driver = None
    if pos:
        feats["position"] = float(pos[-1]["position"])

    # 같은 시각 각 차량의 최신 순위로 바로 앞차를 찾는다.
    all_positions = _before(await openf1.get_positions(session_key), cutoff)
    latest_by_driver: dict[int, dict] = {}
    for row in all_positions:
        dn, date, position = row.get("driver_number"), row.get("date"), row.get("position")
        if dn is None or not date or position is None:
            continue
        dn = int(dn)
        if dn not in latest_by_driver or date > latest_by_driver[dn]["date"]:
            latest_by_driver[dn] = row
    subject_position = latest_by_driver.get(driver_number, {}).get("position")
    if subject_position is not None:
        target_position = int(subject_position) - 1
        for dn, row in latest_by_driver.items():
            if row.get("position") == target_position:
                ahead_driver = dn
                feats["position_delta"] = float(int(subject_position) - target_position)
                break

    # 세션 메타데이터: 학습과 같은 circuit_key/type 인코딩을 사용한다.
    metadata = await openf1.get_session_metadata(session_key)
    if metadata.get("circuit_key") is not None:
        feats["circuit_key"] = float(metadata["circuit_key"])
    circuit_type = str(metadata.get("circuit_type", "")).strip().lower()
    circuit_type_codes = {
        "permanent": 0.0,
        "temporary - street": 1.0,
        "temporary - road": 2.0,
    }
    if circuit_type in circuit_type_codes:
        feats["circuit_type_code"] = circuit_type_codes[circuit_type]

    # 학습 데이터는 재시작 구간을 품질 필터에서 제거하므로, 런타임 유효 행은 항상 0이다.
    feats["restart_phase"] = 0.0

    # laps → current_lap, is_lap1, sector
    subject_laps = await openf1.get_laps(session_key, driver_number)
    current_lap, cur_lap_rec = _current_lap(subject_laps, cutoff)
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
        subject_stints = await openf1.get_stints(session_key, driver_number)
        age = _tyre_age(subject_stints, current_lap)
        if age is not None:
            feats["tyre_age"] = age

        if ahead_driver is not None:
            ahead_lap, _ = _current_lap(await openf1.get_laps(session_key, ahead_driver), cutoff)
            if ahead_lap is not None:
                feats["same_lap"] = 1.0 if ahead_lap == current_lap else 0.0
                ahead_age = _tyre_age(await openf1.get_stints(session_key, ahead_driver), ahead_lap)
                if age is not None and ahead_age is not None:
                    feats["tyre_age_delta"] = age - ahead_age

    # weather → air_temperature, track_temperature, humidity, rainfall (최신 ≤ cutoff)
    weather = _before(await openf1.get_weather(session_key), cutoff)
    weather.sort(key=lambda r: r["date"])
    if weather:
        w = weather[-1]
        for field in ("air_temperature", "track_temperature", "humidity", "rainfall"):
            v = _num(w.get(field))
            if v is not None:
                feats[field] = v
        rain_now = (_num(w.get("rainfall")) or 0.0) > 0
        transition = False
        for earlier in reversed(weather[:-1]):
            if _seconds_between(earlier["date"], w["date"]) > 300.0:
                break
            rain_before = (_num(earlier.get("rainfall")) or 0.0) > 0
            if rain_before != rain_now:
                transition = True
                break
        feats["weather_regime_code"] = 1.0 if transition else (2.0 if rain_now else 0.0)

    if "gap_ahead" in feats:
        feats["drs_range"] = 1.0 if feats["gap_ahead"] < 1.0 else 0.0

    # car_data(시점 근처 창) → current and 5-10 second relative-speed pattern.
    start_iso = _iso_minus(cutoff, 11.0) if cutoff else None
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
                # 학습과 동일: speed_delta = 내 속도 - 바로 앞차 속도.
                if ahead_driver is not None:
                    ahead_car = await openf1.get_car_data_window(
                        session_key, ahead_driver, start_iso, cutoff
                    )
                    ahead_car = [c for c in ahead_car if c.get("date") and _num(c.get("speed")) is not None]
                    if ahead_car:
                        closest = min(ahead_car, key=lambda c: _seconds_between(c["date"], now["date"]))
                        feats["speed_delta"] = sp - float(closest["speed"])
                        t_now_dt = _dt.datetime.fromisoformat(now["date"].replace("Z", "+00:00"))
                        deltas: list[tuple[float, float]] = []
                        drs_samples: list[tuple[float, float]] = []
                        for subject in car:
                            subject_speed = _num(subject.get("speed"))
                            if subject_speed is None:
                                continue
                            target = min(ahead_car, key=lambda c: _seconds_between(c["date"], subject["date"]))
                            age = (t_now_dt - _dt.datetime.fromisoformat(subject["date"].replace("Z", "+00:00"))).total_seconds()
                            if 0 <= age <= 10.5:
                                deltas.append((age, subject_speed - float(target["speed"])))
                                drs_samples.append((age, 1.0 if subject.get("drs") in (10, 12, 14) else 0.0))
                        for seconds in (5, 10):
                            values = [delta for age, delta in deltas if age <= seconds]
                            if values:
                                feats[f"speed_delta_mean_{seconds}s"] = sum(values) / len(values)
                        values_10 = [delta for _, delta in deltas]
                        if values_10:
                            feats["speed_advantage_ratio_10s"] = sum(v > 0 for v in values_10) / len(values_10)
                        drs_10 = [value for _, value in drs_samples]
                        if drs_10:
                            feats["drs_active_ratio_10s"] = sum(drs_10) / len(drs_10)

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
