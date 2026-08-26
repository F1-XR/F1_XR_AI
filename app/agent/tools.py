"""에이전트 도구 8종.

성격에 따라 두 종류:
  - 조회형 (getDriverInfo/getRaceStatus/explainConcept/explainWhy)
      → 데이터를 읽어 dict를 반환. LLM이 이걸 근거로 한국어 답변을 만든다.
  - 명령형 (highlightDriver/controlReplay/jumpToEvent) + 능동형(pointOutMoment)
      → emit_command()로 Unity 명령을 쌓고, LLM에겐 짧은 확인만 반환.

도구 docstring이 곧 LLM에게 주는 사용 설명서이므로 명확하게 적는다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.tools import tool

from ..data import openf1
from .commands import emit_command
from .context import (current_recent_overtake, current_selected, current_session, current_time,
                      get_driver_prob, get_driver_prob_snapshot, set_session)

logger = logging.getLogger(__name__)

_GLOSSARY = json.loads(
    (Path(__file__).parent.parent / "data" / "glossary.json").read_text(encoding="utf-8")
)


# ───────────────────────────── 세션 전환 ─────────────────────────────

@tool
async def find_session(year: int, race: str, session_name: str = "Race") -> dict:
    """사용자가 특정 경기를 말로 지정하면 그 경기로 전환한다.
    예: "2024 모나코 경기 보여줘" → find_session(2024, "Monaco").
    race 는 영문 국가명 또는 서킷명(Monaco, Italy, Japan, Silverstone 등).
    session_name 은 'Race' | 'Qualifying' | 'Sprint' 등(기본 Race).
    전환 후에는 이후 모든 질문이 이 경기 기준으로 답해진다."""
    # 국가명으로 먼저 시도, 없으면 서킷명으로
    sessions = await openf1.find_sessions(year=year, country=race, session_name=session_name)
    if not sessions:
        sessions = await openf1.find_sessions(year=year, circuit=race, session_name=session_name)
    if not sessions:
        return {"found": False,
                "note": f"{year} {race} {session_name} 경기를 찾지 못했어요. 영문 지명으로 다시 시도해 보세요."}
    s = sessions[0]
    set_session(s["session_key"])
    # 말로 경기를 바꿨으면 Unity 화면도 그 경기로 전환하도록 명령한다
    # (에이전트 context와 Unity 화면을 항상 일치시킨다).
    emit_command("loadSession", session_key=s["session_key"])
    return {
        "found": True,
        "session_key": s.get("session_key"),
        "race": s.get("country_name"),
        "circuit": s.get("circuit_short_name"),
        "year": s.get("year"),
        "session": s.get("session_name"),
    }


# ────────────────────────────── 조회형 ──────────────────────────────

@tool
async def get_driver_info(driver_number: int) -> dict:
    """드라이버 번호로 선수 정보를 조회한다. 사용자가 "이 선수 누구야?"처럼
    특정 선수의 이름·팀·국적·통산 기록을 물을 때 사용한다."""
    session = current_session()
    driver = await openf1.get_driver(session, driver_number)
    if not driver:
        return {"error": f"{driver_number}번 선수를 찾을 수 없어요."}

    info = {
        "number": driver.get("driver_number"),
        "name": driver.get("full_name"),
        "acronym": driver.get("name_acronym"),
        "team": driver.get("team_name"),
        "country": driver.get("country_code"),
        "headshot_url": driver.get("headshot_url"),
    }
    # 커리어(통산 기록)는 데이터 서버가 Jolpica를 캐시해 제공 — 실패해도 기본 정보는 반환.
    try:
        career = await openf1.get_career(session, driver_number)
        if career:
            if career.get("dateOfBirth"):
                info["date_of_birth"] = career["dateOfBirth"]
            if career.get("nationality"):
                info["nationality"] = career["nationality"]
            if career.get("wins"):
                info["career_wins"] = career["wins"]
    except Exception:
        pass
    return info


@tool
async def get_race_status() -> dict:
    """현재 리플레이 시점의 경기 상황(깃발/세이프티카/순위)을 조회한다.
    "지금 무슨 상황이야?", "누가 1등이야?" 같은 질문에 사용한다.
    리플레이 현재 시각(at_time) 이하의 데이터만 보므로 아직 안 지난 결과는 스포일러하지 않는다."""
    session = current_session()
    cutoff = current_time()   # 리플레이 현재 시각(ISO). None이면 전체(=최신).
    rc = await openf1.get_race_control(session)
    positions = await openf1.get_positions(session)

    def _before(rows: list[dict]) -> list[dict]:
        """cutoff 이하 시점의 기록만 남긴다(cutoff 없으면 전체)."""
        if not cutoff:
            return rows
        return [r for r in rows if not r.get("date") or r["date"] <= cutoff]

    # 가장 최근 깃발/세이프티카 (cutoff 이하 이벤트 중 마지막)
    latest_flag = None
    safety_car = False
    for ev in _before(rc):
        if ev.get("flag"):
            latest_flag = {"flag": ev.get("flag"), "message": ev.get("message")}
        cat, msg = ev.get("category"), (ev.get("message") or "").upper()
        if cat == "SafetyCar":
            # "DEPLOYED/IN THIS LAP" → 투입, "IN THIS LAP"에 ENDING 류면 해제
            safety_car = "ENDING" not in msg and "IN THIS LAP" not in msg

    # 현재 순위: 드라이버별 cutoff 이하 '가장 최근' position 기록으로 산출
    latest: dict[int, tuple] = {}   # driver_number -> (date, position)
    for p in _before(positions):
        dn, pos, d = p.get("driver_number"), p.get("position"), p.get("date")
        if dn is None or pos is None:
            continue
        prev = latest.get(dn)
        if prev is None or (d or "") >= (prev[0] or ""):
            latest[dn] = (d, pos)
    # 번호→약칭 매핑을 붙여 순위를 사람이 알아보게 한다.
    # (도구가 번호만 주면 LLM이 '1등'(순위)과 '1번'(차량번호)을 혼동할 수 있음)
    try:
        name_of = {
            d.get("driver_number"): (d.get("full_name") or d.get("name_acronym"))
            for d in await openf1.get_drivers(session)
        }
    except Exception:
        name_of = {}

    standings = [
        {"position": pos, "driver_number": dn, "driver": name_of.get(dn)}
        for dn, (d, pos) in sorted(latest.items(), key=lambda kv: kv[1][1])
    ]

    return {
        "latest_flag": latest_flag,
        "safety_car": safety_car,
        "standings": standings[:5],   # [{position, driver_number, driver(전체 이름)}]
        "at_time": cutoff,
    }


@tool
def explain_concept(term: str) -> dict:
    """F1 용어를 입문자 눈높이로 설명한다. "DRS가 뭐야?", "언더컷이 뭐야?"처럼
    규칙·용어를 물을 때 사용한다. 용어집(로컬)에서 조회한다."""
    # 부분 일치 허용(예: "drs 존" → "DRS")
    key = term.strip().upper()
    for k, v in _GLOSSARY.items():
        if k.upper() == key or k.upper() in key or key in k.upper():
            return {"term": k, "explanation": v}
    return {"term": term, "explanation": None,
            "note": "용어집에 없어요. LLM 일반 지식으로 간단히 설명하되 확실하지 않으면 모른다고 하세요."}


@tool
async def explain_why(driver_number: int) -> dict:
    """특정 선수의 행동 이유를 데이터 근거로 설명한다. "쟤 왜 피트인했어?",
    "왜 느려졌어?"처럼 '왜?'를 물을 때 사용한다. 타이어/피트/갭을 종합한다.
    리플레이 현재 시각(at_time) 이하의 데이터만 보므로 아직 안 지난 결과는 스포일러하지 않는다."""
    session = current_session()
    cutoff = current_time()   # 리플레이 현재 시각(ISO). None이면 전체(=최신).
    # 실제 선수 이름을 데이터에서 가져온다 → 모델이 이름을 지어내지(할루시네이션) 못하게.
    driver = await openf1.get_driver(session, driver_number)
    pit = await openf1.get_pit(session, driver_number)
    stints = await openf1.get_stints(session, driver_number)
    intervals = await openf1.get_intervals(session, driver_number)

    # 스포일러 방지: cutoff 이후(아직 안 지난 미래) 데이터는 감춘다. get_race_status와 동일 규칙.
    if cutoff:
        def _before(rows: list[dict]) -> list[dict]:
            """date 필드가 있는 기록만 cutoff 이하로 컷(없으면 그대로 통과)."""
            return [r for r in rows if not r.get("date") or r["date"] <= cutoff]

        pit = _before(pit)
        intervals = _before(intervals)
        # 스틴트는 date가 없고 랩 기반이다. '피트 1회 = 새 스틴트 1개'이므로,
        # 지금까지 완료된 피트 수(len(pit)) + 1개 스틴트까지만 남겨 미래 타이어 교체를 감춘다.
        stints = sorted(stints, key=lambda s: s.get("stint_number") or 0)[: len(pit) + 1]

    return {
        "driver_number": driver_number,
        "driver_name": (driver or {}).get("full_name"),
        "team": (driver or {}).get("team_name"),
        "pit_stops": pit[-3:] if pit else [],
        "tire_stints": stints[-3:] if stints else [],
        "recent_gap": intervals[-1] if intervals else None,
        "at_time": cutoff,
        "hint": "타이어 스틴트가 길면 노후화로 피트인, 갭이 좁혀지면 추월 압박일 수 있음.",
        "note": "driver_name 을 실제 선수 이름으로 그대로 쓰세요. 이름을 추측·변경하지 마세요.",
    }


# ────────────────────────────── 명령형 ──────────────────────────────

@tool
def highlight_driver(driver_number: int) -> str:
    """지정한 드라이버를 화면에서 강조/마커 표시하도록 Unity에 명령한다.
    "해밀턴 강조해줘", "그 선수 어디 있는지 보여줘"에 사용한다."""
    emit_command("highlightDriver", driver_number=driver_number)
    return f"{driver_number}번 선수를 화면에서 강조했어요."


@tool
def control_replay(action: str, value: float | None = None) -> str:
    """리플레이 재생을 제어하도록 Unity에 명령한다.
    action은 'play' | 'pause' | 'speed' | 'seek' 중 하나.
    "천천히"→speed(0.5), "다시 재생"→play, "멈춰"→pause, "여기로"→seek(value=시각).
    복합 명령("그 장면 다시 천천히")은 여러 도구를 순서대로 호출해 처리한다."""
    emit_command("controlReplay", action=action, value=value)
    label = {"play": "재생", "pause": "정지", "speed": "속도 조절", "seek": "이동"}.get(action, action)
    return f"리플레이를 {label}했어요."


@tool
async def jump_to_event(event_type: str) -> str:
    """의미 있는 장면으로 리플레이를 점프시킨다. event_type은
    'first_pit' | 'first_overtake' | 'first_overtake_before' | 'safety_car' | 'yellow_flag' 등.
    이벤트의 시각을 데이터에서 찾아 그 지점으로 Seek 명령을 보낸다.
    "첫 피트스톱 보여줘" → first_pit.
    "첫 추월 장면으로 가줘" → first_overtake.
    "그 추월 장면 직전으로 돌아가" → first_overtake_before.
    "사고 장면으로" → safety_car 또는 yellow_flag."""
    session = current_session()
    cutoff = current_time()
    target_time = None

    def _visible(rows: list[dict]) -> list[dict]:
        rows = [r for r in rows if r.get("date")]
        return [r for r in rows if not cutoff or r["date"] <= cutoff]

    if event_type == "first_pit":
        pit = _visible(await openf1.get_pit(session))
        if pit:
            target_time = min(pit, key=lambda p: p["date"])["date"]
    elif event_type in ("first_overtake", "first_overtake_before"):
        recent = current_recent_overtake()
        if event_type == "first_overtake_before" and recent and recent.get("event_time"):
            target_time = recent["event_time"]
        else:
            positions = _visible([
                p for p in await openf1.get_positions(session)
                if p.get("date") and p.get("driver_number") is not None and p.get("position") is not None
            ])
            positions.sort(key=lambda p: p["date"])
            pits = _visible(await openf1.get_pit(session))
            prev_pos: dict[int, tuple[int, str]] = {}
            for p in positions:
                dn, pos = int(p["driver_number"]), int(p["position"])
                before = prev_pos.get(dn)
                if before is not None and pos < before[0]:
                    # A real on-track pass should have a counterpart dropping into
                    # the subject's old place at nearly the same timestamp.  This
                    # rejects most pit-cycle/retirement position gains.
                    counterpart = next((
                        other for other, state in prev_pos.items()
                        if other != dn and state[0] == pos
                        and any(q.get("driver_number") == other
                                and int(q.get("position")) == before[0]
                                and abs((datetime.fromisoformat(q["date"].replace("Z", "+00:00")) -
                                         datetime.fromisoformat(p["date"].replace("Z", "+00:00"))).total_seconds()) <= 3
                                for q in positions)
                    ), None)
                    event_dt = datetime.fromisoformat(p["date"].replace("Z", "+00:00"))
                    pit_nearby = any(
                        pit.get("driver_number") in (dn, counterpart)
                        and abs((datetime.fromisoformat(pit["date"].replace("Z", "+00:00")) - event_dt).total_seconds()) <= 30
                        for pit in pits
                    )
                    if counterpart is not None and not pit_nearby:
                        target_time = p["date"]
                        break
                prev_pos[dn] = (pos, p["date"])
    else:
        rc = _visible(await openf1.get_race_control(session))
        want = {"safety_car": "SafetyCar", "yellow_flag": "Flag"}.get(event_type)
        for ev in rc:
            if want and ev.get("category") == want and ev.get("date"):
                target_time = ev["date"]
                break

    if not target_time:
        return f"'{event_type}' 장면을 데이터에서 찾지 못했어요."

    if event_type.endswith("_before"):
        target_time = seek_before(target_time, 5.0)

    # 절대 시각(ISO) → Unity가 리플레이 상대시간으로 매핑해 Seek
    emit_command("controlReplay", action="seek", value=target_time)
    return f"'{event_type}' 장면({target_time})으로 이동했어요."


def seek_before(iso: str, seconds: float = 5.0) -> str:
    """ISO 시각보다 seconds초 앞선 시각. 실패하면 원본을 돌려준다."""
    try:
        t = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return (t - timedelta(seconds=seconds)).isoformat()
    except (ValueError, AttributeError):
        return iso


# ────────────────────────────── 예측형 ──────────────────────────────

@tool
async def predict_overtake(driver_number: int) -> dict:
    """지정 드라이버가 '앞으로 30초 안에' 추월/순위 변동할 **확률**을 예측한다.
    "쟤 추월할 것 같아?", "곧 순위 바뀔까?", "추월 가능성 있어?"처럼 미래 가능성을 물을 때 사용.
    학습된 예측 모델(LightGBM) 기반이며, 이미 일어난 사실이 아니라 '예측(추정치)'이다.
    현재 리플레이 시각(at_time) 이하 데이터만 보므로 스포일러하지 않는다."""
    session = current_session()
    cutoff = current_time()
    # get_drivers는 세션 캐시를 공유하므로 매 음성 질문마다 선수 단건 HTTP를 추가하지 않는다.
    drivers = await openf1.get_drivers(session)
    driver = next((row for row in drivers
                   if row.get("driver_number") is not None
                   and int(row["driver_number"]) == int(driver_number)), None)
    cached = get_driver_prob_snapshot(session, driver_number, max_age_sec=20.0)
    if cached and cutoff and cached.get("at_time"):
        try:
            now_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
            cached_dt = datetime.fromisoformat(cached["at_time"].replace("Z", "+00:00"))
            replay_age = abs((now_dt - cached_dt).total_seconds())
        except (ValueError, TypeError):
            replay_age = 999.0
        if replay_age <= 3.0:
            return {
                "driver_number": driver_number,
                "driver_name": (driver or {}).get("full_name"),
                "team": (driver or {}).get("team_name"),
                "overtake_probability": cached["probability"],
                "position_gain_probability": None,
                "position_loss_probability": None,
                "at_time": cached["at_time"],
                "inputs": {},
                "source": "watcher_cache",
                "note": "현재 화면과 3초 이내인 watcher ML 계산값을 재사용했습니다.",
            }
    try:
        from ..ml import features as _feat
        from ..ml import predict as _pred
        feats = await _feat.build_features(session, cutoff, driver_number)
        probs = _pred.predict(feats)
    except Exception:
        logger.exception("추월 예측 실패")
        return {"available": False, "note": "예측 모델을 부를 수 없어요(모델·의존성 확인 필요)."}

    return {
        "driver_number": driver_number,
        "driver_name": (driver or {}).get("full_name"),
        "team": (driver or {}).get("team_name"),
        "overtake_probability": probs.get("overtake_probability"),
        "position_gain_probability": probs.get("position_gain_probability"),
        "position_loss_probability": probs.get("position_loss_probability"),
        "at_time": cutoff,
        "inputs": {
            "gap_ahead": feats.get("gap_ahead"),
            "gap_trend": feats.get("gap_trend"),
            "speed_delta": feats.get("speed_delta"),
            "drs_active": feats.get("drs_active"),
            "feature_count": len(feats),
        },
        "note": "30초 내 확률(예측치). driver_name 을 그대로 쓰고 이름을 지어내지 마세요.",
    }


@tool
async def show_battle_context(driver_number: int) -> dict:
    """지정 드라이버와 '바로 앞차' 사이의 배틀 상황(간격·추세·DRS)을 계산하고,
    두 차 사이에 Gap Line + 배지("0.8s · Closing · DRS")를 화면에 표시하도록 Unity에 명령한다.
    "지금 앞차랑 배틀 상황 보여줘", "얼마나 붙었어?", "왜 추월 압박이야?"처럼
    두 차의 근접 상황을 공간적으로 보여줄 때 사용. 현재 시각(at_time) 이하만 본다(스포일러 방지)."""
    from ..ml.features import _before, _num, _seconds_between, _iso_minus

    session = current_session()
    cutoff = current_time()

    # 1) 최신 순위 → subject 순위와 '바로 앞차'(target) 찾기 (get_race_status와 동일 방식)
    latest: dict[int, tuple] = {}   # driver_number -> (date, position)
    for p in _before(await openf1.get_positions(session), cutoff):
        dn, pos, d = p.get("driver_number"), p.get("position"), p.get("date")
        if dn is None or pos is None:
            continue
        prev = latest.get(dn)
        if prev is None or (d or "") >= (prev[0] or ""):
            latest[dn] = (d, pos)
    subj = latest.get(driver_number)
    if subj is None:
        return {"shown": False, "note": f"{driver_number}번의 위치 정보를 찾지 못했어요."}
    subj_pos = subj[1]
    if subj_pos <= 1:
        return {"shown": False, "note": "지금 선두라 앞에 배틀 상대가 없어요."}
    target = next((dn for dn, (d, pos) in latest.items() if pos == subj_pos - 1), None)
    if target is None:
        return {"shown": False, "note": "앞차를 특정하지 못했어요."}

    # 2) 간격(초)과 추세 — subject의 intervals (음수 변화 = 좁혀짐 = closing)
    #    API가 시간순을 보장하지 않으므로 date로 정렬해 iv[-1]이 '진짜 최신'이 되게 한다.
    iv = _before(await openf1.get_intervals(session, driver_number), cutoff)
    iv = sorted(iv, key=lambda r: r.get("date") or "")
    gap = _num(iv[-1].get("interval")) if iv else None
    trend = "stable"
    if iv and gap is not None and iv[-1].get("date"):
        for r in reversed(iv[:-1]):
            if r.get("date") and _seconds_between(r["date"], iv[-1]["date"]) >= 3:
                gp = _num(r.get("interval"))
                if gp is not None:
                    if gap < gp - 0.05:
                        trend = "closing"
                    elif gap > gp + 0.05:
                        trend = "opening"
                break

    def _nearest_relative_speed(
        when_iso: str,
        subject_car: list[dict],
        target_car: list[dict],
        max_dt: float = 1.5,
    ) -> float | None:
        """subject speed - target speed near when_iso.

        This is the second sensor for Feature 1-B. It must be relative speed
        between the two battle cars, not the existing ML feature's own-car
        speed-vs-past-speed delta.
        """
        best_subject = None
        best_subject_dt = None
        for row in subject_car:
            d = row.get("date")
            sp = _num(row.get("speed"))
            if not d or sp is None:
                continue
            dt = abs(_seconds_between(d, when_iso))
            if dt <= max_dt and (best_subject_dt is None or dt < best_subject_dt):
                best_subject = (d, sp)
                best_subject_dt = dt
        if best_subject is None:
            return None

        best_target_sp = None
        best_target_dt = None
        for row in target_car:
            d = row.get("date")
            sp = _num(row.get("speed"))
            if not d or sp is None:
                continue
            dt = abs(_seconds_between(d, best_subject[0]))
            if dt <= max_dt and (best_target_dt is None or dt < best_target_dt):
                best_target_sp = sp
                best_target_dt = dt
        if best_target_sp is None:
            return None
        return float(best_subject[1] - best_target_sp)

    # 2.5) 3초 뒤 갭 예측 — 칼만 상태추정(등속 모델 + 불확실성).
    #   Feature 1-A: intervals gap만으로 상태[gap, gap_rate] 추정.
    #   Feature 1-B: car_data의 subject-target 상대속도를 약한 두 번째 센서로 융합.
    horizon = 3.0
    predicted_gap = None
    predicted_gap_std = None
    fusion_used = False
    relative_speed_kmh = None
    subject_car_data: list[dict] = []
    target_car_data: list[dict] = []
    car_data_end = cutoff or (iv[-1].get("date") if iv else None)
    if car_data_end:
        start = _iso_minus(car_data_end, 8)
        if start:
            subject_car_data = [
                c for c in await openf1.get_car_data_window(session, driver_number, start, car_data_end)
                if c.get("date") and c["date"] <= car_data_end
            ]
            target_car_data = [
                c for c in await openf1.get_car_data_window(session, target, start, car_data_end)
                if c.get("date") and c["date"] <= car_data_end
            ]

    if iv and gap is not None and iv[-1].get("date"):
        from ..ml.state_estimator import GapEstimator
        t_end = iv[-1]["date"]
        est = GapEstimator(fuse_speed_delta=bool(subject_car_data and target_car_data))
        used = 0
        for r in iv:
            d, g = r.get("date"), _num(r.get("interval"))
            if d and g is not None:
                t_rel = -_seconds_between(d, t_end)   # 과거 음수, 최신 0 (단조 증가)
                est.observe(t_rel, g)
                speed_delta = _nearest_relative_speed(d, subject_car_data, target_car_data)
                if speed_delta is not None:
                    est.observe_speed_delta(t_rel, speed_delta)
                    fusion_used = True
                    relative_speed_kmh = speed_delta
                used += 1
        if used >= 2 and est.ready:
            mean, std = est.predict(horizon)
            predicted_gap = round(mean, 2)            # 3초 뒤 예측 갭(칼만 평균)
            predicted_gap_std = round(std, 2)         # 예측 불확실성(±초)

    # 3) DRS — subject의 car_data 창(현재 시각 근처). drs 코드 10/12/14 = 작동 중.
    drs = False
    if subject_car_data:
        drs = subject_car_data[-1].get("drs") in (10, 12, 14)

    # 4) confidence(간단 휴리스틱): 가깝고 좁혀질수록 높게
    conf = 0.0
    if gap is not None:
        conf = max(0.0, min(1.0, (1.2 - gap) * (0.7 if trend == "closing" else 0.4)))

    emit_command(
        "showBattleContext",
        subject_driver=driver_number,
        target_driver=target,
        gap_seconds=round(gap, 2) if gap is not None else None,
        gap_available=gap is not None,
        predicted_gap_seconds=predicted_gap,      # 3초 뒤 예측 갭(없으면 None → Unity 화살표 생략)
        predicted_gap_std_seconds=predicted_gap_std,   # 예측 불확실성(±초). Unity가 밴드/투명도로 사용 가능
        predict_horizon_sec=horizon,
        fusion_used=fusion_used,
        relative_speed_kmh=round(relative_speed_kmh, 1) if relative_speed_kmh is not None else None,
        trend=trend,
        drs=bool(drs),
        confidence=round(conf, 2),
        reason=("앞차와의 간격·추세·DRS·상대속도 기반(센서 융합 칼만 상태추정)"
                if fusion_used else "앞차와의 간격·추세·DRS 기반(칼만 상태추정)"),
    )
    return {
        "shown": True,
        "subject_driver": driver_number,
        "target_driver": target,
        "gap_seconds": round(gap, 2) if gap is not None else None,
        "gap_available": gap is not None,
        "trend": trend,
        "drs": bool(drs),
        "predicted_gap_seconds": predicted_gap,
        "predicted_gap_std_seconds": predicted_gap_std,
        "predict_horizon_sec": horizon,
        "fusion_used": fusion_used,
        "relative_speed_kmh": round(relative_speed_kmh, 1) if relative_speed_kmh is not None else None,
        "note": ("두 차 사이에 Gap Line·배지·예측 화살표를 표시했어요. 현재 갭과 함께 "
                 "predicted_gap_seconds가 있으면 '지금 X초, 3초 뒤 Y초로 좁혀질(벌어질) 것 같아요'처럼 "
                 "예측을 한 문장 덧붙이세요(사실 X / 예측 Y 구분). fusion_used가 True면 "
                 "'차량 속도 데이터까지 함께 반영했다'고 덧붙일 수 있습니다. 없으면 예측은 언급하지 마세요."),
    }


@tool
async def recommend_battle_action(driver_number: int) -> dict:
    """Battle Decision Lite.

    지정 드라이버가 앞차를 상대로 지금 공격해야 할지, DRS를 기다려야 할지,
    압박만 유지해야 할지 결정론적 정책으로 추천한다. LLM이 결정을 지어내지 않도록
    상태추정(Kalman gap prediction), 불확실성, DRS, gap trend, 보정된 추월확률을
    근거로 action label과 이유를 반환한다.
    """
    battle = await show_battle_context.ainvoke({"driver_number": driver_number})
    if not isinstance(battle, dict) or not battle.get("shown"):
        return {
            "available": False,
            "driver_number": driver_number,
            "action": "NO_TARGET",
            "confidence": 0.0,
            "reason": (battle or {}).get("note", "앞차 배틀 상대를 찾지 못했어요."),
        }

    overtake_probability = None
    try:
        pred = await predict_overtake.ainvoke({"driver_number": driver_number})
        if isinstance(pred, dict) and pred.get("available", True) is not False:
            overtake_probability = pred.get("overtake_probability")
    except Exception:
        logger.exception("battle action probability lookup failed")

    from ..ml.decision_policy import recommend_battle_policy

    policy = recommend_battle_policy(
        gap_seconds=battle.get("gap_seconds"),
        predicted_gap_seconds=battle.get("predicted_gap_seconds"),
        predicted_gap_std_seconds=battle.get("predicted_gap_std_seconds"),
        trend=battle.get("trend"),
        drs=bool(battle.get("drs")),
        fusion_used=bool(battle.get("fusion_used")),
        relative_speed_kmh=battle.get("relative_speed_kmh"),
        overtake_probability=overtake_probability,
    )

    return {
        "available": True,
        "driver_number": driver_number,
        "target_driver": battle.get("target_driver"),
        **policy,
        "note": ("이 결정은 LLM 추측이 아니라 deterministic policy 결과입니다. "
                 "답변할 때 '공격 추천/대기/유지'를 먼저 말하고, 근거는 갭·DRS·예측 불확실성 중심으로 짧게 설명하세요."),
    }


# ────────────────────────────── 상황·전략 해설 ──────────────────────────────

@tool
async def explain_situation(driver_number: int | None = None) -> dict:
    """지금 경기의 '상황 + 전략'을 종합 해설할 근거를 모은다.
    단순 단답이 아니라 상황과 전략을 묶어 설명해야 할 때 사용한다.
    예: "지금 상황 어때?", "무슨 전략이야?", "지금 경기 흐름/전략 설명해줘",
        "왜 추월 압박이야?", "지금 전략 싸움 어떻게 돌아가?".
    driver_number를 주면 그 선수 기준, 없으면 선택 차량 → 없으면 상위권에서 가장
    접전인 선수를 자동으로 고른다. 현재 시각(at_time) 이하 데이터만 본다(스포일러 방지).
    반환한 사실(갭·추세·타이어·DRS·피트·추월확률)을 근거로 사용자 눈높이에 맞춰
    쉽게 먼저 2~4문장으로 설명하고, 사실과 추론을 구분하라."""
    import asyncio
    from ..ml.features import _before, _iso_minus, _num, _seconds_between

    session = current_session()
    cutoff = current_time()

    # 1) 순위 + 드라이버명 동시 조회
    positions, drivers = await asyncio.gather(
        openf1.get_positions(session),
        openf1.get_drivers(session),
    )
    latest: dict[int, tuple] = {}
    for p in _before(positions, cutoff):
        dn, pos, d = p.get("driver_number"), p.get("position"), p.get("date")
        if dn is None or pos is None:
            continue
        prev = latest.get(dn)
        if prev is None or (d or "") >= (prev[0] or ""):
            latest[dn] = (d, pos)
    if not latest:
        return {"available": False, "note": "아직 순위 데이터가 없어요."}
    order = sorted(latest.items(), key=lambda kv: kv[1][1])
    pos_of = {dn: pos for dn, (d, pos) in latest.items()}
    name_of = {d.get("driver_number"): (d.get("full_name") or d.get("name_acronym"))
               for d in (drivers or [])}

    # 2) subject 선택: 인자 > 선택 차량 > 상위권 최소 갭(후보 intervals 병렬)
    subject = driver_number or current_selected()
    if subject is None:
        cand = [dn for dn, (d, pos) in order[:8] if pos > 1]
        ivs = await asyncio.gather(*[openf1.get_intervals(session, dn) for dn in cand],
                                   return_exceptions=True)
        best = None
        for dn, ivr in zip(cand, ivs):
            if isinstance(ivr, Exception):
                continue
            iv = _before(ivr, cutoff)
            g = _num(iv[-1].get("interval")) if iv else None
            if g is not None and (best is None or g < best[1]):
                best = (dn, g)
        subject = best[0] if best else order[0][0]
    subj_pos = pos_of.get(subject)
    target = next((dn for dn, (d, pos) in latest.items()
                   if subj_pos and pos == subj_pos - 1), None)

    # ── 개별 조회 코루틴(병렬 실행용) ──
    async def _tyre(dn):
        if not dn:
            return {"compound": None, "age_laps": None}
        try:
            laps = [r for r in _before(await openf1.get_laps(session, dn), cutoff, key="date_start")
                    if r.get("lap_number") is not None]
            cur_lap = max((int(r["lap_number"]) for r in laps), default=None)
            stints = await openf1.get_stints(session, dn)
            cur = None
            if cur_lap is not None:
                for st in stints:
                    ls, le = st.get("lap_start"), st.get("lap_end")
                    if ls is not None and ls <= cur_lap and (le is None or cur_lap <= le):
                        cur = st
                if cur is None:
                    started = [st for st in stints if st.get("lap_start") is not None and st["lap_start"] <= cur_lap]
                    cur = max(started, key=lambda st: st["lap_start"]) if started else None
            if cur:
                age = None
                if cur_lap is not None and cur.get("lap_start") is not None:
                    age = (cur_lap - int(cur["lap_start"])) + int(cur.get("tyre_age_at_start") or 0)
                return {"compound": cur.get("compound"), "age_laps": age}
        except Exception:
            pass
        return {"compound": None, "age_laps": None}

    async def _gap_trend(dn):
        try:
            iv = _before(await openf1.get_intervals(session, dn), cutoff)
        except Exception:
            return None, None
        gap = trend = None
        if iv:
            gap = _num(iv[-1].get("interval"))
            if gap is not None and iv[-1].get("date"):
                for r in reversed(iv[:-1]):
                    if r.get("date") and _seconds_between(r["date"], iv[-1]["date"]) >= 3:
                        gp = _num(r.get("interval"))
                        if gp is not None:
                            trend = ("closing" if gap < gp - 0.05
                                     else "opening" if gap > gp + 0.05 else "stable")
                        break
        return gap, trend

    async def _drs(dn):
        if not cutoff:
            return None
        start = _iso_minus(cutoff, 4)
        if not start:
            return None
        try:
            cd = [c for c in await openf1.get_car_data_window(session, dn, start, cutoff)
                  if c.get("date") and c["date"] <= cutoff]
            return (cd[-1].get("drs") in (10, 12, 14)) if cd else None
        except Exception:
            return None

    async def _pit(dn):
        try:
            pit = _before(await openf1.get_pit(session, dn), cutoff)
            return f"최근 피트 {len(pit)}회" if pit else None
        except Exception:
            return None

    _PROB_GATE_SEC = 2.5   # 이보다 멀리 있으면(추월 가능성 낮음) 무거운 확률 계산을 생략해 속도↑

    async def _prob(dn, gap):
        # (C) watcher가 백그라운드로 미리 계산한 확률이 있으면 즉시 사용(재계산 안 함).
        cached = get_driver_prob(session, dn)
        if cached is not None:
            return cached, "watcher"
        # 캐시에 없고 '먼 차'면 계산 생략(먼 차의 추월확률은 낮고 계산이 무겁다).
        if gap is None or gap > _PROB_GATE_SEC:
            return None, "skipped_far"
        # 가까운 차만 여기서 계산(다른 조회들과 병렬).
        try:
            from ..ml import features as _feat
            from ..ml import predict as _pred
            feats = await _feat.build_features(session, cutoff, dn)
            return _pred.predict(feats).get("overtake_probability"), "computed"
        except Exception:
            return None, None

    # 3) 게이팅 판단용 gap 먼저(가벼움), 나머지는 병렬 실행
    gap, trend = await _gap_trend(subject)
    drs, subj_tyre, target_tyre, pit_note, (overtake_prob, prob_src) = \
        await asyncio.gather(
            _drs(subject), _tyre(subject),
            _tyre(target), _pit(subject), _prob(subject, gap),
        )

    return {
        "available": True,
        "subject_driver": subject,
        "subject_name": name_of.get(subject),
        "subject_position": subj_pos,
        "ahead_driver": target,
        "ahead_name": name_of.get(target) if target else None,
        "gap_to_ahead_sec": round(gap, 2) if gap is not None else None,
        "gap_trend": trend,
        "drs_active": drs,
        "subject_tyre": subj_tyre,
        "ahead_tyre": target_tyre,
        "recent_pit": pit_note,
        "overtake_probability": overtake_prob,
        "overtake_prob_source": prob_src,
        "standings_top5": [
            {"position": pos, "driver": name_of.get(dn) or f"{dn}번", "driver_number": dn}
            for dn, (d, pos) in order[:5]
        ],
        "at_time": cutoff,
        "note": ("이 사실들을 근거로 '지금 상황 + 전략'을 사용자 눈높이에 맞춰 쉽게 먼저 "
                 "2~4문장으로 설명하세요. 타이어 신선도 차이(언더컷 가능성)와 갭·추세·DRS로 "
                 "추월 압박을 엮되, '사실'과 '추론(예상)'을 구분하고, 데이터에 없는 "
                 "코너링·드라이버 실수는 단정하지 마세요. overtake_probability가 없으면(먼 차) 확률은 "
                 "언급하지 말고 갭·타이어·DRS로만 설명하세요. 사용자가 '더 자세히/왜'라고 하면 "
                 "한 단계 더 깊이 설명하세요."),
    }


async def get_recent_overtake_context(driver_number: int, lookback_sec: float = 20.0) -> dict:
    """방금 끝난 추월을 설명할 검증 가능한 근거를 모은다.

    추월 직후에는 ``explain_situation``이 이미 새 앞차를 대상으로 잡기 때문에 방금
    추월당한 차를 잃는다. 최근 position swap을 먼저 찾아 사건 직전 시각으로 되돌린 뒤
    gap·DRS·두 차량 타이어만 반환한다. LLM 자유 생성이 아닌 데모용 결정적 답변에 쓴다.
    """
    import asyncio
    from datetime import datetime, timedelta, timezone

    session = current_session()
    cutoff = current_time()

    def _dt(value):
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    end = _dt(cutoff)
    if session is None or end is None:
        return {"available": False, "note": "현재 경기 시각을 확인할 수 없어요."}

    positions, drivers, pits = await asyncio.gather(
        openf1.get_positions(session), openf1.get_drivers(session), openf1.get_pit(session)
    )
    start = end - timedelta(seconds=lookback_sec)
    by_driver: dict[int, list[tuple[datetime, int]]] = {}
    for row in positions or []:
        dn, pos, when = row.get("driver_number"), row.get("position"), _dt(row.get("date"))
        if dn is None or pos is None or when is None or when > end:
            continue
        by_driver.setdefault(int(dn), []).append((when, int(pos)))
    for rows in by_driver.values():
        rows.sort(key=lambda item: item[0])

    subject_rows = by_driver.get(int(driver_number), [])
    event = None
    previous = None
    for when, pos in subject_rows:
        if when < start:
            previous = (when, pos)
            continue
        if previous is not None and pos < previous[1]:
            event = {"time": when, "before": previous[1], "after": pos}
        previous = (when, pos)
    if event is None:
        return {"available": False, "note": "최근 추월 순위 교환을 찾지 못했어요."}

    # 같은 순간에 subject의 이전 순위로 내려간 차량을 방금 추월한 상대로 본다.
    target = None
    tolerance = timedelta(seconds=3)
    for dn, rows in by_driver.items():
        if dn == int(driver_number):
            continue
        prev = None
        for when, pos in rows:
            if when > event["time"] + tolerance:
                break
            if prev is not None and abs((when - event["time"]).total_seconds()) <= 3:
                if prev[1] == event["after"] and pos == event["before"]:
                    target = dn
                    break
            prev = (when, pos)
        if target is not None:
            break
    if target is None:
        return {"available": False, "note": "방금 추월한 상대 차량을 확인하지 못했어요."}

    # Position swaps around a pit stop are classification changes, not proof of
    # an on-track overtake.  Reject them instead of inventing a racing cause.
    if any(
        row.get("driver_number") in (int(driver_number), target)
        and _dt(row.get("date")) is not None
        and abs((_dt(row.get("date")) - event["time"]).total_seconds()) <= 30
        for row in (pits or [])
    ):
        return {"available": False, "note": "피트 구간의 순위 변동이라 실제 트랙 추월로 단정할 수 없어요."}

    event_time = event["time"]
    event_iso = event_time.isoformat()
    window_start = (event_time - timedelta(seconds=8)).isoformat()
    (intervals, car_data, target_car_data, locations, subj_laps, target_laps,
     subj_stints, target_stints, race_control, metadata) = await asyncio.gather(
        openf1.get_intervals(session, driver_number),
        openf1.get_car_data_window(session, driver_number, window_start, event_iso),
        openf1.get_car_data_window(session, target, window_start, event_iso),
        openf1.get_location_window(session, driver_number, window_start, event_iso),
        openf1.get_laps(session, driver_number), openf1.get_laps(session, target),
        openf1.get_stints(session, driver_number), openf1.get_stints(session, target),
        openf1.get_race_control(session), openf1.get_session_metadata(session),
    )

    gaps: list[tuple[datetime, float]] = []
    for row in intervals or []:
        when = _dt(row.get("date"))
        if when is None or not (event_time - timedelta(seconds=12) <= when < event_time):
            continue
        try:
            gap = float(row.get("interval"))
        except (TypeError, ValueError):
            continue
        if gap >= 0:
            gaps.append((when, gap))
    gaps.sort(key=lambda item: item[0])

    drs_active = any(row.get("drs") in (10, 12, 14) for row in (car_data or []))
    target_drs_active = any(row.get("drs") in (10, 12, 14) for row in (target_car_data or []))

    def _number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _paired_speed_metrics(subject_rows, target_rows):
        """Compare same-time telemetry only (<=0.5 s apart)."""
        target_valid = [(row, _dt(row.get("date")), _number(row.get("speed")))
                        for row in (target_rows or [])]
        target_valid = [(row, when, speed) for row, when, speed in target_valid
                        if when is not None and speed is not None]
        pairs = []
        for row in subject_rows or []:
            when, speed = _dt(row.get("date")), _number(row.get("speed"))
            if when is None or speed is None or not target_valid:
                continue
            target_row, target_when, target_speed = min(
                target_valid, key=lambda item: abs((item[1] - when).total_seconds())
            )
            if abs((target_when - when).total_seconds()) <= 0.5:
                pairs.append({
                    "date": row.get("date"), "subject_speed": speed,
                    "target_speed": target_speed, "delta": speed - target_speed,
                    "subject_drs": row.get("drs") in (10, 12, 14),
                    "target_drs": target_row.get("drs") in (10, 12, 14),
                    "subject_throttle": _number(row.get("throttle")),
                    "target_throttle": _number(target_row.get("throttle")),
                    "subject_brake": bool(row.get("brake")),
                    "target_brake": bool(target_row.get("brake")),
                    "subject_gear": row.get("n_gear"), "target_gear": target_row.get("n_gear"),
                })
        if not pairs:
            return {"sample_count": 0}
        # The approach phase is best represented by samples where the subject is
        # still accelerating/using DRS; braking samples can invert the comparison.
        approach = [p for p in pairs if p["subject_drs"] and not p["subject_brake"]] or \
                   [p for p in pairs if not p["subject_brake"]] or pairs
        deltas = [p["delta"] for p in approach]
        peak = max(approach, key=lambda p: p["subject_speed"])
        return {
            "sample_count": len(pairs), "approach_sample_count": len(approach),
            "mean_advantage_kmh": round(sum(deltas) / len(deltas), 1),
            "max_advantage_kmh": round(max(deltas), 1),
            "subject_peak_speed_kmh": round(peak["subject_speed"]),
            "target_speed_at_peak_kmh": round(peak["target_speed"]),
            "subject_drs_active": any(p["subject_drs"] for p in pairs),
            "target_drs_active": any(p["target_drs"] for p in pairs),
        }

    speed = _paired_speed_metrics(car_data, target_car_data)

    def _recent_lap_pace(laps):
        completed = []
        for lap in laps or []:
            start_at = _dt(lap.get("date_start"))
            duration = _number(lap.get("lap_duration"))
            if start_at is None or duration is None:
                continue
            if start_at + timedelta(seconds=duration) <= event_time and not lap.get("is_pit_out_lap"):
                completed.append((int(lap.get("lap_number") or 0), duration))
        recent = sorted(completed)[-3:]
        if not recent:
            return {"laps": [], "mean_seconds": None}
        return {"laps": [lap for lap, _ in recent],
                "mean_seconds": round(sum(duration for _, duration in recent) / len(recent), 3)}

    subject_pace, target_pace = _recent_lap_pace(subj_laps), _recent_lap_pace(target_laps)
    pace_delta = None
    if subject_pace["mean_seconds"] is not None and target_pace["mean_seconds"] is not None:
        pace_delta = round(target_pace["mean_seconds"] - subject_pace["mean_seconds"], 3)

    # Track location: generic progress/sector plus conservative named zones for
    # Suzuka.  These zones are broad; they describe where, not why the pass occurred.
    track_progress = None
    track_zone = None
    sector = None
    if locations:
        try:
            from ..ml import track_ref
            ref = await track_ref.get_reference(session)
            loc = max((row for row in locations if row.get("x") is not None and row.get("y") is not None),
                      key=lambda row: row.get("date") or "", default=None)
            if ref is not None and loc is not None:
                track_progress = round(track_ref.project(ref, loc["x"], loc["y"]), 3)
        except Exception:
            logger.exception("추월 위치 투영 실패")
    circuit = (metadata or {}).get("circuit_short_name")
    if track_progress is not None:
        sector = 1 if track_progress < 0.36 else (2 if track_progress < 0.72 else 3)
        if str(circuit).lower() == "suzuka":
            p = track_progress
            if p >= 0.96 or p < 0.12:
                track_zone = "메인 스트레이트"
            elif p < 0.18:
                track_zone = "1·2번 코너 진입 구간"
            elif p < 0.34:
                track_zone = "S 커브 구간"
            elif p < 0.43:
                track_zone = "덩롭 구간"
            elif p < 0.52:
                track_zone = "데그너 구간"
            elif p < 0.61:
                track_zone = "헤어핀 구간"
            elif p < 0.75:
                track_zone = "스푼 구간"
            elif p < 0.90:
                track_zone = "백 스트레이트·130R 구간"
            else:
                track_zone = "시케인 구간"

    neutralized = []
    for row in race_control or []:
        when = _dt(row.get("date"))
        if when is None or abs((when - event_time).total_seconds()) > 30:
            continue
        message = str(row.get("message") or "").upper()
        if row.get("category") in ("SafetyCar", "Flag") and any(
            word in message for word in ("YELLOW", "SAFETY", "RED FLAG", "VSC")
        ):
            neutralized.append(row.get("message") or row.get("category"))

    def _tyre(laps, stints):
        lap_no = max(
            (int(row["lap_number"]) for row in (laps or [])
             if row.get("lap_number") is not None
             and (_dt(row.get("date_start")) or end) <= event_time),
            default=None,
        )
        if lap_no is None:
            return {"compound": None, "age_laps": None}
        current = next((st for st in (stints or [])
                        if st.get("lap_start") is not None
                        and int(st["lap_start"]) <= lap_no
                        and (st.get("lap_end") is None or lap_no <= int(st["lap_end"]))), None)
        if current is None:
            return {"compound": None, "age_laps": None}
        age = lap_no - int(current["lap_start"]) + 1 + int(current.get("tyre_age_at_start") or 0)
        return {"compound": current.get("compound"), "age_laps": age}

    name_of = {int(row["driver_number"]): row.get("full_name") or row.get("name_acronym")
               for row in (drivers or []) if row.get("driver_number") is not None}
    evidence = []
    if gaps:
        evidence.append({"level": "CONFIRMED", "factor": "gap_closing",
                         "detail": f"{round(gaps[0][1], 2)}s -> {round(gaps[-1][1], 2)}s"})
    if speed.get("sample_count"):
        evidence.append({"level": "CONFIRMED", "factor": "relative_speed",
                         "detail": f"mean {speed.get('mean_advantage_kmh')} km/h"})
    if drs_active:
        evidence.append({"level": "CONFIRMED", "factor": "drs", "detail": "subject active"})
    evidence.append({"level": "CONFIRMED" if not neutralized else "UNKNOWN",
                     "factor": "race_control",
                     "detail": "normal" if not neutralized else "; ".join(neutralized)})

    return {
        "available": True,
        "subject_driver": int(driver_number),
        "subject_name": name_of.get(int(driver_number)),
        "target_driver": target,
        "target_name": name_of.get(target),
        "event_time": event_iso,
        "gap_start_sec": round(gaps[0][1], 2) if gaps else None,
        "gap_end_sec": round(gaps[-1][1], 2) if gaps else None,
        "drs_active": drs_active,
        "target_drs_active": target_drs_active,
        "speed_comparison": speed,
        "track": {"circuit": circuit, "progress": track_progress,
                  "sector": sector, "zone": track_zone},
        "recent_pace": {"subject": subject_pace, "target": target_pace,
                        "subject_advantage_sec": pace_delta},
        "race_control_clear": not neutralized,
        "race_control_events": neutralized,
        "subject_tyre": _tyre(subj_laps, subj_stints),
        "target_tyre": _tyre(target_laps, target_stints),
        "evidence": evidence,
        "unsupported_causes": ["driver_error", "racing_line", "team_intent"],
    }


# ────────────────────────────── 시점 전환 ──────────────────────────────

@tool
def toggle_drone_view(on: bool) -> str:
    """드론(공중) 시점을 켜거나 끈다.
    "드론으로 봐", "공중에서 봐", "위에서 전체 보여줘" → on=True.
    "드론 꺼", "원래 시점으로" → on=False.
    주의: 특정 선수를 그냥 보고 싶은 거면(누구야/어디 있어) 이 도구가 아니라 highlight_driver를 쓴다.
    드론은 화면 전체가 바뀌는 '큰 시점 전환'이라, 사용자가 드론/공중을 명시적으로 요청할 때만 쓴다."""
    emit_command("droneView", on=on)
    return "드론 시점으로 전환했어요." if on else "원래 시점으로 돌아왔어요."


# 능동형 pointOutMoment는 사용자 발화가 아니라 서버 백그라운드 감시 루프가
# 발동한다(app/agent/watcher.py, 확장 단계). 도구 목록에는 넣지 않는다.

ALL_TOOLS = [
    find_session,
    get_driver_info,
    get_race_status,
    explain_concept,
    explain_why,
    highlight_driver,
    control_replay,
    jump_to_event,
    predict_overtake,
    show_battle_context,
    recommend_battle_action,
    explain_situation,
    toggle_drone_view,
]
