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
from .context import current_recent_overtake, current_session, current_time, set_session

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

    if event_type == "first_pit":
        pit = await openf1.get_pit(session)
        pit = [p for p in pit if p.get("date")]
        if pit:
            target_time = min(pit, key=lambda p: p["date"])["date"]
    elif event_type in ("first_overtake", "first_overtake_before"):
        recent = current_recent_overtake()
        if event_type == "first_overtake_before" and recent and recent.get("event_time"):
            target_time = recent["event_time"]
        else:
            positions = [
                p for p in await openf1.get_positions(session)
                if p.get("date") and p.get("driver_number") is not None and p.get("position") is not None
            ]
            if cutoff:
                positions = [p for p in positions if p["date"] <= cutoff]
            positions.sort(key=lambda p: p["date"])
            prev_pos: dict[int, int] = {}
            for p in positions:
                dn, pos = int(p["driver_number"]), int(p["position"])
                before = prev_pos.get(dn)
                if before is not None and pos < before:
                    target_time = p["date"]
                    break
                prev_pos[dn] = pos
    else:
        rc = await openf1.get_race_control(session)
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
    driver = await openf1.get_driver(session, driver_number)   # 이름 근거(할루시네이션 방지)
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
    iv = _before(await openf1.get_intervals(session, driver_number), cutoff)
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

    # 3) DRS — subject의 car_data 창(현재 시각 근처). drs 코드 10/12/14 = 작동 중.
    drs = False
    if cutoff:
        start = _iso_minus(cutoff, 4)
        if start:
            cd = [c for c in await openf1.get_car_data_window(session, driver_number, start, cutoff)
                  if c.get("date") and c["date"] <= cutoff]
            if cd:
                drs = cd[-1].get("drs") in (10, 12, 14)

    # 4) confidence(간단 휴리스틱): 가깝고 좁혀질수록 높게
    conf = 0.0
    if gap is not None:
        conf = max(0.0, min(1.0, (1.2 - gap) * (0.7 if trend == "closing" else 0.4)))

    emit_command(
        "showBattleContext",
        subject_driver=driver_number,
        target_driver=target,
        gap_seconds=round(gap, 2) if gap is not None else 0.0,
        trend=trend,
        drs=bool(drs),
        confidence=round(conf, 2),
        reason="앞차와의 간격·추세·DRS 기반",
    )
    return {
        "shown": True,
        "subject_driver": driver_number,
        "target_driver": target,
        "gap_seconds": round(gap, 2) if gap is not None else None,
        "trend": trend,
        "drs": bool(drs),
        "note": "두 차 사이에 Gap Line과 배지를 표시했어요. 이 수치를 바탕으로 한두 문장으로 짧게 설명하세요.",
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
    toggle_drone_view,
]
