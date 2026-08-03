"""조회형 도구 단위테스트 — 가짜(mock) 데이터로 로직만 검사(OpenF1 불필요).

OpenF1/Jolpica 호출을 monkeypatch로 가짜 응답으로 바꿔치기해, 도구가 데이터를
'받았을 때 제대로 가공하는지'만 본다. 네트워크·서버·키 전부 불필요.
"""
from app.data import openf1
from app.agent.context import set_context, set_session, current_session
from app.agent.tools import get_driver_info, get_race_status, explain_why, find_session


def _aret(value):
    """항상 value를 반환하는 async 가짜 함수."""
    async def _f(*args, **kwargs):
        return value
    return _f


# ───────────────────────── get_driver_info ─────────────────────────

async def test_driver_info_merges_career(monkeypatch):
    monkeypatch.setattr(openf1, "get_driver", _aret({
        "driver_number": 44, "full_name": "Lewis HAMILTON", "last_name": "Hamilton",
        "name_acronym": "HAM", "team_name": "Mercedes", "country_code": "GBR",
        "headshot_url": "http://x/h.png",
    }))
    monkeypatch.setattr(openf1, "get_career", _aret({
        "dateOfBirth": "1985-01-07", "nationality": "British", "wins": 106,
    }))
    set_session(9523)

    out = await get_driver_info.ainvoke({"driver_number": 44})
    assert out["name"] == "Lewis HAMILTON"
    assert out["team"] == "Mercedes"
    assert out["career_wins"] == 106
    assert out["nationality"] == "British"


async def test_driver_info_not_found(monkeypatch):
    monkeypatch.setattr(openf1, "get_driver", _aret(None))
    set_session(9523)
    out = await get_driver_info.ainvoke({"driver_number": 999})
    assert "error" in out


async def test_driver_info_survives_career_failure(monkeypatch):
    monkeypatch.setattr(openf1, "get_driver", _aret({
        "driver_number": 44, "full_name": "Lewis HAMILTON",
        "last_name": "Hamilton", "team_name": "Mercedes",
    }))

    async def boom(*a, **k):
        raise RuntimeError("career server down")

    monkeypatch.setattr(openf1, "get_career", boom)
    set_session(9523)

    out = await get_driver_info.ainvoke({"driver_number": 44})
    assert out["name"] == "Lewis HAMILTON"     # 기본 정보는 그대로 반환
    assert "career_wins" not in out            # 커리어 실패는 조용히 생략


# ───────────────────────── get_race_status ─────────────────────────

async def test_race_status_no_spoiler(monkeypatch):
    """at_time 이후(미래)의 순위·깃발은 노출하지 않아야 한다."""
    positions = [
        {"date": "2024-05-26T13:10:00+00:00", "driver_number": 16, "position": 1},
        {"date": "2024-05-26T13:10:00+00:00", "driver_number": 44, "position": 5},
        {"date": "2024-05-26T13:50:00+00:00", "driver_number": 55, "position": 3},  # 미래
    ]
    rc = [
        {"date": "2024-05-26T13:09:00+00:00", "category": "Flag", "flag": "CLEAR", "message": "CLEAR"},
        {"date": "2024-05-26T13:50:00+00:00", "category": "Flag", "flag": "BLUE", "message": "BLUE"},  # 미래
    ]
    monkeypatch.setattr(openf1, "get_positions", _aret(positions))
    monkeypatch.setattr(openf1, "get_race_control", _aret(rc))
    set_context(9523, "2024-05-26T13:20:00+00:00")

    out = await get_race_status.ainvoke({})
    nums = [s["driver_number"] for s in out["standings"]]
    assert 55 not in nums                       # 미래 순위 스포일러 안 함
    assert out["latest_flag"]["flag"] == "CLEAR"  # 미래 BLUE도 제외
    assert out["standings"][0]["driver_number"] == 16  # 순위 오름차순
    assert out["at_time"] == "2024-05-26T13:20:00+00:00"


async def test_race_status_safety_car(monkeypatch):
    monkeypatch.setattr(openf1, "get_positions", _aret([]))
    monkeypatch.setattr(openf1, "get_race_control", _aret([
        {"date": "2024-05-26T13:05:00+00:00", "category": "SafetyCar",
         "message": "SAFETY CAR DEPLOYED"},
    ]))
    set_context(9523, None)
    out = await get_race_status.ainvoke({})
    assert out["safety_car"] is True


# ───────────────────────── explain_why ─────────────────────────

async def test_explain_why_structure(monkeypatch):
    monkeypatch.setattr(openf1, "get_pit",
                        _aret([{"date": "a", "lap_number": 1}, {"date": "b", "lap_number": 20}]))
    monkeypatch.setattr(openf1, "get_stints",
                        _aret([{"compound": "MEDIUM"}, {"compound": "HARD"}]))
    monkeypatch.setattr(openf1, "get_intervals",
                        _aret([{"interval": 2.0}, {"interval": 1.055}]))
    set_context(9523, None)   # at_time 없음 → 전체(=최신) 반환, 필터 없음

    out = await explain_why.ainvoke({"driver_number": 44})
    assert out["driver_number"] == 44
    assert out["recent_gap"]["interval"] == 1.055   # 가장 최근 갭
    assert len(out["pit_stops"]) == 2
    assert "hint" in out


async def test_explain_why_no_spoiler(monkeypatch):
    """at_time(현재 시각) 이후의 피트·타이어·갭은 감춰야 한다(스포일러 방지)."""
    # 현재 시각을 "m"으로 두면 "a"는 과거, "z"는 미래.
    monkeypatch.setattr(openf1, "get_pit",
                        _aret([{"date": "a", "lap_number": 5}, {"date": "z", "lap_number": 40}]))
    monkeypatch.setattr(openf1, "get_stints",
                        _aret([{"stint_number": 1, "compound": "MEDIUM"},
                               {"stint_number": 2, "compound": "HARD"},
                               {"stint_number": 3, "compound": "SOFT"}]))
    monkeypatch.setattr(openf1, "get_intervals",
                        _aret([{"date": "a", "interval": 2.0}, {"date": "z", "interval": 0.5}]))
    set_context(9523, "m")   # 리플레이 현재 시각 = "m"

    out = await explain_why.ainvoke({"driver_number": 44})
    assert len(out["pit_stops"]) == 1                 # 미래 피트("z") 숨김
    assert out["recent_gap"]["interval"] == 2.0       # 미래 갭(0.5) 아닌 현재 갭
    compounds = [s["compound"] for s in out["tire_stints"]]
    assert "SOFT" not in compounds                    # 아직 안 낀 미래 타이어 숨김
    assert compounds == ["MEDIUM", "HARD"]


# ───────────────────────── find_session ─────────────────────────

async def test_find_session_found(monkeypatch):
    monkeypatch.setattr(openf1, "find_sessions", _aret([{
        "session_key": 9523, "country_name": "Monaco",
        "circuit_short_name": "Monte Carlo", "year": 2024, "session_name": "Race",
    }]))
    out = await find_session.ainvoke({"year": 2024, "race": "Monaco"})
    assert out["found"] is True
    assert out["session_key"] == 9523
    assert current_session() == 9523    # 세션이 실제로 전환됨


async def test_find_session_not_found(monkeypatch):
    monkeypatch.setattr(openf1, "find_sessions", _aret([]))
    out = await find_session.ainvoke({"year": 1800, "race": "Nowhere"})
    assert out["found"] is False
