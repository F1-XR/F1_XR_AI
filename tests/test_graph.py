"""run_agent 단위테스트 — 정상 응답 + 오류 시 우아한 실패(대화 안 끊김).

LLM(에이전트)을 가짜로 바꿔치기해 네트워크·키 없이 검사한다.
"""
import app.agent.graph as graph
from app.agent.commands import start_capture, emit_command


class _Msg:
    def __init__(self, content, type="ai"):
        self.content = content
        self.type = type


class _FakeAgent:
    def __init__(self, reply="테스트 응답", boom=False, emit=None):
        self._reply = reply
        self._boom = boom
        self._emit = emit

    async def ainvoke(self, payload):
        if self._emit:
            # 도구가 명령을 쌓는 상황 흉내
            emit_command(*self._emit)
        if self._boom:
            raise RuntimeError("LLM down")
        return {"messages": [_Msg(self._reply)]}


async def test_run_agent_happy(monkeypatch):
    monkeypatch.setattr(graph, "get_agent", lambda: _FakeAgent(reply="안녕하세요"))
    reply, commands, _ = await graph.run_agent("안녕", session_key=9523)
    assert reply == "안녕하세요"
    assert commands == []


async def test_run_agent_error_is_graceful(monkeypatch):
    # LLM이 예외를 던져도 run_agent는 던지지 않고 안내 문구 + 빈 명령을 반환해야 한다.
    monkeypatch.setattr(graph, "get_agent", lambda: _FakeAgent(boom=True))
    reply, commands, _ = await graph.run_agent("왜 피트인?", session_key=9523)
    assert "죄송" in reply
    assert commands == []


async def test_run_agent_error_discards_partial_commands(monkeypatch):
    # 도구가 명령을 쌓은 뒤 실패하면, 반쪽 명령은 보내지 않는다(빈 리스트).
    monkeypatch.setattr(
        graph, "get_agent",
        lambda: _FakeAgent(boom=True, emit=("highlightDriver",)),
    )
    reply, commands, _ = await graph.run_agent("해밀턴 강조", session_key=9523)
    assert "죄송" in reply
    assert commands == []


def test_salvage_battle_action_result():
    messages = [
        _Msg("앞차 추월 시도해도 돼?", type="human"),
        _Msg({
            "available": True,
            "action": "PRESS_ATTACK",
            "reason": "DRS 범위 안이고 갭이 더 좁혀질 가능성이 있어요.",
            "inputs": {"gap_seconds": 0.8},
        }, type="tool"),
    ]
    reply = graph._salvage_from_tools(messages, "앞차 추월 시도해도 돼?")
    assert "공격 압박" in reply
    assert "DRS" in reply


def test_recent_overtake_sentence_does_not_invent_tyre_advantage():
    reply = graph._recent_overtake_sentence({
        "available": True,
        "subject_driver": 44,
        "subject_name": "Lewis HAMILTON",
        "target_driver": 6,
        "target_name": "Isack HADJAR",
        "gap_start_sec": 0.38,
        "gap_end_sec": 0.01,
        "drs_active": True,
        "subject_tyre": {"compound": "HARD", "age_laps": 6},
        "target_tyre": {"compound": "MEDIUM", "age_laps": 6},
    })
    assert "0.38초에서 0.01초" in reply
    assert "DRS" in reply
    assert "타이어 우위를 핵심 원인으로 단정할 근거는 없습니다" in reply
    assert "코너" not in reply


def test_why_sentence_does_not_invent_pit_cause():
    reply = graph._why_sentence({
        "driver_number": 44,
        "driver_name": "Lewis HAMILTON",
        "pit_stops": [{"date": "2025-04-06T05:20:00+00:00"}],
        "tire_stints": [{"compound": "HARD"}],
    })
    assert "확인" in reply
    assert "원인을 단정할 수는 없어요" in reply
    assert "위해 피트인" not in reply


async def test_rule_router_recent_overtake_is_grounded(monkeypatch):
    async def fake_context(driver_number):
        assert driver_number == 44
        return {
            "available": True,
            "subject_driver": 44,
            "subject_name": "Lewis HAMILTON",
            "target_driver": 6,
            "target_name": "Isack HADJAR",
            "gap_start_sec": 0.38,
            "gap_end_sec": 0.01,
            "drs_active": True,
            "subject_tyre": {"compound": "HARD", "age_laps": 6},
            "target_tyre": {"compound": "MEDIUM", "age_laps": 6},
        }

    monkeypatch.setattr(graph, "get_recent_overtake_context", fake_context)
    reply, commands, ok = await graph.run_agent(
        "방금 어떻게 추월했어?", session_key=10006,
        at_time="2025-04-06T05:11:55+00:00", selected_driver=44,
    )
    assert ok is True
    assert commands == []
    assert "DRS" in reply
    assert "타이어 우위를 핵심 원인으로 단정할 근거는 없습니다" in reply


async def test_rule_router_resolves_korean_driver_name(monkeypatch):
    async def fake_context(driver_number):
        assert driver_number == 44
        return {
            "available": True, "subject_driver": 44, "subject_name": "Lewis HAMILTON",
            "target_driver": 6, "target_name": "Isack HADJAR",
            "gap_start_sec": 0.39, "gap_end_sec": 0.01, "drs_active": True,
            "subject_tyre": {"age_laps": 6}, "target_tyre": {"age_laps": 6},
        }

    monkeypatch.setattr(graph, "get_recent_overtake_context", fake_context)
    reply, commands, ok = await graph.run_agent(
        "해밀턴이 어떻게 추월했어?", session_key=10006,
        at_time="2025-04-06T05:11:55+00:00", selected_driver=None,
    )
    assert ok is True
    assert commands == []
    assert "DRS" in reply


def test_recent_overtake_sentence_uses_relative_speed_and_limits_claims():
    reply = graph._recent_overtake_sentence({
        "subject_name": "Lewis HAMILTON", "target_name": "Isack HADJAR",
        "gap_start_sec": 0.39, "gap_end_sec": 0.01, "drs_active": True,
        "speed_comparison": {
            "mean_advantage_kmh": 8.4, "subject_peak_speed_kmh": 303,
            "target_speed_at_peak_kmh": 294,
        },
        "track": {"zone": "1·2번 코너 진입 구간"},
        "subject_tyre": {"age_laps": 6}, "target_tyre": {"age_laps": 6},
        "recent_pace": {"subject_advantage_sec": 0.25},
        "race_control_clear": True,
    })
    assert "평균 상대속도가 약 8.4km/h 높았고" in reply
    assert "1·2번 코너 진입 구간" in reply
    assert "주요 기여 요인은 DRS·실측 상대속도 우위·지속적인 간격 감소" in reply
    assert "운전자 실수나 의도는 이 데이터만으로 확인할 수 없습니다" in reply


def test_probability_sentence_keeps_exact_model_value_and_context():
    reply = graph._overtake_probability_sentence({
        "driver_number": 44,
        "driver_name": "Lewis HAMILTON",
        "overtake_probability": 0.333,
        "inputs": {"gap_ahead": 0.38, "speed_delta": 15.0},
    })
    assert "33%" in reply
    assert "어려" not in reply
    assert "0.38초" in reply
    assert "15.0km/h" in reply


async def test_explicit_driver_name_overrides_stale_selection_for_probability(monkeypatch):
    class FakePredictTool:
        async def ainvoke(self, payload):
            number = payload["driver_number"]
            assert number == 63
            return {
                "driver_number": number,
                "driver_name": "George RUSSELL",
                "overtake_probability": 0.81,
                "inputs": {"gap_ahead": 0.23},
            }

    monkeypatch.setattr(graph, "predict_overtake", FakePredictTool())
    reply, commands, ok = await graph.run_agent(
        "러셀 추월 가능성 있어?",
        session_key=10006,
        at_time="2025-04-06T05:36:44+00:00",
        selected_driver=44,
    )
    assert ok is True
    assert commands == []
    assert "81%" in reply
