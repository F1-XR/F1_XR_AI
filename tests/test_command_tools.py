"""명령형 도구 단위테스트 — Unity로 보낼 명령이 제대로 방출되는지(OpenF1 불필요).

start_capture()로 명령 버퍼를 열고, 도구 실행 후 drain()으로 쌓인 명령을 확인한다.
Unity 없이도 '올바른 명령을 내보내는가'는 검증 가능하다.
"""
from app.data import openf1
from app.agent.commands import start_capture, drain
from app.agent.context import set_context, set_session
from app.agent.tools import (
    highlight_driver,
    control_replay,
    jump_to_event,
    recommend_battle_action,
    show_battle_context,
)


def _aret(value):
    async def _f(*args, **kwargs):
        return value
    return _f


def test_highlight_driver_emits():
    start_capture()
    msg = highlight_driver.invoke({"driver_number": 44})
    cmds = drain()
    assert len(cmds) == 1
    assert cmds[0]["name"] == "highlightDriver"
    assert cmds[0]["args"]["driver_number"] == 44
    assert "44" in msg


def test_control_replay_emits():
    start_capture()
    control_replay.invoke({"action": "speed", "value": 0.5})
    cmds = drain()
    assert cmds[0]["name"] == "controlReplay"
    assert cmds[0]["args"] == {"action": "speed", "value": 0.5}


async def test_jump_to_event_first_pit(monkeypatch):
    set_session(9523)
    # 가장 이른 피트 시각으로 seek 명령을 보내야 한다
    monkeypatch.setattr(openf1, "get_pit", _aret([
        {"date": "2024-05-26T13:44:36+00:00", "driver_number": 44},
        {"date": "2024-05-26T13:20:00+00:00", "driver_number": 16},
    ]))
    start_capture()
    await jump_to_event.ainvoke({"event_type": "first_pit"})
    cmds = drain()
    assert cmds[0]["name"] == "controlReplay"
    assert cmds[0]["args"]["action"] == "seek"
    assert cmds[0]["args"]["value"] == "2024-05-26T13:20:00+00:00"   # 가장 이른 피트


async def test_show_battle_context_uses_relative_speed_fusion(monkeypatch):
    set_context(9523, "2024-05-26T13:20:08+00:00")
    monkeypatch.setattr(openf1, "get_positions", _aret([
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 16, "position": 3},
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 44, "position": 4},
    ]))
    monkeypatch.setattr(openf1, "get_intervals", _aret([
        {"date": "2024-05-26T13:20:00+00:00", "driver_number": 44, "interval": 1.20},
        {"date": "2024-05-26T13:20:04+00:00", "driver_number": 44, "interval": 0.95},
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 44, "interval": 0.80},
    ]))

    async def car_data(session, driver_number, start, end):
        if driver_number == 44:
            return [
                {"date": "2024-05-26T13:20:04+00:00", "speed": 294, "drs": 12},
                {"date": "2024-05-26T13:20:08+00:00", "speed": 296, "drs": 12},
            ]
        if driver_number == 16:
            return [
                {"date": "2024-05-26T13:20:04+00:00", "speed": 288, "drs": 0},
                {"date": "2024-05-26T13:20:08+00:00", "speed": 289, "drs": 0},
            ]
        return []

    monkeypatch.setattr(openf1, "get_car_data_window", car_data)

    start_capture()
    out = await show_battle_context.ainvoke({"driver_number": 44})
    cmds = drain()

    assert out["shown"] is True
    assert out["fusion_used"] is True
    assert out["relative_speed_kmh"] == 7.0
    assert cmds[0]["name"] == "showBattleContext"
    assert cmds[0]["args"]["fusion_used"] is True
    assert cmds[0]["args"]["relative_speed_kmh"] == 7.0
    assert cmds[0]["args"]["drs"] is True


async def test_recommend_battle_action_press_attack(monkeypatch):
    set_context(9523, "2024-05-26T13:20:08+00:00")
    monkeypatch.setattr(openf1, "get_positions", _aret([
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 16, "position": 3},
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 44, "position": 4},
    ]))
    monkeypatch.setattr(openf1, "get_intervals", _aret([
        {"date": "2024-05-26T13:20:00+00:00", "driver_number": 44, "interval": 1.20},
        {"date": "2024-05-26T13:20:04+00:00", "driver_number": 44, "interval": 0.95},
        {"date": "2024-05-26T13:20:08+00:00", "driver_number": 44, "interval": 0.80},
    ]))

    async def car_data(session, driver_number, start, end):
        if driver_number == 44:
            return [{"date": "2024-05-26T13:20:08+00:00", "speed": 296, "drs": 12}]
        if driver_number == 16:
            return [{"date": "2024-05-26T13:20:08+00:00", "speed": 289, "drs": 0}]
        return []

    monkeypatch.setattr(openf1, "get_car_data_window", car_data)

    start_capture()
    out = await recommend_battle_action.ainvoke({"driver_number": 44})
    cmds = drain()

    assert out["available"] is True
    assert out["action"] == "PRESS_ATTACK"
    assert out["inputs"]["fusion_used"] is True
    assert cmds[0]["name"] == "showBattleContext"
