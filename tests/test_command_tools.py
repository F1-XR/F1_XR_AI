"""명령형 도구 단위테스트 — Unity로 보낼 명령이 제대로 방출되는지(OpenF1 불필요).

start_capture()로 명령 버퍼를 열고, 도구 실행 후 drain()으로 쌓인 명령을 확인한다.
Unity 없이도 '올바른 명령을 내보내는가'는 검증 가능하다.
"""
from app.data import openf1
from app.agent.commands import start_capture, drain
from app.agent.context import set_session
from app.agent.tools import highlight_driver, control_replay, jump_to_event


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
