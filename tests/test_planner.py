from app.agent.planner import build_command_plan, normalize_command_order


def _kinds(text, selected=None):
    return [s.kind for s in build_command_plan(text, selected)]


def test_compound_overtake_before_plan():
    steps = build_command_plan("그 추월 장면 직전으로 돌아가서 천천히 보여줘", selected_driver=44)
    assert [(s.kind, s.args) for s in steps] == [
        ("seek_event", {"event_type": "first_overtake_before"}),
        ("replay", {"action": "speed", "value": 0.5}),
        ("highlight", {"driver_number": 44}),
    ]


def test_compound_pit_and_slow_plan():
    steps = build_command_plan("첫 피트스톱 장면으로 가서 천천히 보여줘")
    assert [(s.kind, s.args) for s in steps] == [
        ("seek_event", {"event_type": "first_pit"}),
        ("replay", {"action": "speed", "value": 0.5}),
    ]


def test_selected_battle_plan():
    assert _kinds("앞차랑 얼마나 붙었어?", selected=44) == ["battle_context"]


def test_control_and_drone_plan():
    assert build_command_plan("멈춰")[0].args == {"action": "pause", "value": None}
    assert build_command_plan("드론 시점으로 바꿔줘")[0].args == {"on": True}
    assert build_command_plan("원래 시점으로 돌아가")[0].args == {"on": False}


def test_relative_scene_plan():
    steps = build_command_plan("방금 장면 다시 보여줘")
    assert [(s.kind, s.args) for s in steps] == [
        ("seek_relative", {"seconds": 5.0}),
    ]


def test_seconds_before_slow_plan():
    steps = build_command_plan("5초 전부터 천천히 보여줘")
    assert [(s.kind, s.args) for s in steps] == [
        ("seek_relative", {"seconds": 5.0}),
        ("replay", {"action": "speed", "value": 0.5}),
    ]


def test_minutes_before_highlight_plan():
    steps = build_command_plan("1분 전으로 돌아가서 44번 강조해줘")
    assert [(s.kind, s.args) for s in steps] == [
        ("seek_relative", {"seconds": 60.0}),
        ("highlight", {"driver_number": 44}),
    ]


def test_replay_does_not_rewind():
    assert _kinds("다시 재생해") == ["replay"]


def test_command_order_normalization():
    commands = [
        {"type": "command", "name": "highlightDriver", "args": {"driver_number": 44}},
        {"type": "command", "name": "controlReplay", "args": {"action": "speed", "value": 0.5}},
        {"type": "command", "name": "controlReplay", "args": {"action": "seek", "value": "t"}},
    ]
    ordered = normalize_command_order(commands)
    assert [c["args"].get("action") for c in ordered[:2]] == ["seek", "speed"]
    assert ordered[2]["name"] == "highlightDriver"
