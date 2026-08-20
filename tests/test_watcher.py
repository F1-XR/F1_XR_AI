import asyncio

from app.agent import watcher


def _confirmed(driver=44):
    return [{
        "driver": driver,
        "gap": 0.4,
        "trend": -0.1,
        "position_before": 8,
        "position_after": 7,
        "event_time": "2025-04-06T05:11:51+00:00",
    }]


def test_candidate_score_prefers_explainable_closing_battle():
    small_gap_but_flat = {"driver": 5, "gap": 0.36, "trend": 0.0}
    wider_gap_but_closing = {"driver": 44, "gap": 0.44, "trend": -0.058}

    assert watcher._candidate_score(wider_gap_but_closing) > watcher._candidate_score(small_gap_but_flat)


async def test_confirmed_gain_candidates_ignore_past_position_gain(monkeypatch):
    rows = [
        {"driver_number": 44, "date": "2025-04-06T05:11:20+00:00", "position": 8},
        {"driver_number": 44, "date": "2025-04-06T05:11:40+00:00", "position": 7},
    ]

    async def positions(*_args, **_kwargs):
        return rows

    monkeypatch.setattr(watcher.openf1, "get_positions", positions)

    out = await watcher._confirmed_gain_candidates(
        10006,
        "2025-04-06T05:11:50+00:00",
        [{"driver": 44, "gap": 0.4, "trend": -0.1}],
    )

    assert out == []


async def test_confirmed_gain_candidates_keep_future_position_gain(monkeypatch):
    rows = [
        {"driver_number": 44, "date": "2025-04-06T05:11:20+00:00", "position": 8},
        {"driver_number": 44, "date": "2025-04-06T05:11:40+00:00", "position": 7},
    ]

    async def positions(*_args, **_kwargs):
        return rows

    monkeypatch.setattr(watcher.openf1, "get_positions", positions)

    out = await watcher._confirmed_gain_candidates(
        10006,
        "2025-04-06T05:11:30+00:00",
        [{"driver": 44, "gap": 0.4, "trend": -0.1}],
    )

    assert out[0]["driver"] == 44
    assert out[0]["event_time"] == "2025-04-06T05:11:40+00:00"


async def test_confirmed_gain_candidates_prioritize_current_battle_gain(monkeypatch):
    rows = [
        {"driver_number": 5, "date": "2025-04-06T05:11:20+00:00", "position": 6},
        {"driver_number": 5, "date": "2025-04-06T05:11:40+00:00", "position": 5},
        {"driver_number": 44, "date": "2025-04-06T05:11:20+00:00", "position": 8},
        {"driver_number": 44, "date": "2025-04-06T05:11:40+00:00", "position": 7},
    ]

    async def positions(*_args, **_kwargs):
        return rows

    monkeypatch.setattr(watcher.openf1, "get_positions", positions)

    out = await watcher._confirmed_gain_candidates(
        10006,
        "2025-04-06T05:11:30+00:00",
        [{"driver": 44, "gap": 0.4, "trend": -0.1}],
    )

    assert out[0]["driver"] == 44
    assert any(c["driver"] == 5 for c in out)


async def test_watcher_reannounces_confirmed_event_after_replay_seek(monkeypatch):
    states = [
        {"session_key": 10006, "at_time": "2025-04-06T05:11:31+00:00", "is_playing": True},
        {"session_key": 10006, "at_time": "2025-04-06T05:11:32+00:00", "is_playing": True},
        {"session_key": 10006, "at_time": "2025-04-06T05:11:12+00:00", "is_playing": True},
    ]
    tick = {"i": -1}
    announced = []

    async def fake_sleep(_period):
        tick["i"] += 1
        if tick["i"] >= len(states):
            raise asyncio.CancelledError

    def get_state():
        return states[min(tick["i"], len(states) - 1)]

    async def announce(*args, **kwargs):
        announced.append(args)

    monkeypatch.setattr(watcher.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: tick["i"] * 3.0)
    monkeypatch.setattr(watcher, "_battle_candidates", lambda *a, **k: asyncio.sleep(0) or [])

    async def battle_candidates(*_args, **_kwargs):
        return [{"driver": 44, "gap": 0.4, "trend": -0.1}]

    async def confirmed_candidates(*_args, **_kwargs):
        return _confirmed(44)

    monkeypatch.setattr(watcher, "_battle_candidates", battle_candidates)
    monkeypatch.setattr(watcher, "_confirmed_gain_candidates", confirmed_candidates)
    monkeypatch.setattr(watcher.watcher_eval, "log_prediction", lambda *a, **k: None)

    try:
        await watcher.watch(get_state, announce)
    except asyncio.CancelledError:
        pass

    assert len(announced) == 2


async def test_watcher_suppresses_announce_if_paused_before_emit(monkeypatch):
    state = {"session_key": 10006, "at_time": "2025-04-06T05:11:31+00:00", "is_playing": True}
    tick = {"i": -1}
    announced = []

    async def fake_sleep(_period):
        tick["i"] += 1
        if tick["i"] >= 1:
            raise asyncio.CancelledError

    def get_state():
        return state

    async def announce(*args, **kwargs):
        announced.append(args)

    async def battle_candidates(*_args, **_kwargs):
        return [{"driver": 44, "gap": 0.4, "trend": -0.1}]

    async def confirmed_candidates(*_args, **_kwargs):
        state["is_playing"] = False
        return _confirmed(44)

    monkeypatch.setattr(watcher.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(watcher, "_battle_candidates", battle_candidates)
    monkeypatch.setattr(watcher, "_confirmed_gain_candidates", confirmed_candidates)
    monkeypatch.setattr(watcher.watcher_eval, "log_prediction", lambda *a, **k: None)

    try:
        await watcher.watch(get_state, announce)
    except asyncio.CancelledError:
        pass

    assert announced == []


async def test_watcher_fires_gap_trend_before_slow_ml(monkeypatch):
    state = {"session_key": 10006, "at_time": "2025-04-06T05:11:31+00:00", "is_playing": True}
    tick = {"i": -1}
    announced = []

    async def fake_sleep(_period):
        tick["i"] += 1
        if tick["i"] >= 1:
            raise asyncio.CancelledError

    def get_state():
        return state

    async def announce(*args, **kwargs):
        announced.append(args)

    async def battle_candidates(*_args, **_kwargs):
        return [{"driver": 44, "gap": 0.44, "trend": -0.058}]

    async def confirmed_candidates(*_args, **_kwargs):
        return []

    async def slow_build_features(*_args, **_kwargs):
        raise AssertionError("gap/trend fire should happen before ML feature building")

    monkeypatch.setattr(watcher.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(watcher, "_battle_candidates", battle_candidates)
    monkeypatch.setattr(watcher, "_confirmed_gain_candidates", confirmed_candidates)
    monkeypatch.setattr(watcher.watcher_eval, "log_prediction", lambda *a, **k: None)

    from app.ml import features
    monkeypatch.setattr(features, "build_features", slow_build_features)

    try:
        await watcher.watch(get_state, announce)
    except asyncio.CancelledError:
        pass

    assert announced
    assert announced[0][0] == 44
