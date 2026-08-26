import asyncio

import pytest

from app.agent import watcher


@pytest.fixture
def _isolate_watcher_safety_gates(monkeypatch):
    """Existing loop tests focus on their named branch; gate behavior has dedicated tests."""
    async def no_neutralization(*_args, **_kwargs):
        return None

    async def no_pit_blocks(*_args, **_kwargs):
        return set()

    monkeypatch.setattr(watcher, "_session_neutralization_reason", no_neutralization)
    monkeypatch.setattr(watcher, "_pit_blocked_drivers", no_pit_blocks)
    monkeypatch.setattr(watcher.settings, "watcher_min_consecutive_hits", 1)


def _confirmed(driver=44):
    return [{
        "driver": driver,
        "gap": 0.4,
        "trend": -0.1,
        "position_before": 8,
        "position_after": 7,
        "event_time": "2025-04-06T05:11:51+00:00",
    }]


async def _started(*_args, **_kwargs):
    return True


def test_candidate_score_prefers_explainable_closing_battle():
    small_gap_but_flat = {"driver": 5, "gap": 0.36, "trend": 0.0}
    wider_gap_but_closing = {"driver": 44, "gap": 0.44, "trend": -0.058}

    assert watcher._candidate_score(wider_gap_but_closing) > watcher._candidate_score(small_gap_but_flat)


async def test_race_start_gate_blocks_formation_and_allows_lap1(monkeypatch):
    watcher._race_start_cache.clear()

    async def laps(*_args, **_kwargs):
        return [
            {"driver_number": 1, "lap_number": 1, "date_start": "2025-04-06T05:03:53.788Z"},
            {"driver_number": 44, "lap_number": 1, "date_start": "2025-04-06T05:03:53.788Z"},
        ]

    monkeypatch.setattr(watcher.openf1, "get_laps", laps)

    assert not await watcher._race_has_started(10006, "2025-04-06T05:03:40.000Z")
    assert await watcher._race_has_started(10006, "2025-04-06T05:03:53.788Z")
    assert await watcher._race_has_started(10006, "2025-04-06T05:04:10.000Z")


async def test_race_start_gate_fails_closed_without_lap_start(monkeypatch):
    watcher._race_start_cache.clear()

    async def no_laps(*_args, **_kwargs):
        return []

    monkeypatch.setattr(watcher.openf1, "get_laps", no_laps)
    assert not await watcher._race_has_started(10006, "2025-04-06T05:04:10.000Z")


async def test_pit_gate_blocks_candidate_when_car_ahead_is_pitting(monkeypatch):
    async def positions(*_args, **_kwargs):
        return [
            {"driver_number": 1, "position": 7, "date": "2025-04-06T05:10:00Z"},
            {"driver_number": 44, "position": 8, "date": "2025-04-06T05:10:00Z"},
        ]

    async def pits(*_args, **_kwargs):
        return [{
            "driver_number": 1,
            "date": "2025-04-06T05:10:20Z",
            "pit_duration": 20.0,
        }]

    monkeypatch.setattr(watcher.openf1, "get_positions", positions)
    monkeypatch.setattr(watcher.openf1, "get_pit", pits)
    blocked = await watcher._pit_blocked_drivers(
        10006,
        "2025-04-06T05:10:05Z",
        [{"driver": 44, "gap": 0.4, "trend": -0.1}],
    )
    assert blocked == {44}


async def test_neutralization_gate_blocks_safety_car_and_restart_buffer(monkeypatch):
    async def controls(*_args, **_kwargs):
        return [
            {"date": "2025-04-06T05:10:00Z", "message": "SAFETY CAR DEPLOYED"},
            {"date": "2025-04-06T05:11:00Z", "message": "GREEN FLAG"},
        ]

    monkeypatch.setattr(watcher.openf1, "get_race_control", controls)
    assert await watcher._session_neutralization_reason(
        10006, "2025-04-06T05:10:30Z"
    ) == "safety_car"
    assert await watcher._session_neutralization_reason(
        10006, "2025-04-06T05:11:05Z"
    ) == "restart_buffer"
    assert await watcher._session_neutralization_reason(
        10006, "2025-04-06T05:11:20Z"
    ) is None


def test_consecutive_hit_gate_rejects_sparse_probability_spikes():
    hits = {}
    assert not watcher._register_consecutive_hit(
        hits, 44, "ml", "2025-04-06T05:11:30Z", 3.0, 2
    )
    assert watcher._register_consecutive_hit(
        hits, 44, "ml", "2025-04-06T05:11:31Z", 3.0, 2
    )
    assert not watcher._register_consecutive_hit(
        hits, 44, "ml", "2025-04-06T05:11:40Z", 3.0, 2
    )


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


async def test_watcher_reannounces_confirmed_event_after_replay_seek(
    monkeypatch, _isolate_watcher_safety_gates
):
    monkeypatch.setattr(watcher, "_race_has_started", _started)
    monkeypatch.setattr(watcher.settings, "watcher_replay_confirmation_enabled", True)
    monkeypatch.setattr(watcher.settings, "watcher_hybrid_enabled", False)
    monkeypatch.setattr(watcher.settings, "watcher_fast_hybrid_enabled", False)
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


async def test_watcher_suppresses_announce_if_paused_before_emit(
    monkeypatch, _isolate_watcher_safety_gates
):
    monkeypatch.setattr(watcher, "_race_has_started", _started)
    monkeypatch.setattr(watcher.settings, "watcher_replay_confirmation_enabled", True)
    monkeypatch.setattr(watcher.settings, "watcher_hybrid_enabled", False)
    monkeypatch.setattr(watcher.settings, "watcher_fast_hybrid_enabled", False)
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


async def test_watcher_does_not_fire_gap_trend_when_ml_probability_is_low(
    monkeypatch, _isolate_watcher_safety_gates
):
    monkeypatch.setattr(watcher, "_race_has_started", _started)
    monkeypatch.setattr(watcher.settings, "watcher_replay_confirmation_enabled", False)
    monkeypatch.setattr(watcher.settings, "watcher_hybrid_enabled", True)
    monkeypatch.setattr(watcher.settings, "watcher_fast_hybrid_enabled", False)
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

    async def low_probability_features(*_args, **_kwargs):
        return {"gap_ahead": 0.44, "gap_trend": -0.058, "is_lap1": 0.0}

    monkeypatch.setattr(watcher.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(watcher.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(watcher, "_battle_candidates", battle_candidates)
    monkeypatch.setattr(watcher, "_confirmed_gain_candidates", confirmed_candidates)
    monkeypatch.setattr(watcher.watcher_eval, "log_prediction", lambda *a, **k: None)

    from app.ml import features
    monkeypatch.setattr(features, "build_features", low_probability_features)
    from app.ml import predict
    monkeypatch.setattr(predict, "predict", lambda _feats: {"overtake_probability": 0.04})

    try:
        await watcher.watch(get_state, announce)
    except asyncio.CancelledError:
        pass

    assert announced == []
