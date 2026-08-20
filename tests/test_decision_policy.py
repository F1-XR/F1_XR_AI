import json

from app.ml.decision_policy import recommend_battle_policy
from scripts.eval_decision_policy import evaluate


def test_decision_policy_press_attack():
    out = recommend_battle_policy(
        gap_seconds=0.8,
        predicted_gap_seconds=0.55,
        predicted_gap_std_seconds=0.12,
        trend="closing",
        drs=True,
        fusion_used=True,
    )
    assert out["action"] == "PRESS_ATTACK"
    assert out["confidence"] > 0.76
    assert out["inputs"]["fusion_used"] is True


def test_decision_policy_wait_for_drs():
    out = recommend_battle_policy(
        gap_seconds=0.9,
        predicted_gap_seconds=0.85,
        predicted_gap_std_seconds=0.12,
        trend="stable",
        drs=False,
    )
    assert out["action"] == "WAIT_FOR_DRS"


def test_eval_decision_policy_intervals_only(tmp_path):
    rows = []
    for i in range(50):
        # Driver 44 steadily closes from 1.4s to 0.42s.
        rows.append({
            "date": f"2024-05-26T13:20:{i:02d}+00:00",
            "driver_number": 44,
            "interval": 1.4 - i * 0.02,
        })
        # Driver 16 slowly opens from 0.7s to 1.19s.
        rows.append({
            "date": f"2024-05-26T13:20:{i:02d}+00:00",
            "driver_number": 16,
            "interval": 0.7 + i * 0.01,
        })
    path = tmp_path / "intervals.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    result = evaluate(path, outcome_horizon=10.0, stride=5)
    assert result["sample_count"] > 0
    assert "PRESS_ATTACK" in result["by_action"]
    assert result["by_action"]["PRESS_ATTACK"]["n"] > 0
