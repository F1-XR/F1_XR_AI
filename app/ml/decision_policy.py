"""Deterministic battle action policy over probabilistic AI outputs.

This module intentionally keeps the decision rule outside the LLM/tool layer so
runtime recommendations and offline evaluation use the same logic.
"""
from __future__ import annotations

from typing import Any


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recommend_battle_policy(
    *,
    gap_seconds: float | None,
    predicted_gap_seconds: float | None,
    predicted_gap_std_seconds: float | None,
    trend: str | None,
    drs: bool,
    fusion_used: bool = False,
    relative_speed_kmh: float | None = None,
    overtake_probability: float | None = None,
) -> dict[str, Any]:
    """Return a compact action recommendation from state-estimation outputs."""
    gap = _float_or_none(gap_seconds)
    predicted_gap = _float_or_none(predicted_gap_seconds)
    predicted_std = _float_or_none(predicted_gap_std_seconds)
    prob = _float_or_none(overtake_probability)

    closing_expected = (
        gap is not None and predicted_gap is not None and
        gap - predicted_gap >= 0.10
    )
    opening_expected = (
        gap is not None and predicted_gap is not None and
        predicted_gap - gap >= 0.10
    )
    high_uncertainty = predicted_std is not None and predicted_std >= 0.35

    action = "HOLD_PRESSURE"
    risk = "medium"
    confidence = 0.55
    reason = "간격과 추세를 보면 압박은 가능하지만 즉시 공격 신호는 강하지 않아요."

    if high_uncertainty:
        action = "LOW_CONFIDENCE"
        risk = "high"
        confidence = 0.35
        reason = "예측 불확실성이 커서 공격 판단을 보수적으로 봐야 해요."
    elif gap is not None and gap <= 1.0 and drs and (closing_expected or trend == "closing"):
        action = "PRESS_ATTACK"
        risk = "medium"
        confidence = 0.76
        reason = "DRS 범위 안이고, 갭이 더 좁혀질 가능성이 있어 공격 압박을 걸 만해요."
    elif gap is not None and gap <= 1.2 and prob is not None and prob >= 0.20:
        action = "PRESS_ATTACK"
        risk = "medium"
        confidence = 0.70
        reason = "보정된 추월 확률과 현재 갭이 공격 가능 구간을 가리켜요."
    elif gap is not None and gap <= 1.2 and not drs:
        action = "WAIT_FOR_DRS"
        risk = "low"
        confidence = 0.68
        reason = "간격은 가깝지만 DRS 신호가 없어, 바로 공격보다 DRS 구간을 기다리는 편이 좋아요."
    elif opening_expected or trend == "opening":
        action = "HOLD_POSITION"
        risk = "low"
        confidence = 0.64
        reason = "예측상 간격이 벌어지는 흐름이라 무리한 공격보다는 위치 유지가 나아요."

    if fusion_used and confidence < 0.95:
        confidence += 0.04

    return {
        "action": action,
        "confidence": round(min(confidence, 0.95), 2),
        "risk": risk,
        "reason": reason,
        "inputs": {
            "gap_seconds": gap_seconds,
            "predicted_gap_seconds": predicted_gap_seconds,
            "predicted_gap_std_seconds": predicted_gap_std_seconds,
            "trend": trend,
            "drs": bool(drs),
            "fusion_used": bool(fusion_used),
            "relative_speed_kmh": relative_speed_kmh,
            "overtake_probability": overtake_probability,
        },
    }
