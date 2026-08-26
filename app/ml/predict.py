"""추월 예측 — 저장된 LightGBM 부스터 로드 + 피처 예측 + isotonic 보정(표시확률).

계약(unity_contract.json)이 단일 소스:
  - feature_order : 입력 피처 순서(26). 이 순서대로 벡터를 만든다.
  - missing_value_fill : 계산 못 한 피처를 채울 값(-1.0). 학습과 동일.
  - outputs[target] : 모델 파일·보정 파일·표시 출력명.
보정: raw 확률을 raw_thresholds→display_values 선형보간으로 표시확률(0~1)로 바꾼다.

lightgbm/numpy는 지연 import — 미설치여도 모듈 import는 되고, predict 호출 시에만 필요.
"""
from __future__ import annotations

import json
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_RUN = "races_initial_event_type_final"
_CONTRACT = _MODELS_DIR / f"{_RUN}_unity_contract.json"
_SUPPORTED_SCHEMA = "event_type_v1_26"
_SUPPORTED_TIME_UNIT = "seconds"

# 지연 로드 캐시 (첫 predict에서 1회 로드)
_state: dict = {
    "loaded": False,
    "order": None,
    "fill": -1.0,
    "outputs": {},
    "temporal": None,
    "contract": None,
}

_TEMPORAL_MODEL = _MODELS_DIR / "suzuka_temporal_event_label_overtake.txt"
_TEMPORAL_BASE_MODEL = _MODELS_DIR / "suzuka_temporal_event_base_label_overtake.txt"
_TEMPORAL_METADATA = _MODELS_DIR / "suzuka_temporal_event_model.json"
_TEMPORAL_ORDER = [
    "base_probability", "gap_ahead", "gap_delta_3s", "gap_delta_5s", "gap_delta_10s",
    "gap_min_5s", "gap_std_10s", "closing_ratio_10s", "speed_delta",
    "speed_delta_mean_5s", "speed_delta_mean_10s", "speed_advantage_ratio_10s",
    "drs_active", "drs_active_ratio_10s", "drs_range", "track_progress_sin",
    "track_progress_cos", "sector", "segment",
]


def _basename(path_str: str) -> str:
    """계약의 model_file은 'data\\models\\x.txt'(윈도우 구분자)일 수 있어 파일명만 안전 추출."""
    return Path(path_str.replace("\\", "/")).name


def _load() -> None:
    if _state["loaded"]:
        return
    import lightgbm as lgb   # 지연 import

    contract = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    schema = contract.get("schema_version")
    if schema != _SUPPORTED_SCHEMA:
        raise RuntimeError(
            f"Unsupported ML feature schema: {schema!r}; expected {_SUPPORTED_SCHEMA!r}"
        )
    if contract.get("time_unit") != _SUPPORTED_TIME_UNIT:
        raise RuntimeError(
            "ML contract time_unit must be 'seconds'; refusing a possible ms/sec mismatch"
        )
    order = contract.get("feature_order") or []
    if len(order) != int(contract.get("feature_count", -1)) or len(order) != len(set(order)):
        raise RuntimeError("Invalid ML contract: feature_count/order mismatch or duplicate names")
    _state["order"] = contract["feature_order"]
    _state["fill"] = float(contract.get("missing_value_fill", -1.0))
    _state["contract"] = contract

    outputs = {}
    for _target, o in contract["outputs"].items():
        booster = lgb.Booster(model_file=str(_MODELS_DIR / _basename(o["model_file"])))
        if booster.num_feature() != len(order):
            raise RuntimeError(
                f"Model {o['model_file']} expects {booster.num_feature()} features, "
                f"but contract declares {len(order)}"
            )
        cal = json.loads((_MODELS_DIR / _basename(o["calibration_file"])).read_text(encoding="utf-8"))
        outputs[o["output_name"]] = {
            "booster": booster,
            "raw_thresholds": cal.get("raw_thresholds", []),
            "display_values": cal.get("display_values", []),
        }
    _state["outputs"] = outputs
    if _TEMPORAL_MODEL.exists():
        temporal_metadata = json.loads(_TEMPORAL_METADATA.read_text(encoding="utf-8"))
        if temporal_metadata.get("model_bundle_version") != contract.get("model_bundle_version"):
            raise RuntimeError("Temporal model metadata does not match deployed model bundle")
        if temporal_metadata.get("time_unit") != _SUPPORTED_TIME_UNIT:
            raise RuntimeError("Temporal model metadata time_unit mismatch")
        _state["temporal"] = lgb.Booster(model_file=str(_TEMPORAL_MODEL))
        _state["temporal_base"] = lgb.Booster(model_file=str(_TEMPORAL_BASE_MODEL))
        if _state["temporal"].num_feature() != len(_TEMPORAL_ORDER):
            raise RuntimeError("Temporal model feature count does not match runtime order")
        if _state["temporal_base"].num_feature() != len(order):
            raise RuntimeError("Temporal base model feature count does not match runtime order")
    _state["loaded"] = True


def model_info() -> dict:
    """Return the exact deployed model/feature contract for health checks and reports."""
    _load()
    contract = _state["contract"]
    return {
        "run_name": contract["run_name"],
        "model_bundle_version": contract["model_bundle_version"],
        "feature_schema_version": contract["schema_version"],
        "feature_count": len(_state["order"]),
        "time_unit": contract["time_unit"],
        "missing_value_fill": _state["fill"],
        "temporal_model_enabled": _state["temporal"] is not None,
    }


def _calibrate(raw: float, thresholds: list, values: list) -> float:
    import numpy as np
    if not thresholds:
        return float(raw)
    return float(np.interp(raw, thresholds, values))


def predict(feats: dict) -> dict:
    """피처 dict → {표시출력명: 확률(0~1)} 4종.

    feats에 없는 피처는 계약의 결측값(-1.0)으로 채운다(모델이 그렇게 학습됨).
    """
    import numpy as np
    _load()

    order, fill = _state["order"], _state["fill"]
    x = np.array([[float(feats.get(name, fill)) for name in order]], dtype=float)

    result = {}
    for name, o in _state["outputs"].items():
        raw = float(o["booster"].predict(x)[0])
        display_override = None
        if name == "overtake_probability" and _state["temporal"] is not None:
            temporal_feats = dict(feats)
            temporal_feats["base_probability"] = float(_state["temporal_base"].predict(x)[0])
            temporal_x = np.array(
                [[float(temporal_feats.get(key, fill)) for key in _TEMPORAL_ORDER]],
                dtype=float,
            )
            temporal_raw = float(_state["temporal"].predict(temporal_x)[0])
            # Stable UI operating point: temporal raw 0.60 maps to the existing
            # watcher threshold 0.30. The generic model's isotonic curve must not be
            # applied again: it was fitted to a different model and caused double calibration.
            display_override = _calibrate(
                temporal_raw, [0.0, 0.6, 1.0], [0.0, 0.3, 1.0]
            )
        display = (
            display_override
            if display_override is not None
            else _calibrate(raw, o["raw_thresholds"], o["display_values"])
        )
        result[name] = round(display, 4)
    return result


def coverage(feats: dict) -> dict:
    """디버그용 — 계약 피처 중 실제 계산된 개수/목록(나머지는 결측 채움)."""
    _load()
    order = _state["order"]
    computed = [n for n in order if n in feats]
    return {"computed": computed, "computed_count": len(computed), "total": len(order)}
