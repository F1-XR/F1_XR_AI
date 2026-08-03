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

# 지연 로드 캐시 (첫 predict에서 1회 로드)
_state: dict = {"loaded": False, "order": None, "fill": -1.0, "outputs": {}}


def _basename(path_str: str) -> str:
    """계약의 model_file은 'data\\models\\x.txt'(윈도우 구분자)일 수 있어 파일명만 안전 추출."""
    return Path(path_str.replace("\\", "/")).name


def _load() -> None:
    if _state["loaded"]:
        return
    import lightgbm as lgb   # 지연 import

    contract = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    _state["order"] = contract["feature_order"]
    _state["fill"] = float(contract.get("missing_value_fill", -1.0))

    outputs = {}
    for _target, o in contract["outputs"].items():
        booster = lgb.Booster(model_file=str(_MODELS_DIR / _basename(o["model_file"])))
        cal = json.loads((_MODELS_DIR / _basename(o["calibration_file"])).read_text(encoding="utf-8"))
        outputs[o["output_name"]] = {
            "booster": booster,
            "raw_thresholds": cal.get("raw_thresholds", []),
            "display_values": cal.get("display_values", []),
        }
    _state["outputs"] = outputs
    _state["loaded"] = True


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
        result[name] = round(_calibrate(raw, o["raw_thresholds"], o["display_values"]), 4)
    return result


def coverage(feats: dict) -> dict:
    """디버그용 — 계약 피처 중 실제 계산된 개수/목록(나머지는 결측 채움)."""
    _load()
    order = _state["order"]
    computed = [n for n in order if n in feats]
    return {"computed": computed, "computed_count": len(computed), "total": len(order)}
