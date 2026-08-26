import json

import pytest

from app.ml import predict


def _reset():
    predict._state.update({
        "loaded": False,
        "order": None,
        "fill": -1.0,
        "outputs": {},
        "temporal": None,
        "contract": None,
    })


def test_deployed_contract_declares_version_and_seconds():
    contract = json.loads(predict._CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema_version"] == "event_type_v1_26"
    assert contract["model_bundle_version"]
    assert contract["time_unit"] == "seconds"
    assert contract["feature_count"] == len(contract["feature_order"])
    assert len(contract["feature_order"]) == len(set(contract["feature_order"]))


def test_loader_rejects_millisecond_contract(monkeypatch, tmp_path):
    contract = json.loads(predict._CONTRACT.read_text(encoding="utf-8"))
    contract["time_unit"] = "milliseconds"
    bad = tmp_path / "bad_contract.json"
    bad.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(predict, "_CONTRACT", bad)
    _reset()
    with pytest.raises(RuntimeError, match="time_unit"):
        predict._load()
    _reset()
