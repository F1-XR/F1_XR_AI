"""로컬(네트워크 불필요) 도구 테스트 — explain_concept + 용어집 무결성.

실행: pytest   (OpenF1 없이 오프라인으로 동작)
"""
import json
from pathlib import Path

from app.agent.tools import explain_concept


def test_explain_concept_known():
    out = explain_concept.invoke({"term": "DRS"})
    assert out["explanation"]  # 용어집에 있으면 설명 반환


def test_explain_concept_partial():
    out = explain_concept.invoke({"term": "drs 존"})
    assert out["term"] == "DRS"  # 부분 일치 허용


def test_explain_concept_unknown():
    out = explain_concept.invoke({"term": "없는용어xyz"})
    assert out["explanation"] is None  # 없으면 None + 안내


def test_glossary_has_enough_terms():
    path = Path(__file__).parent.parent / "app" / "data" / "glossary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) >= 20  # 기획 목표 20~40개
    # 모든 설명이 비어 있지 않은 문자열
    assert all(isinstance(v, str) and v.strip() for v in data.values())
    # 핵심 입문 용어는 반드시 포함
    for term in ["DRS", "피트스톱", "세이프티카", "언더컷"]:
        assert term in data
