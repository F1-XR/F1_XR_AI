"""도구 기본 동작 테스트 (오프라인 위주).

실행: pytest   (pip install pytest 필요)
네트워크가 필요한 조회 도구는 통합 테스트로 분리 예정.
"""
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
