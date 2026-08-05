"""로컬/호환 LLM(예: Gemma 4) 연결 + tool calling 지원 확인 스크립트.

서버 전체 안 켜고 .env 의 LLM_BASE_URL / OPENAI_API_KEY / LLM_MODEL 만으로
① 엔드포인트가 살아있나(기본 대화) ② function/tool calling 을 지원하나 를 각각 확인한다.

실행(F1_XR_AI 루트에서):
    python -m scripts.test_local_llm

기대:
  [1] 기본 대화  → 답변 텍스트가 나오면 연결 OK
  [2] 툴 콜      → tool_calls 가 나오면 도구 사용 가능(에이전트 정상 작동 가능)
                   안 나오면 이 엔드포인트는 tool calling 미지원 → 조회·강조·예측 도구가 안 불림
"""
import json
import os

from openai import OpenAI

from app.config import settings   # .env 값을 그대로 사용

BASE_URL = settings.llm_base_url or os.environ.get("LLM_BASE_URL")
API_KEY = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
MODEL = settings.llm_model


def main() -> None:
    print(f"base_url = {BASE_URL}")
    print(f"model    = {MODEL}")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=30.0)

    # ── [1] 기본 대화 (연결 확인) ─────────────────────────
    print("\n[1] 기본 대화 테스트…")
    r1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "간결하게 한국어로 답하세요."},
            {"role": "user", "content": "F1에서 DRS가 뭐야?"},
        ],
        temperature=0.3,
        max_tokens=1024,   # 추론형 모델이 생각에 토큰을 쓸 수 있어 넉넉히
    )
    m1 = r1.choices[0].message
    text = (m1.content or "").strip()
    print("  응답:", text or "(빈 응답)")
    print("  finish_reason:", r1.choices[0].finish_reason)
    # 일부 추론형 모델은 답을 content 대신 reasoning_content 에 담기도 함 → 있으면 같이 출력
    reasoning = getattr(m1, "reasoning_content", None)
    if not text and reasoning:
        print("  reasoning_content(앞 200자):", reasoning[:200])
    print("  raw usage:", r1.usage)
    print("  → 연결", "OK ✅" if text else "본문 빔 ⚠️ (아래 finish_reason/usage 확인)")

    # ── [2] tool calling 지원 확인 ────────────────────────
    print("\n[2] tool calling 테스트…")
    tools = [{
        "type": "function",
        "function": {
            "name": "get_race_status",
            "description": "현재 경기 상황(순위 등)을 조회한다.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]
    r2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "필요하면 도구를 호출하세요."},
            {"role": "user", "content": "지금 1등 누구야?"},
        ],
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
        max_tokens=200,
    )
    msg = r2.choices[0].message
    calls = getattr(msg, "tool_calls", None)
    if calls:
        print("  tool_calls:", [c.function.name for c in calls])
        print("  → tool calling 지원 ✅  (에이전트 도구 정상 작동 가능)")
    else:
        print("  tool_calls 없음. 그냥 텍스트:", (msg.content or "").strip()[:120])
        print("  → tool calling 미지원 ❌  (조회·강조·예측 도구가 안 불림 — 엔드포인트/모델 설정 확인)")


if __name__ == "__main__":
    main()
