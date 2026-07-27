"""에이전트 모델 검증 — LLM이 실제로 응답하는지 최소 확인 (도구·서버 없이).

실행: python -m scripts.check_agent
- .env 의 LLM_MODEL / OPENAI_API_KEY 로 짧은 한 마디를 보내 응답을 확인한다.
- 실패하면 원인을 알려주고, 모델 문제면 사용 가능한 모델 목록을 출력한다.
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()  # .env → 환경변수

from app.config import settings


async def main() -> None:
    print(f"모델 검증: LLM_MODEL={settings.llm_model!r}")
    if not settings.openai_api_key:
        print("❌ OPENAI_API_KEY가 비어 있어요. .env를 확인하세요.")
        return

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        resp = await llm.ainvoke("한국어로 '안녕하세요'라고만 답해줘.")
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        print(f"✅ 응답 성공: {text.strip()[:80]}")
        print("→ 모델·키 정상. 이제 `python -m app.cli` 로 에이전트를 쓰면 됩니다.")
    except Exception as exc:
        print(f"❌ 실패: {type(exc).__name__}: {exc}")
        msg = str(exc).lower()
        if "model" in msg or "invalid" in msg or "not found" in msg:
            print("\n원인: 모델 ID 문제로 보여요. 사용 가능한 모델(gpt 계열):")
            try:
                from openai import OpenAI

                ids = sorted(m.id for m in OpenAI().models.list().data if "gpt" in m.id)
                for m in ids:
                    print("  -", m)
                print("\n→ 위 중 하나를 .env 의 LLM_MODEL 에 넣고 다시 실행하세요.")
            except Exception as e2:
                print("  (모델 목록 조회도 실패:", e2, ")")
        elif "key" in msg or "auth" in msg or "credential" in msg:
            print("\n원인: 인증 문제. .env 의 OPENAI_API_KEY 를 확인하세요.")


if __name__ == "__main__":
    asyncio.run(main())
