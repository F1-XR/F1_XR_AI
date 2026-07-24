"""LangGraph 에이전트 — 의도 파악 → 툴콜 루프 → 한국어 응답.

핵심 파이프라인:
    사용자 텍스트
      → LLM이 '답할까 / 도구 부를까' 판단 (ReAct 루프)
      → 도구 실행 (조회형은 데이터 반환 / 명령형은 Unity 명령 적재)
      → 도구 결과를 다시 LLM에 → 최종 한국어 응답 생성
    반환: (응답 텍스트, Unity로 보낼 명령 리스트)

초기엔 LangGraph 프리빌트 create_react_agent로 충분하다. 질문 유형별
분기(정보/제어/강의)가 필요해지면 그때 명시적 그래프로 확장한다.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config import settings
from .commands import drain, start_capture
from .context import set_context
from .tools import ALL_TOOLS

SYSTEM_PROMPT = """당신은 F1을 처음 보는 사람을 위한 친절한 관람 가이드입니다.

원칙:
- 사용자가 특정 경기(연도·나라·서킷)를 언급하면 먼저 find_session으로 그 경기로 전환한 뒤 답하세요.
- 항상 도구로 얻은 실제 데이터에 근거해 답하세요. 모르면 모른다고 하세요(지어내지 말 것).
- 초등학생도 이해할 만큼 쉽고 짧게, 존댓말로 설명하세요.
- 선수를 언급하면 필요 시 highlight_driver로 화면에서 함께 짚어주세요.
- "천천히 다시 보여줘"처럼 여러 동작이 섞인 요청은 도구를 순서대로 여러 번 호출하세요.
- 전문용어가 나오면 먼저 explain_concept로 뜻을 풀어주세요.
"""


def build_agent():
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.openai_api_key or None,
    )
    try:
        return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
    except TypeError:
        # 구버전 langgraph 호환 (prompt → state_modifier)
        return create_react_agent(llm, ALL_TOOLS, state_modifier=SYSTEM_PROMPT)


# 에이전트는 첫 요청 때 lazy 생성한다.
# import 시점에 만들면 키·버전 문제 시 서버가 통째로 안 떠서 /health 도 죽는다.
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 없습니다. .env에 키를 넣어야 에이전트가 동작합니다."
            )
        _agent = build_agent()
    return _agent


async def run_agent(
    text: str,
    session_key: int | None = None,
    at_time: str | None = None,
    history: list | None = None,
) -> tuple[str, list[dict]]:
    """한 번의 사용자 발화를 처리한다.

    Returns:
        reply: 사용자에게 보낼 한국어 텍스트(이후 TTS로 음성화)
        commands: Unity로 보낼 명령 리스트(마커·리플레이·점프)
    """
    set_context(session_key, at_time)   # 이번 요청의 세션/시각 고정
    start_capture()                     # 명령 버퍼 열기

    messages = list(history or []) + [("user", text)]
    result = await get_agent().ainvoke({"messages": messages})
    reply = result["messages"][-1].content

    commands = drain()                  # 도구가 쌓아둔 Unity 명령 회수
    return reply, commands
