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

import ast
import json
import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config import settings
from .commands import drain, emit_command, start_capture
from .context import current_selected, current_session, current_time, set_context
from .planner import build_command_plan, execute_command_plan, normalize_command_order
from .tools import ALL_TOOLS, get_driver_info, jump_to_event, show_battle_context

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 F1을 처음 보는 사람을 위한 친절한 관람 가이드입니다.

원칙:
- 사용자는 지금 리플레이 화면을 보며 그 경기에 대해 묻습니다. 아래 '현재 관람 맥락'에
  경기가 지정되어 있으면 '어느 경기인지'는 되묻지 말고 그 경기 기준으로 답하세요.
- 단, 선수·장면·행동의 '대상'이 불명확하면(예: "쟤 왜 저래?"의 '쟤'가 누군지 모호)
  지어내지 말고 무엇을 말하는지 짧게 되물어 확인하세요.
  (경기는 이미 알지만, 대상이 애매하면 확인해도 됩니다.)
- 사용자가 다른 경기(연도·나라·서킷)로 바꾸길 원하면 그때만 find_session을 호출하세요.
- 항상 도구로 얻은 실제 데이터에 근거해 답하세요. 모르면 모른다고 하세요(지어내지 말 것).
- 초등학생도 이해할 만큼 쉽고, 존댓말로 설명하세요.
- 기본은 쉽고 짧게 답하고, 사용자가 "더 자세히·왜·무슨 뜻이야"라고 하면 그때 한 단계 더 깊이 풀어서 설명하세요(입문자 눈높이는 유지, 전문용어는 풀어서).
- ⚠️ 답변은 음성(TTS)으로 읽어줍니다. 그러니 **2~3문장, 최대 60자 안팎**으로 아주 짧게 핵심만 말하세요.
  글머리표(-)·번호 목록·긴 나열은 쓰지 마세요(음성엔 부적합). 한 호흡에 들리게 간결하게.
  더 알려줄 게 있으면 "더 설명해드릴까요?"처럼 한 줄로 물어보고 멈추세요.
- '1등·2등'은 순위(position)이고 '1번·16번'은 차량 번호(driver_number)입니다. 둘을 절대 혼동하지 마세요.
- 선수는 약칭(VER)이 아니라 한국어 전체 이름(예: 막스 베르스타펜)으로 말하세요. 순위는 도구가 준 standings 값을 그대로 쓰고 지어내지 마세요.
- 선수를 언급하면 필요 시 highlight_driver로 화면에서 함께 짚어주세요.
- 두 차의 근접·추월 압박(간격/추세/DRS)을 물으면 show_battle_context로 두 차 사이에 공간 표시(Gap Line+배지)를 함께 띄우세요.
- "지금 상황 어때?", "무슨 전략이야?", "경기 흐름/전략 설명해줘"처럼 상황과 전략을 종합해 설명해야 하면 explain_situation을 호출하고, 그 결과(갭·타이어·DRS·추월확률)를 근거로 쉽게 먼저 2~4문장으로 해설하세요.
- "천천히 다시 보여줘"처럼 여러 동작이 섞인 요청은 도구를 순서대로 여러 번 호출하세요.
  예: "그 추월 장면 직전으로 돌아가서 천천히 보여줘" →
  jump_to_event(event_type="first_overtake_before") 호출 후
  control_replay(action="speed", value=0.5) 호출. 선택 차량이 있으면 highlight_driver도 함께 호출하세요.
- 전문용어가 나오면 먼저 explain_concept로 뜻을 풀어주세요.
- ⚠️ 매우 중요: 도구(get_driver_info 등)를 호출했다면, 그 결과를 바탕으로 **반드시 한국어 최종 문장**으로 답하세요.
  도구만 부르고 아무 말 없이 끝내지 마세요. 선수를 조회했으면 이름을 먼저 말하고
  (예: "이 선수는 막스 베르스타펜 선수예요"), 세부 정보는 화면 표시로 넘겨도 됩니다.
"""


def _context_message() -> str | None:
    """이번 발화의 '현재 관람 맥락'(경기·시각)을 LLM에게 알려주는 시스템 메시지.

    유니티가 매 발화에 session_key/at_time을 보내면, 그 값이 여기 담겨 LLM이
    '지금 무슨 경기를 보고 있는지'를 알게 된다. 그래서 '왜 피트인?' 같은 질문에도
    어느 경기인지 되묻지 않는다.
    """
    session = current_session()
    if session is None:
        return None
    lines = [
        "[현재 관람 맥락] 사용자는 지금 아래 경기 리플레이를 보고 있습니다. "
        "'어느 경기인지'는 되묻지 말고 이 경기·시각 기준으로 답하세요. "
        "(단, 선수·장면 등 '대상'이 애매하면 그건 확인차 되물어도 됩니다.)",
        f"- 현재 경기 세션 ID: {session}",
    ]
    at = current_time()
    if at:
        lines.append(f"- 리플레이 현재 시각: {at} (이 시각 이후의 미래 결과는 아직 일어나지 않았으니 언급 금지)")
    sel = current_selected()
    if sel:
        lines.append(
            f"- 사용자가 지금 화면에서 선택(지목)한 차량 번호: {sel}. "
            f"'이 선수·이 차·얘·쟤·여기'처럼 대상을 가리키는 말은 이 {sel}번 차량을 뜻합니다. "
            f"그러니 이런 지시어가 나오면 번호를 되묻지 말고 {sel}번 기준으로 도구를 호출하세요. "
            f"(단, 사용자가 다른 번호·이름을 명시하면 그 대상을 우선합니다.)"
        )
    return "\n".join(lines)


def build_agent():
    # 추론(reasoning) 모델의 '생각'을 요청 단위로 끈다. 생각 토큰이 max_tokens를 소진해
    # 최종 답(content)이 비는 문제를 막는다. 공유 서버 설정이 아니라 이 클라이언트 요청에만
    # 적용된다(vLLM/Qwen: chat_template_kwargs.enable_thinking=false).
    llm_kwargs = dict(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key or None,
        base_url=settings.llm_base_url or None,   # 로컬/호환 엔드포인트면 그쪽으로(비우면 OpenAI)
        max_tokens=settings.llm_max_tokens,       # 음성 답변용 상한(장황 방지·잘림 방지 균형)
    )
    if settings.llm_disable_thinking:
        llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    llm = ChatOpenAI(**llm_kwargs)
    # SYSTEM_PROMPT는 run_agent에서 '동적 관람 맥락'과 합쳐 하나의 system 메시지로
    # 주입한다. 여기서 prompt=로 또 넣으면 system 메시지가 둘이 되므로 넣지 않는다.
    return create_react_agent(llm, ALL_TOOLS)


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


def _msg_text(content) -> str:
    """LangChain 메시지 content(문자열 또는 content-block 리스트)를 순수 텍스트로 만든다."""
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return (content or "").strip()


def _current_turn(messages: list) -> list:
    """이번 발화 부분만(마지막 user 메시지 이후) 잘라낸다.

    run_agent는 history(이전 대화)를 함께 모델에 넘긴다. 그래서 결과 메시지를 끝에서부터
    전부 훑으면 '이번 답'이 비었을 때 이전 턴의 답변을 잘못 집을 수 있다(→ 같은 답 반복).
    반드시 이번 턴 구간만 본다.
    """
    start = 0
    for i, m in enumerate(messages):
        if getattr(m, "type", None) == "human":
            start = i + 1
    return messages[start:]


def _extract_reply(messages: list) -> str:
    """이번 턴에서 '내용이 있는' AI 메시지 텍스트를 고른다.

    create_react_agent가 도구 호출로 끝나면 마지막 메시지 content가 빈 문자열일 수 있다
    (특히 로컬/OSS 모델). 그래서 맨 끝만 보지 않고, 이번 턴 구간을 뒤에서부터 훑어
    비지 않은 AI 텍스트를 찾는다.
    """
    for m in reversed(_current_turn(messages)):
        if getattr(m, "type", None) == "ai":
            text = _msg_text(m.content)
            if text:
                return text
    return ""


def _parse_tool_data(raw):
    """도구 메시지 content를 dict로 파싱(가능하면). LangChain은 dict 결과를
    JSON 또는 파이썬 repr(작은따옴표) 문자열로 담으므로 둘 다 시도한다."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        for _parse in (json.loads, ast.literal_eval):
            try:
                v = _parse(raw)
                return v if isinstance(v, dict) else None
            except (ValueError, TypeError, SyntaxError):
                continue
    return None


_TREND_KO = {
    "closing": " 간격이 점점 좁혀지는 중이에요.",
    "opening": " 간격이 벌어지는 중이에요.",
    "stable": "",
}


def _driver_sentence(d: dict) -> str:
    who, team, num = d["name"], d.get("team"), d.get("number")
    head = f"{num}번은" if num else "이 선수는"
    tail = f" {team} 소속이에요." if team else ""
    return f"{head} {who} 선수예요.{tail}"


def _battle_sentence(d: dict) -> str:
    gap = d["gap_seconds"]
    drs = " DRS도 열렸어요." if d.get("drs") else ""
    return (f"앞차와 {gap}초 차이예요.{_TREND_KO.get(d.get('trend'), '')}"
            f"{drs} 두 차 사이를 화면에 표시했어요.")


def _concept_sentence(d: dict) -> str | None:
    term, explanation = d.get("term"), d.get("explanation")
    if not explanation:
        return None
    return f"{term}는 {explanation}"


def _race_status_sentence(d: dict) -> str | None:
    standings = d.get("standings") or []
    if standings:
        leader = standings[0]
        who = leader.get("driver") or f"{leader.get('driver_number')}번"
        return f"현재 1등은 {who} 선수예요."
    flag = d.get("latest_flag") or {}
    if flag.get("message"):
        return f"현재 경기 상황은 {flag['message']}입니다."
    return None


def _why_sentence(d: dict) -> str | None:
    name = d.get("driver_name") or f"{d.get('driver_number')}번"
    pits = d.get("pit_stops") or []
    stints = d.get("tire_stints") or []
    gap = d.get("recent_gap") or {}
    if pits:
        compound = stints[-1].get("compound") if stints else None
        tail = f" 지금은 {compound} 타이어를 쓰고 있어요." if compound else ""
        return f"{name} 선수는 타이어 교체를 위해 피트인했어요.{tail}"
    if gap.get("interval") is not None:
        return f"{name} 선수는 앞차와 {gap['interval']}초 차이예요."
    return None


def _salvage_from_tools(messages: list, text: str = "") -> str | None:
    """LLM이 최종 문장을 못 냈을 때, 이번 턴 도구 결과로 최소한의 답을 복구한다.

    모델이 한 턴에 여러 도구를 부를 수 있으므로(예: 갭 질문에 show_battle_context와
    get_driver_info를 둘 다 호출) '질문 의도'에 맞는 결과를 우선 고른다.
    """
    turn = _current_turn(messages)   # 이번 턴 도구 결과만(이전 턴 오염 방지)

    dicts: list[dict] = []
    strings: list[str] = []
    for m in reversed(turn):         # 최근 것부터
        if getattr(m, "type", None) != "tool":
            continue
        data = _parse_tool_data(m.content)
        if isinstance(data, dict):
            dicts.append(data)
        elif isinstance(m.content, str) and m.content.strip():
            strings.append(m.content.strip())

    battle = next((d for d in dicts if d.get("gap_seconds") is not None), None)
    driver = next((d for d in dicts if d.get("name")), None)
    concept = next((d for d in dicts if d.get("term") and "explanation" in d), None)
    race_status = next((d for d in dicts if "standings" in d or "latest_flag" in d), None)
    why = next((d for d in dicts if "pit_stops" in d or "tire_stints" in d or "recent_gap" in d), None)

    want_battle = any(k in text for k in ("갭", "간격", "차이", "붙", "앞차", "배틀", "추격", "따라"))
    want_name = any(k in text for k in ("누구", "이름"))
    want_status = any(k in text for k in ("몇 등", "몇등", "1등", "일등", "순위", "상황"))
    want_why = "왜" in text
    want_concept = any(k in text for k in ("뭐야", "무엇", "뜻", "설명"))

    # 1) 질문 의도에 맞는 결과 우선
    if want_battle and battle:
        return _battle_sentence(battle)
    if want_name and driver:
        return _driver_sentence(driver)
    if want_why and why:
        return _why_sentence(why)
    if want_status and race_status:
        return _race_status_sentence(race_status)
    if want_concept and concept:
        return _concept_sentence(concept)
    # 2) 의도가 모호하면 이름 → 배틀 순
    if driver:
        return _driver_sentence(driver)
    if battle:
        return _battle_sentence(battle)
    if why:
        return _why_sentence(why)
    if race_status:
        return _race_status_sentence(race_status)
    if concept:
        return _concept_sentence(concept)
    # 3) 명령형 도구의 확인 문구(문자열)
    if strings:
        return strings[0]
    return None


async def _rule_based_demo_route(text: str) -> tuple[str, list[dict], bool] | None:
    """데모 핵심 명령은 LLM 전에 안정적으로 처리한다.

    Gemma q4가 짧은 명령을 빈 답으로 끝내거나 엉뚱한 조회 도구를 고르는 경우를 막기 위한
    얇은 안전장치다. 데이터 조회가 필요한 지목/배틀/추월 장면은 기존 도구를 그대로 호출한다.
    """
    t = text.replace(" ", "")
    selected = current_selected()

    # 지목 grounding: "이 선수/이 차/쟤 누구야?"
    if selected and any(k in t for k in ("이선수누구", "이차누구", "쟤누구", "얘누구")):
        start_capture()
        data = await get_driver_info.ainvoke({"driver_number": selected})
        emit_command("highlightDriver", driver_number=selected)
        reply = _driver_sentence(data) if isinstance(data, dict) and data.get("name") else \
            f"{selected}번 선수 정보를 찾지 못했어요."
        return reply, drain(), bool(data.get("name") if isinstance(data, dict) else False)

    # 공간 배틀 배지: 선택 차량 기준으로 앞차와의 간격 표시.
    if selected and any(k in t for k in ("앞차", "얼마나붙", "배틀상황", "간격", "갭")):
        start_capture()
        data = await show_battle_context.ainvoke({"driver_number": selected})
        if isinstance(data, dict) and data.get("shown"):
            return _battle_sentence(data), drain(), True
        return (data.get("note") if isinstance(data, dict) else "배틀 상황을 찾지 못했어요."), drain(), False

    # 복합 추월 장면: seek → slow → optional highlight 순서 보장.
    if "추월" in t and "장면" in t and any(k in t for k in ("가줘", "보여", "돌아", "직전")):
        start_capture()
        event_type = "first_overtake_before" if any(k in t for k in ("직전", "돌아")) else "first_overtake"
        msg = await jump_to_event.ainvoke({"event_type": event_type})
        if "천천히" in t or "느리게" in t:
            emit_command("controlReplay", action="speed", value=0.5)
        if selected:
            emit_command("highlightDriver", driver_number=selected)
        commands = drain()
        ok = bool(commands)
        if not ok:
            return msg, commands, ok
        if event_type.endswith("_before"):
            reply = "추월 장면 직전으로 이동해서 천천히 보여드릴게요."
        else:
            reply = "첫 추월 장면으로 이동할게요."
        return reply, normalize_command_order(commands), ok

    # 기본 리플레이 제어.
    if any(k in t for k in ("멈춰", "정지", "일시정지")):
        start_capture()
        emit_command("controlReplay", action="pause", value=None)
        return "네, 화면을 멈췄어요.", drain(), True
    if any(k in t for k in ("다시재생", "재생해", "플레이")):
        start_capture()
        emit_command("controlReplay", action="play", value=None)
        return "리플레이를 다시 재생할게요.", drain(), True
    if any(k in t for k in ("천천히", "느리게", "슬로우")):
        start_capture()
        emit_command("controlReplay", action="speed", value=0.5)
        return "리플레이 속도를 0.5배로 늦춰서 보여드릴게요.", drain(), True

    # 드론 시점.
    if "드론" in t or "공중" in t:
        start_capture()
        on = not any(k in t for k in ("꺼", "원래", "돌아"))
        emit_command("droneView", on=on)
        return ("드론 시점으로 전환했어요." if on else "원래 시점으로 돌아왔어요."), drain(), True

    return None


async def run_agent(
    text: str,
    session_key: int | None = None,
    at_time: str | None = None,
    history: list | None = None,
    selected_driver: int | None = None,
) -> tuple[str, list[dict]]:
    """한 번의 사용자 발화를 처리한다.

    Args:
        selected_driver: 사용자가 XR Ray/클릭으로 지목한 차량 번호("이 선수"의 대상).

    Returns:
        reply: 사용자에게 보낼 한국어 텍스트(이후 TTS로 음성화)
        commands: Unity로 보낼 명령 리스트(마커·리플레이·점프)
    """
    set_context(session_key, at_time, selected_driver)   # 이번 요청의 세션/시각/선택대상 고정
    # SYSTEM_PROMPT(고정 지침) + 현재 관람 맥락(동적)을 하나의 system 메시지로 합쳐 주입.
    system = SYSTEM_PROMPT
    ctx = _context_message()
    if ctx:
        system = f"{SYSTEM_PROMPT}\n\n{ctx}"
    messages: list = [("system", system)] + list(history or []) + [("user", text)]
    try:
        if settings.command_planner_enabled:
            plan = build_command_plan(text, current_selected())
            if plan:
                return await execute_command_plan(plan)

        if settings.demo_rule_router_enabled:
            routed = await _rule_based_demo_route(text)
            if routed is not None:
                return routed

        # 로컬(양자화) 모델이 도구콜 JSON을 깨뜨려 500이 나는 경우가 잦다(비결정적).
        # 실패하면 명령 버퍼를 비우고 몇 번 더 시도 → 대개 성공하는 시도가 나온다.
        attempts = 1 + max(0, settings.tool_error_retries)
        result = None
        last_exc: Exception | None = None
        for attempt in range(attempts):
            start_capture()   # 시도마다 명령 버퍼 초기화(부분 실패 명령 누적 방지)
            try:
                result = await get_agent().ainvoke({"messages": messages})
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.warning("[재시도] LLM 호출 실패(%d/%d): %s",
                               attempt + 1, attempts, str(e)[:140])
        if last_exc is not None:
            raise last_exc

        # [임시 진단] 모델이 어떤 도구를 부르고 무엇을 반환했는지, 최종 텍스트가 비었는지 확인용.
        # 원인 파악 후 제거 예정.
        for _m in result["messages"]:
            _t = getattr(_m, "type", "?")
            _tc = getattr(_m, "tool_calls", None)
            _names = [c.get("name") for c in _tc] if _tc else None
            _prev = (_msg_text(_m.content) if _t != "tool" else str(_m.content))[:140]
            logger.warning("[chain] %-5s tool_calls=%s | %s", _t, _names, _prev)

        reply = _extract_reply(result["messages"])

        # 1차 방어(재요청): 최종 텍스트가 비면 한 번 더 시도한다.
        # (빈 답일 때만 1회 — 정상 답이면 추가 호출 없음.)
        # 문구는 '도구가 필요하면 부르고, 결과로 반드시 답하라'로 — 모델이 도구를 아예
        # 안 부르고 비운 경우(tool_calls=None)에도 도구 호출을 유도할 수 있게 한다.
        if not reply and settings.empty_reply_retry:
            # history를 뺀 '깨끗한 문맥'으로 현재 질문만 다시 던진다. 반복 질문에서 모델이
            # 이전 답을 보고 아무것도 안 하거나(whiff), 오염된 문맥으로 비우는 걸 막고
            # 새 샘플링으로 다시 시도하게 한다.
            clean_msgs = [
                ("system", system),
                ("user", text),
                ("user", "위 질문에 답해줘. 필요하면 도구를 호출하고, 한국어 1~2문장으로 반드시 답해."),
            ]
            # 버퍼는 초기화하지 않는다 — 1차 시도의 Unity 명령(gap line 등)을 보존.
            # 재시도가 같은 도구를 또 불러도 핸들러가 idempotent라 무해.
            for _ in range(1 + max(0, settings.tool_error_retries)):
                try:
                    retry = await get_agent().ainvoke({"messages": clean_msgs})
                    reply = _extract_reply(retry["messages"])
                    logger.warning("[재요청] 결과: %s",
                                   f"복구됨: {reply!r}" if reply else "재요청도 빈 답")
                    if reply:
                        result = retry   # salvage도 재시도의 도구 결과를 보도록 교체
                        break
                except Exception as e:
                    logger.warning("[재요청] 실패(재시도): %s", str(e)[:120])

        # 2차 방어(복구): 재요청도 비면 도구 결과로 최소 답 복구, 그래도 없으면 안내.
        # ok = 이 답을 대화 history에 남겨도 되는지. generic 폴백은 남기면 안 된다
        # (이전 폴백이 history에 쌓여 모델을 오염시키는 악순환 방지).
        ok = bool(reply)
        if not reply:
            salvaged = _salvage_from_tools(result["messages"], text)
            reply = salvaged or "화면에 표시했어요. 더 궁금한 점 있으신가요?"
            ok = salvaged is not None   # 이름/배틀/명령 복구는 저장 OK, generic은 제외
            logger.warning("빈 답 폴백: %r (history저장=%s)", reply, ok)
        commands = normalize_command_order(drain())  # 도구가 쌓아둔 Unity 명령 회수
        return reply, commands, ok
    except Exception as exc:
        # LLM·도구·데이터서버 오류 시 대화를 끊지 않고 우아하게 실패한다.
        # (에러는 로그로 남기고, 사용자에겐 짧은 안내만.)
        logger.exception("run_agent 처리 중 오류")
        drain()                         # 남은 명령 버퍼 비우기
        # 인증/키 오류만 '설정 오류'로 명확히 안내한다. (도구콜 JSON 파싱 실패 같은
        # 일시적 500은 여기에 걸리면 안 됨 → 아래 우아한 재시도 안내로 빠진다.)
        m = str(exc).lower()
        if any(k in m for k in ("api_key", "api key", "authentication",
                                "credential", "unauthorized", "invalid_api_key")):
            return (
                f"⚠️ 설정 오류로 보여요: {exc}\n"
                "→ .env 의 OPENAI_API_KEY / LLM_MODEL 을 확인하세요. "
                "(모델 목록: python -m scripts.list_models)"
            ), [], False
        return "죄송해요, 잠시 문제가 생겼어요. 다시 한 번 말씀해 주세요.", [], False
