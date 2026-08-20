"""Deterministic command planner for compound replay instructions.

The LLM still handles open-ended F1 Q&A. This planner handles operational
commands that can be safely decomposed into a small action plan, then executes
the same tool/command path the LLM would use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .commands import drain, emit_command, start_capture
from .context import current_selected, current_time
from .tools import get_driver_info, jump_to_event, recommend_battle_action, show_battle_context


@dataclass(frozen=True)
class PlanStep:
    kind: str
    args: dict[str, Any]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _driver_from_text(text: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*번", text)
    return int(m.group(1)) if m else None


def _mentions_selected(text: str) -> bool:
    t = _compact(text)
    return any(k in t for k in ("이선수", "이차", "얘", "쟤", "그선수", "그차", "선택한"))


def _relative_seek_seconds(text: str) -> float | None:
    """Parse natural Korean rewind expressions into seconds before current time."""
    t = _compact(text)

    m = re.search(r"(\d{1,3})\s*초\s*(전부터|전으로|전|뒤로|되감)", text)
    if m:
        return float(m.group(1))

    m = re.search(r"(\d{1,2})\s*분\s*(전부터|전으로|전|뒤로|되감)", text)
    if m:
        return float(m.group(1)) * 60.0

    if "다시재생" in t or "재생해" in t:
        return None

    scene_words = ("방금장면", "방금", "조금전", "아까장면", "아까")
    replay_words = ("다시", "보여", "돌아", "가줘", "되감")
    if any(k in t for k in scene_words) and any(k in t for k in replay_words):
        return 5.0

    if any(k in t for k in ("되감아", "되돌려", "뒤로가", "뒤로돌려")):
        return 5.0

    return None


def _iso_minus_seconds(iso: str, seconds: float) -> str | None:
    try:
        normalized = iso.replace("Z", "+00:00")
        return (datetime.fromisoformat(normalized) - timedelta(seconds=seconds)).isoformat()
    except ValueError:
        return None


def build_command_plan(text: str, selected_driver: int | None = None) -> list[PlanStep]:
    """Build a command plan for clear replay/control requests.

    Returns an empty list when the utterance is better left to the LLM agent.
    """
    t = _compact(text)
    driver = _driver_from_text(text) or selected_driver
    action_words = ("공격", "압박", "시도", "해야", "해도돼", "들어가", "붙어", "밀어붙", "추월할까")
    if driver and any(k in t for k in action_words) and any(k in t for k in ("추월", "공격", "압박", "앞차")):
        return [PlanStep("battle_action", {"driver_number": driver})]

    # 설명·질문형(상황/전략/왜/추월 가능성 등)은 결정적 라우터가 가로채지 말고 LLM 도구에 맡긴다.
    # (explain_situation·explain_why·predict_overtake가 처리해야 하는데 highlight로 새는 것 방지)
    _ASK = ("전략", "상황", "왜", "설명", "어때", "추월할", "추월가능", "추월확률", "추월할까", "가능성", "무슨전략")
    _CMD = ("장면", "돌아", "직전", "천천히", "느리게", "멈춰", "정지", "드론", "강조")
    if any(k in t for k in _ASK) and not any(k in t for k in _CMD):
        return []
    steps: list[PlanStep] = []

    wants_visual_driver = bool(driver and (_mentions_selected(text) or f"{driver}번" in text))

    if driver and any(k in t for k in ("누구", "이름")) and _mentions_selected(text):
        steps.append(PlanStep("driver_info", {"driver_number": driver}))
        steps.append(PlanStep("highlight", {"driver_number": driver}))
        return steps

    if driver and any(k in t for k in ("앞차", "얼마나붙", "배틀상황", "간격", "갭")):
        steps.append(PlanStep("battle_context", {"driver_number": driver}))
        return steps

    if any(k in t for k in ("피트스톱", "피트인")) and any(k in t for k in ("장면", "보여", "가줘", "돌아")):
        steps.append(PlanStep("seek_event", {"event_type": "first_pit"}))

    if "추월" in t and any(k in t for k in ("장면", "보여", "가줘", "돌아", "직전")):
        event_type = "first_overtake_before" if any(k in t for k in ("직전", "전으로", "돌아")) else "first_overtake"
        steps.append(PlanStep("seek_event", {"event_type": event_type}))

    if any(k in t for k in ("세이프티카", "사고장면")) and any(k in t for k in ("장면", "보여", "가줘", "돌아")):
        steps.append(PlanStep("seek_event", {"event_type": "safety_car"}))

    if any(k in t for k in ("옐로", "노란깃발", "황색기")) and any(k in t for k in ("장면", "보여", "가줘", "돌아")):
        steps.append(PlanStep("seek_event", {"event_type": "yellow_flag"}))

    if not any(step.kind == "seek_event" for step in steps):
        relative_seconds = _relative_seek_seconds(text)
        if relative_seconds is not None:
            steps.append(PlanStep("seek_relative", {"seconds": relative_seconds}))

    if any(k in t for k in ("멈춰", "정지", "일시정지")):
        steps.append(PlanStep("replay", {"action": "pause", "value": None}))

    if any(k in t for k in ("다시재생", "재생해", "플레이")):
        steps.append(PlanStep("replay", {"action": "play", "value": None}))

    if any(k in t for k in ("천천히", "느리게", "슬로우")):
        steps.append(PlanStep("replay", {"action": "speed", "value": 0.5}))
    elif any(k in t for k in ("빠르게", "빨리", "두배속", "2배속")):
        steps.append(PlanStep("replay", {"action": "speed", "value": 2.0}))

    if driver and (("강조" in t) or ("보여" in t) or (steps and any(k in t for k in ("보여", "따라", "장면")))):
        steps.append(PlanStep("highlight", {"driver_number": driver}))

    if "드론" in t or "공중" in t or any(k in t for k in ("원래시점", "기본시점")):
        on = not any(k in t for k in ("꺼", "원래", "돌아"))
        steps.append(PlanStep("drone", {"on": on}))

    # A single bare speed/pause/play/drone command is also safe to execute here.
    return _dedupe_plan(steps)


def _dedupe_plan(steps: list[PlanStep]) -> list[PlanStep]:
    seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
    out: list[PlanStep] = []
    for step in steps:
        key = (step.kind, tuple(sorted(step.args.items())))
        if key in seen:
            continue
        seen.add(key)
        out.append(step)
    return out


def normalize_command_order(commands: list[dict]) -> list[dict]:
    """Send compound replay commands to Unity in a stable operational order."""
    def priority(cmd: dict) -> int:
        name = cmd.get("name")
        args = cmd.get("args") or {}
        if name == "controlReplay":
            action = args.get("action")
            if action == "seek":
                return 0
            if action == "speed":
                return 1
            if action == "pause":
                return 2
            if action == "play":
                return 3
            return 10
        if name == "highlightDriver":
            return 20
        if name == "showBattleContext":
            return 20
        return 30

    return sorted(commands, key=priority)


async def execute_command_plan(steps: list[PlanStep]) -> tuple[str, list[dict], bool]:
    """Execute a plan and return the assistant reply, Unity commands, and history flag."""
    if not steps:
        return "", [], False

    start_capture()
    reply_parts: list[str] = []
    ok = True

    for step in steps:
        if step.kind == "driver_info":
            data = await get_driver_info.ainvoke(step.args)
            if isinstance(data, dict) and data.get("name"):
                team = f"{data.get('team')} 팀의 " if data.get("team") else ""
                reply_parts.append(f"이 선수는 {team}{data['name']} 선수예요.")
            else:
                ok = False
                reply_parts.append("선수 정보를 찾지 못했어요.")
        elif step.kind == "battle_context":
            data = await show_battle_context.ainvoke(step.args)
            if isinstance(data, dict) and data.get("shown"):
                gap = data.get("gap_seconds")
                trend = {"closing": "간격이 좁혀지는 중이에요.", "opening": "간격이 벌어지는 중이에요."}.get(data.get("trend"), "")
                drs = " DRS도 열렸어요." if data.get("drs") else ""
                # 3초 뒤 예측 갭을 음성에도 (의미 있게 변할 때만, 예측임을 명시)
                pred = data.get("predicted_gap_seconds")
                hz = int(data.get("predict_horizon_sec") or 3)
                fc = ""
                if pred is not None and gap is not None:
                    if gap - pred >= 0.1:
                        fc = f" {hz}초 뒤엔 {pred}초로 좁혀질 것 같아요."
                    elif gap - pred <= -0.1:
                        fc = f" {hz}초 뒤엔 {pred}초로 벌어질 것 같아요."
                reply_parts.append(f"앞차와 {gap}초 차이예요. {trend}{fc}{drs}".strip())
            else:
                ok = False
                reply_parts.append((data or {}).get("note", "배틀 상황을 찾지 못했어요."))
        elif step.kind == "battle_action":
            data = await recommend_battle_action.ainvoke(step.args)
            if isinstance(data, dict) and data.get("available"):
                reply_parts.append(_battle_action_reply(data))
            else:
                ok = False
                reply_parts.append((data or {}).get("reason", "행동 추천을 계산하지 못했어요."))
        elif step.kind == "seek_event":
            msg = await jump_to_event.ainvoke(step.args)
            if "찾지 못했어요" in msg:
                ok = False
                reply_parts.append(msg)
            elif step.args.get("event_type", "").endswith("_before"):
                reply_parts.append("장면 직전으로 이동할게요.")
            else:
                reply_parts.append("장면으로 이동할게요.")
        elif step.kind == "seek_relative":
            now = current_time()
            target = _iso_minus_seconds(now, step.args["seconds"]) if now else None
            if target:
                emit_command("controlReplay", action="seek", value=target)
                seconds = step.args["seconds"]
                label = f"{int(seconds)}초" if seconds < 60 else f"{int(seconds // 60)}분"
                reply_parts.append(f"{label} 전부터 보여드릴게요.")
            else:
                ok = False
                reply_parts.append("현재 재생 시각을 몰라서 되감기 위치를 계산하지 못했어요.")
        elif step.kind == "replay":
            emit_command("controlReplay", **step.args)
            action, value = step.args.get("action"), step.args.get("value")
            if action == "pause":
                reply_parts.append("화면을 멈췄어요.")
            elif action == "play":
                reply_parts.append("다시 재생할게요.")
            elif action == "speed" and value == 0.5:
                reply_parts.append("0.5배속으로 천천히 보여드릴게요.")
            elif action == "speed":
                reply_parts.append(f"{value}배속으로 보여드릴게요.")
        elif step.kind == "highlight":
            emit_command("highlightDriver", **step.args)
            reply_parts.append(f"{step.args['driver_number']}번 선수를 강조했어요.")
        elif step.kind == "drone":
            emit_command("droneView", **step.args)
            reply_parts.append("드론 시점으로 전환했어요." if step.args.get("on") else "원래 시점으로 돌아왔어요.")

    commands = normalize_command_order(drain())
    return _merge_reply(reply_parts), commands, ok and bool(commands)


def _battle_action_reply(data: dict[str, Any]) -> str:
    label = {
        "PRESS_ATTACK": "지금은 공격 압박을 걸 만해요.",
        "WAIT_FOR_DRS": "바로 공격보다 DRS 구간을 기다리는 게 좋아요.",
        "HOLD_POSITION": "지금은 무리하지 말고 위치를 지키는 편이 좋아요.",
        "HOLD_PRESSURE": "공격보다는 압박을 유지하는 흐름이 좋아요.",
        "LOW_CONFIDENCE": "지금은 판단이 불확실해서 보수적으로 보는 게 좋아요.",
        "NO_TARGET": "앞차 배틀 상대를 찾지 못했어요.",
    }.get(data.get("action"), "상황을 보고 보수적으로 판단하는 게 좋아요.")
    reason = data.get("reason")
    return f"{label} {reason}" if reason else label


def _merge_reply(parts: list[str]) -> str:
    compact: list[str] = []
    for part in parts:
        part = part.strip()
        if part and part not in compact:
            compact.append(part)
    if not compact:
        return "처리했어요."
    # Keep voice output short while preserving the action summary.
    return " ".join(compact[:3])
