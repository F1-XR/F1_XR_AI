"""명령 싱크(Command Sink).

명령형 도구(highlightDriver 등)는 LLM에게 '데이터를 돌려주는' 게 아니라
Unity로 '명령을 보내야' 한다. 하지만 도구 함수는 WebSocket을 직접 모른다.
그래서 요청 1건 동안 도구가 명령을 여기 쌓아두고, WS 핸들러가 나중에 꺼내
Unity로 전송한다. (요청별 격리를 위해 contextvar 사용)
"""
from __future__ import annotations

import contextvars

_commands: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "commands", default=None
)


def start_capture() -> None:
    """요청 시작 시 호출 — 이 요청의 명령 버퍼를 연다."""
    _commands.set([])


def emit_command(name: str, **args) -> None:
    """명령형 도구가 호출 — Unity로 보낼 명령을 버퍼에 쌓는다."""
    buf = _commands.get()
    if buf is not None:
        buf.append({"type": "command", "name": name, "args": args})


def drain() -> list[dict]:
    """요청 끝에서 호출 — 쌓인 명령을 반환하고 버퍼를 비운다."""
    buf = _commands.get() or []
    _commands.set([])
    return buf
