"""run_agent 단위테스트 — 정상 응답 + 오류 시 우아한 실패(대화 안 끊김).

LLM(에이전트)을 가짜로 바꿔치기해 네트워크·키 없이 검사한다.
"""
import app.agent.graph as graph
from app.agent.commands import start_capture, emit_command


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeAgent:
    def __init__(self, reply="테스트 응답", boom=False, emit=None):
        self._reply = reply
        self._boom = boom
        self._emit = emit

    async def ainvoke(self, payload):
        if self._emit:
            # 도구가 명령을 쌓는 상황 흉내
            emit_command(*self._emit)
        if self._boom:
            raise RuntimeError("LLM down")
        return {"messages": [_Msg(self._reply)]}


async def test_run_agent_happy(monkeypatch):
    monkeypatch.setattr(graph, "get_agent", lambda: _FakeAgent(reply="안녕하세요"))
    reply, commands = await graph.run_agent("안녕", session_key=9523)
    assert reply == "안녕하세요"
    assert commands == []


async def test_run_agent_error_is_graceful(monkeypatch):
    # LLM이 예외를 던져도 run_agent는 던지지 않고 안내 문구 + 빈 명령을 반환해야 한다.
    monkeypatch.setattr(graph, "get_agent", lambda: _FakeAgent(boom=True))
    reply, commands = await graph.run_agent("왜 피트인?", session_key=9523)
    assert "죄송" in reply
    assert commands == []


async def test_run_agent_error_discards_partial_commands(monkeypatch):
    # 도구가 명령을 쌓은 뒤 실패하면, 반쪽 명령은 보내지 않는다(빈 리스트).
    monkeypatch.setattr(
        graph, "get_agent",
        lambda: _FakeAgent(boom=True, emit=("highlightDriver",)),
    )
    reply, commands = await graph.run_agent("해밀턴 강조", session_key=9523)
    assert "죄송" in reply
    assert commands == []
