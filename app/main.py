"""FastAPI 앱 — 헬스체크 + WebSocket 엔드포인트.

Unity가 /ws로 붙어 발화를 보내면, 에이전트를 돌려
① Unity 명령(마커·리플레이)을 먼저 보내고 ② 텍스트 응답을 보낸다.
(Day14부터 ②를 TTS 오디오로 대체/추가)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .agent.graph import run_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="F1 Tutor Agent")

# 대화 history 상한(한 턴 = user+assistant 2개). 토큰·비용·지연 관리.
MAX_HISTORY_TURNS = 8


@app.get("/health")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    history: list = []
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") != "utterance":
                continue
            text = msg.get("text")
            if not text:
                continue

            # 한 발화 처리 중 오류가 나도 연결·대화는 유지한다(우아한 실패).
            try:
                reply, commands = await run_agent(
                    text=text,
                    session_key=msg.get("session_key"),
                    at_time=msg.get("at_time"),
                    history=history,
                )
                history += [("user", text), ("assistant", reply)]
                del history[: -MAX_HISTORY_TURNS * 2]   # 최근 N턴만 유지

                for cmd in commands:               # ① Unity 명령 먼저
                    await websocket.send_json(cmd)
                await websocket.send_json(          # ② 텍스트 응답
                    {"type": "assistant_text", "text": reply}
                )
            except WebSocketDisconnect:
                raise                              # 연결 끊김은 바깥에서 처리
            except Exception:
                logger.exception("발화 처리 실패")
                try:
                    await websocket.send_json({
                        "type": "assistant_text",
                        "text": "죄송해요, 잠시 문제가 있었어요. 다시 말씀해 주세요.",
                    })
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
