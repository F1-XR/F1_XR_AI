"""FastAPI 앱 — 헬스체크 + WebSocket 엔드포인트.

Unity가 /ws로 붙어 발화를 보내면, 에이전트를 돌려
① Unity 명령(마커·리플레이)을 먼저 보내고 ② 텍스트 응답을 보낸다.
(Day14부터 ②를 TTS 오디오로 대체/추가)
"""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .agent.graph import run_agent

app = FastAPI(title="F1 Tutor Agent")


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

            reply, commands = await run_agent(
                text=msg["text"],
                session_key=msg.get("session_key"),
                at_time=msg.get("at_time"),
                history=history,
            )
            history += [("user", msg["text"]), ("assistant", reply)]

            for cmd in commands:               # ① Unity 명령 먼저
                await websocket.send_json(cmd)
            await websocket.send_json(          # ② 텍스트 응답
                {"type": "assistant_text", "text": reply}
            )
    except WebSocketDisconnect:
        pass
