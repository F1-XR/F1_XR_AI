"""FastAPI 앱 — 헬스체크 + WebSocket 엔드포인트.

Unity가 /ws로 붙어 발화(텍스트 또는 음성)를 보내면:
  ① (음성이면) STT로 텍스트 변환 → transcript 회신
  ② 에이전트 실행
  ③ Unity 명령(마커·리플레이) 전송
  ④ 응답 자막(assistant_text) 전송
  ⑤ 응답 음성(tts_audio, base64 wav) 전송  ← TTS_ENABLED일 때
"""
from __future__ import annotations

import base64
import binascii
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .agent.graph import run_agent
from .config import settings
from .voice import stt, tts

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 부팅 시 STT/TTS 모델을 미리 로드(워밍업)한다.

    기본은 첫 요청 때 모델을 로드하는데(지연 로드), 그러면 '첫 사용자'만 크게 느리다.
    부팅 때 더미 합성/인식을 1회 돌려 모델을 메모리에 올려두면 첫 실요청이 빠르다.
    (부팅이 그만큼 늦어지지만, 준비되면 매 요청이 빨라진다.)
    """
    if settings.warmup_on_start:
        try:
            logger.info("워밍업: STT/TTS 모델 미리 로드 중…")
            if settings.tts_enabled:
                audio = await tts.synthesize("안녕하세요.")
                await stt.transcribe(audio, language="ko")  # TTS 결과로 STT까지 예열
            logger.info("워밍업 완료 — 이제 요청이 빠릅니다.")
        except Exception:
            logger.exception("워밍업 실패(무시하고 계속 — 첫 요청에서 로드됨)")
    yield


app = FastAPI(title="F1 Tutor Agent", lifespan=lifespan)

# 대화 history 상한(한 턴 = user+assistant 2개). 토큰·비용·지연 관리.
MAX_HISTORY_TURNS = 8


@app.get("/health")
def health():
    return {"status": "ok"}


async def _transcribe_safe(data_b64: str) -> str | None:
    """base64 wav → 텍스트. 실패하면 None(호출부에서 무시)."""
    try:
        audio = base64.b64decode(data_b64)
    except (binascii.Error, ValueError):
        logger.warning("오디오 base64 디코드 실패")
        return None
    try:
        return await stt.transcribe(audio, language="ko")
    except Exception:
        logger.exception("STT 인식 실패")
        return None


import re

_SINO_ONES = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]


def _sino(n: int) -> str:
    """정수(1~99) → 한자어 한글. 예: 7→'칠', 44→'사십사', 16→'십육'."""
    if n <= 0 or n > 99:
        return str(n)
    if n < 10:
        return _SINO_ONES[n]
    tens, ones = divmod(n, 10)
    s = ("" if tens == 1 else _SINO_ONES[tens]) + "십"
    return s + (_SINO_ONES[ones] if ones else "")


def _normalize_ko_numbers(text: str) -> str:
    """'N번/N등/N랩'의 숫자를 한자어 한글로 → TTS가 '칠 번'으로 읽게(‘일곱 번’ 방지).
    드라이버 번호·순위·랩은 한국어에서 한자어로 읽는다."""
    return re.sub(r"(\d+)\s*(번|등|랩)", lambda m: _sino(int(m.group(1))) + m.group(2), text)


async def _synthesize_safe(text: str) -> str | None:
    """텍스트 → base64 wav. TTS가 꺼져있거나 실패하면 None(텍스트만 전송)."""
    if not settings.tts_enabled:
        return None
    try:
        audio = await tts.synthesize(_normalize_ko_numbers(text))
        return base64.b64encode(audio).decode("ascii")
    except Exception:
        logger.exception("TTS 합성 실패 — 텍스트만 전송")
        return None


async def _handle_utterance(websocket: WebSocket, text: str, msg: dict, history: list) -> None:
    """텍스트 한 건을 에이전트에 넘기고 명령·자막·음성을 순서대로 전송."""
    reply, commands = await run_agent(
        text=text,
        session_key=msg.get("session_key"),
        at_time=msg.get("at_time"),
        history=history,
    )
    history += [("user", text), ("assistant", reply)]
    del history[: -MAX_HISTORY_TURNS * 2]   # 최근 N턴만 유지

    for cmd in commands:                    # ③ Unity 명령 먼저
        await websocket.send_json(cmd)
    await websocket.send_json(               # ④ 자막용 텍스트
        {"type": "assistant_text", "text": reply}
    )
    audio_b64 = await _synthesize_safe(reply)  # ⑤ TTS 오디오(가능하면)
    if audio_b64 is not None:
        await websocket.send_json(
            {"type": "tts_audio", "format": "wav", "data": audio_b64}
        )


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    history: list = []
    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            # 능동 안내(pointOut): 짧은 문장을 '음성만' 빠르게 합성해 돌려준다.
            # 에이전트(LLM)를 안 거치므로 시간에 민감한 "곧 추월!" 안내에 적합.
            if mtype == "speak":
                say = msg.get("text")
                if say:
                    audio_b64 = await _synthesize_safe(say)
                    if audio_b64 is not None:
                        # 능동 안내 전용 타입 → Unity가 "답변 재생 중이면 건너뛰기"로 처리
                        await websocket.send_json(
                            {"type": "tts_announce", "format": "wav", "data": audio_b64}
                        )
                continue

            # 입력 정규화: 텍스트/음성 어느 쪽이 와도 text 한 줄로 만든다.
            if mtype == "utterance":
                text = msg.get("text")
            elif mtype == "audio_utterance":
                text = await _transcribe_safe(msg.get("data", ""))
                if text:                     # 무엇으로 인식됐는지 Unity에 회신(자막 확인)
                    await websocket.send_json({"type": "transcript", "text": text})
            else:
                continue

            if not text:
                continue

            # 한 발화 처리 중 오류가 나도 연결·대화는 유지한다(우아한 실패).
            try:
                await _handle_utterance(websocket, text, msg, history)
            except WebSocketDisconnect:
                raise                        # 연결 끊김은 바깥에서 처리
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
