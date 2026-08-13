"""FastAPI 앱 — 헬스체크 + WebSocket 엔드포인트.

Unity가 /ws로 붙어 발화(텍스트 또는 음성)를 보내면:
  ① (음성이면) STT로 텍스트 변환 → transcript 회신
  ② 에이전트 실행
  ③ Unity 명령(마커·리플레이) 전송
  ④ 응답 자막(assistant_text) 전송
  ⑤ 응답 음성(tts_audio, base64 wav) 전송  ← TTS_ENABLED일 때
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .agent.graph import run_agent
from .agent.context import set_recent_overtake
from .agent.watcher import watch
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


# TTS 결과(base64 wav) 캐시 — 같은 문장은 다시 합성하지 않는다.
# 능동 안내("N번, 곧 추월할 것 같아요!")는 문구 패턴이 정해져 있어, 처음 한 번만 합성하면
# 이후 같은 안내는 즉시 재생된다(음성 지연 체감 감소). 최근 N개만 유지(메모리 상한).
from collections import OrderedDict

_TTS_CACHE: "OrderedDict[str, str]" = OrderedDict()
_TTS_CACHE_MAX = 64


async def _synthesize_safe(text: str) -> str | None:
    """텍스트 → base64 wav. TTS가 꺼져있거나 실패하면 None(텍스트만 전송). 같은 문장은 캐시."""
    if not settings.tts_enabled:
        return None
    norm = _normalize_ko_numbers(text)   # '7번'→'칠 번' 정규화된 최종 문장이 캐시 키
    cached = _TTS_CACHE.get(norm)
    if cached is not None:
        _TTS_CACHE.move_to_end(norm)     # LRU: 최근 사용으로 갱신
        return cached
    try:
        audio = await tts.synthesize(norm)
        b64 = base64.b64encode(audio).decode("ascii")
    except Exception:
        logger.exception("TTS 합성 실패 — 텍스트만 전송")
        return None
    _TTS_CACHE[norm] = b64
    _TTS_CACHE.move_to_end(norm)
    if len(_TTS_CACHE) > _TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)   # 가장 오래된 것 제거
    return b64


async def _handle_utterance(send, text: str, msg: dict, history: list) -> int | None:
    """텍스트 한 건을 에이전트에 넘기고 명령·자막·음성을 순서대로 전송.

    send: 이 연결의 (Lock으로 직렬화된) 송신 함수. 직접 websocket.send_json을 쓰지 않는다.
    Returns: 음성으로 경기를 바꿨으면(find_session→loadSession) 그 session_key, 아니면 None.
             호출부(ws)가 이 값을 연결 상태로 기억해 이후 발화에도 유지한다.
    """
    # 공간 맥락: 사용자가 지목(클릭·XR Ray)한 차량 번호 → "이 선수" 해석용.
    ictx = msg.get("interaction_context") or {}
    selected_driver = ictx.get("driver_number")
    # [진단] 선택 드라이버가 실제로 왔는지 서버 로그로 바로 확인.
    #   None 이면 → Unity에서 차 선택이 안 실려온 것(interaction_context 없음).
    #   숫자면 → grounding 데이터는 정상, 이후는 모델 문제.
    logger.warning("[수신] text=%r | selected_driver=%s | session=%s",
                   text, selected_driver, msg.get("session_key"))

    reply, commands, ok = await run_agent(
        text=text,
        session_key=msg.get("session_key"),
        at_time=msg.get("at_time"),
        history=history,
        selected_driver=selected_driver,
    )
    # 실제 답(ok=True)만 history에 남긴다. generic 폴백("화면에 표시했어요")을 저장하면
    # 다음 턴 모델이 그걸 보고 따라 하며 빈 답을 반복하는 오염이 생긴다 → 그 턴은 통째로 버린다.
    if ok:
        history += [("user", text), ("assistant", reply)]
        del history[: -MAX_HISTORY_TURNS * 2]   # 최근 N턴만 유지

    switched_session: int | None = None
    for cmd in commands:                    # ③ Unity 명령 먼저
        await send(cmd)
        # find_session이 경기를 바꾸면 loadSession 명령에 새 세션이 실린다 → 기억해 둔다.
        if cmd.get("name") == "loadSession":
            switched_session = (cmd.get("args") or {}).get("session_key", switched_session)
    await send({"type": "assistant_text", "text": reply})   # ④ 자막용 텍스트
    audio_b64 = await _synthesize_safe(reply)  # ⑤ TTS 오디오(가능하면)
    if audio_b64 is not None:
        await send({"type": "tts_audio", "format": "wav", "data": audio_b64})
    return switched_session


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    # 이 연결의 모든 송신을 하나의 Lock으로 직렬화한다.
    # 메인 루프(답변 전송)와 watcher 태스크(능동 안내)가 동시에 같은 websocket에 쓰면
    # ASGI 프레임이 깨질 수 있으므로, 항상 이 send()로만 보낸다(직접 send_json 금지).
    send_lock = asyncio.Lock()
    conn_open = {"v": True}   # 연결 생존 플래그 — 끊긴 뒤 전송 시도(watcher 스팸)를 막는다.

    async def send(payload: dict) -> None:
        if not conn_open["v"]:
            return
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except Exception:
                # 클라이언트가 끊긴 뒤엔 조용히 중단. watcher가 죽은 소켓에 계속 쏘며
                # 예외를 스팸하는 것을 막는다(실제 연결 정리는 메인 루프 finally가 담당).
                conn_open["v"] = False

    history: list = []
    # 이 연결에서 마지막으로 보던 경기. 서버가 '현재 경기'를 직접 기억해,
    #   ① 음성 find_session 전환이 이후 발화에도 유지되고
    #   ② 발화에 session_key를 안 싣는 클라이언트(CLI 등)도 올바른 경기를 보게 한다.
    # (전에는 태스크-로컬 contextvar의 우연한 유지에 기댔는데, 메시지 처리를 병렬화하면 깨진다.)
    session_key: int | None = None

    # 예측형 능동 안내(watcher) 상태 — heartbeat로 갱신되는 최신 리플레이 상태 + 감시 태스크.
    latest_state: dict = {"v": None}
    watcher_task: asyncio.Task | None = None

    async def _announce(
        driver_number: int,
        probability: float,
        message: str,
        event: dict | None = None,
    ) -> None:
        """watcher가 부르는 콜백 — 그 차에 예측 리본 표시 + 안내 음성(TTS).

        highlightDriver(선택 강조) 대신 predictOvertake를 보낸다: 능동 안내는 수동 예측이라
        사용자의 '선택 차량(이 선수)'을 가로채면 안 되고, 리본이 예측 표현에 더 맞다.
        probability(0~1)는 Unity가 리본 강도로 사용한다.
        """
        if event:
            set_recent_overtake(event)
        await send({
            "type": "command", "name": "predictOvertake",
            "args": {
                "driver_number": driver_number,
                "probability": round(max(probability, 0.85), 4),
                "risk_label": "Overtake Risk High",
            },
        })
        await send({"type": "assistant_text", "text": message})
        audio_b64 = await _synthesize_safe(message)
        if audio_b64 is not None:
            await send({"type": "tts_announce", "format": "wav", "data": audio_b64})
        else:   # TTS 꺼짐/실패 시 자막만
            await send({"type": "assistant_text", "text": message})

    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")

            # 세션 폴백: 발화에 session_key가 오면 그걸로 갱신(Unity가 다른 경기 로드 시 우선),
            #            없으면 이 연결의 마지막 세션을 채워 넣는다.
            if msg.get("session_key") is not None:
                session_key = msg["session_key"]
            elif session_key is not None:
                msg["session_key"] = session_key

            # 리플레이 상태 heartbeat — 발화 없이도 현재 시각을 알려준다(예측형 능동 안내용).
            # 첫 heartbeat가 오고 기능이 켜져 있으면 감시 루프를 백그라운드로 시작한다.
            if mtype == "replay_state":
                msg["_received_monotonic"] = time.monotonic()
                if "is_playing" not in msg and "isPlaying" in msg:
                    msg["is_playing"] = msg["isPlaying"]
                latest_state["v"] = msg
                if settings.predict_watcher_enabled and watcher_task is None:
                    logger.info("[hb] 첫 replay_state 수신 → watcher 태스크 시작")
                    watcher_task = asyncio.create_task(
                        watch(lambda: latest_state["v"], _announce)
                    )
                continue

            # 능동 안내(pointOut): 짧은 문장을 '음성만' 빠르게 합성해 돌려준다.
            # 에이전트(LLM)를 안 거치므로 시간에 민감한 "곧 추월!" 안내에 적합.
            if mtype == "speak":
                say = msg.get("text")
                if say:
                    audio_b64 = await _synthesize_safe(say)
                    if audio_b64 is not None:
                        # 능동 안내 전용 타입 → Unity가 "답변 재생 중이면 건너뛰기"로 처리
                        await send({"type": "tts_announce", "format": "wav", "data": audio_b64})
                continue

            # 입력 정규화: 텍스트/음성 어느 쪽이 와도 text 한 줄로 만든다.
            if mtype == "utterance":
                text = msg.get("text")
            elif mtype == "audio_utterance":
                text = await _transcribe_safe(msg.get("data", ""))
                if text:                     # 무엇으로 인식됐는지 Unity에 회신(자막 확인)
                    await send({"type": "transcript", "text": text})
            else:
                continue

            if not text:
                continue

            # 발화에 현재 리플레이 시각이 빠졌으면 최신 heartbeat 값을 보강한다.
            # Unity가 at_time을 매 발화에 싣지 못해도 스포일러 방지와 "현재 상황" 답변이
            # 결승 완료 기준으로 새는 문제를 막는다.
            if not msg.get("at_time"):
                st = latest_state["v"] or {}
                if st.get("at_time"):
                    msg["at_time"] = st["at_time"]
            if msg.get("session_key") is None:
                st = latest_state["v"] or {}
                if st.get("session_key") is not None:
                    msg["session_key"] = st["session_key"]

            # 한 발화 처리 중 오류가 나도 연결·대화는 유지한다(우아한 실패).
            try:
                switched = await _handle_utterance(send, text, msg, history)
                if switched is not None:      # 음성으로 경기를 바꿨으면 연결 상태에 반영
                    session_key = switched
            except WebSocketDisconnect:
                raise                        # 연결 끊김은 바깥에서 처리
            except Exception:
                logger.exception("발화 처리 실패")
                try:
                    await send({
                        "type": "assistant_text",
                        "text": "죄송해요, 잠시 문제가 있었어요. 다시 말씀해 주세요.",
                    })
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    finally:
        if watcher_task is not None:        # 연결 끊기면 감시 루프 정리
            watcher_task.cancel()
