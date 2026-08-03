"""WebSocket 메시지 스키마 (에이전트 서버 ↔ Unity 클라이언트).

이 계약은 F1_XR_AI와 F1_XR_Visualizer가 공유한다. 바꾸면 양쪽 동시 갱신.
"""
from __future__ import annotations

from pydantic import BaseModel


# ── Client(Unity) → Server(AI) ──
class InteractionContext(BaseModel):
    """공간 맥락 — 사용자가 XR Ray/클릭으로 '지목한' 대상. (Battle Lens Phase 1)
    이게 있으면 "이 선수/이 차/얘" 같은 지시어를 해당 차량으로 해석한다.
    번호를 채우는 입력이 XR Ray든 마우스 클릭이든 AI엔 동일하다."""
    target_type: str | None = None       # 예: "driver"
    driver_number: int | None = None     # 선택/지목된 차량 번호
    input_modality: str | None = None    # "click" | "controller_ray" 등(로깅·분석용)


class Utterance(BaseModel):
    type: str = "utterance"
    text: str
    session_key: int | None = None
    at_time: str | None = None          # 현재 리플레이 시각(ISO)
    interaction_context: InteractionContext | None = None


class AudioUtterance(BaseModel):
    """음성 발화 — base64로 인코딩한 wav. 서버가 STT로 텍스트 변환 후 에이전트로 넘긴다."""
    type: str = "audio_utterance"
    data: str                            # base64 wav
    session_key: int | None = None
    at_time: str | None = None
    interaction_context: InteractionContext | None = None


class Speak(BaseModel):
    """능동 안내(pointOut) 전용 — 짧은 문장을 '음성만' 빠르게 합성 요청.
    에이전트(LLM)를 안 거치고 TTS만 → 시간에 민감한 안내에 사용. 응답은 tts_audio."""
    type: str = "speak"
    text: str


class ReplayState(BaseModel):
    """리플레이 상태 heartbeat — 발화가 없어도 서버가 현재 시각을 알게 주기 전송(0.5~1초).
    예측형 능동 안내(watcher)가 '지금 몇 분인지'를 알아야 스스로 안내할 수 있다."""
    type: str = "replay_state"
    session_key: int | None = None
    at_time: str | None = None       # 현재 리플레이 시각(ISO)
    is_playing: bool = True
    speed: float = 1.0
    selected_driver: int | None = None


# ── Server(AI) → Client(Unity) ──
class Command(BaseModel):
    """Unity가 실행할 명령. name ∈ {loadSession, highlightDriver, controlReplay, pointOut}"""
    type: str = "command"
    name: str
    args: dict


class Transcript(BaseModel):
    """STT 인식 결과(음성 발화가 뭐로 인식됐는지 자막 확인용)."""
    type: str = "transcript"
    text: str


class AssistantText(BaseModel):
    type: str = "assistant_text"
    text: str                            # 응답 자막(Unity 화면 표시용)


class TtsAudio(BaseModel):
    type: str = "tts_audio"
    format: str = "wav"
    data: str                            # base64 wav (TTS 합성 결과)
