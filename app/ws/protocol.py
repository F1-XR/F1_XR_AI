"""WebSocket 메시지 스키마 (에이전트 서버 ↔ Unity 클라이언트).

이 계약은 F1_XR_AI와 F1_XR_Visualizer가 공유한다. 바꾸면 양쪽 동시 갱신.
"""
from __future__ import annotations

from pydantic import BaseModel


# ── Client(Unity) → Server(AI) ──
class Utterance(BaseModel):
    type: str = "utterance"
    text: str
    session_key: int | None = None
    at_time: str | None = None          # 현재 리플레이 시각(ISO)


class AudioUtterance(BaseModel):
    """음성 발화 — base64로 인코딩한 wav. 서버가 STT로 텍스트 변환 후 에이전트로 넘긴다."""
    type: str = "audio_utterance"
    data: str                            # base64 wav
    session_key: int | None = None
    at_time: str | None = None


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
