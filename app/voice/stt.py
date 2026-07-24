"""STT 래퍼 (Day 12) — 음성 바이트 → 텍스트.

지금은 스캐폴드 stub. Day 12에 실제 모델을 붙인다.
  - 기본: Whisper large-v3-turbo (+faster-whisper) 또는 Voxtral
  - 완전 오프라인: Sentis tiny (Unity 온디바이스)
"""
from __future__ import annotations


async def transcribe(audio: bytes, language: str = "ko") -> str:
    """음성 → 텍스트. TODO(Day12): Whisper/Voxtral 연결."""
    raise NotImplementedError("STT 미구현 — Day12")
