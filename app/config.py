"""환경변수/설정 로딩. .env 파일에서 API 키와 모델명을 읽는다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (에이전트 두뇌) — tool calling 지원 모델.
    # 실제 값은 .env 의 LLM_MODEL 이 덮어쓴다. 이건 .env 가 없을 때의 기본값.
    llm_model: str = "gpt-5.4-mini"
    openai_api_key: str = ""

    # 음성 공급자 — .env 의 STT_PROVIDER/TTS_PROVIDER 로 교체(모델 교체 시 여기만 바꿈)
    stt_provider: str = "whisper"    # whisper | voxtral
    tts_provider: str = "qwen3"    # melotts | qwen3 | cosyvoice2 | elevenlabs
    stt_model: str = "large-v3-turbo"   # faster-whisper 모델 (large-v3-turbo | small | base)

    # Qwen3-TTS 전용 (tts_provider=qwen3 일 때만 사용)
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_speaker: str = "Sohee"       # 한국어 여성 화자(Qwen3). 다른 화자로 바꿔 실험 가능
    tts_instruct: str = ""           # 말투 지시(예: "밝고 친절한 톤으로"). 비우면 기본
    tts_enabled: bool = True         # WS 응답에 TTS 오디오 포함 여부(.env: TTS_ENABLED=false 로 끔)

    # 데이터 소스(직결 폴백용). 기본 경로는 F1_XR_Server 경유이며,
    # 커리어(Jolpica)도 서버가 담당하므로 AI는 Jolpica를 직접 호출하지 않는다.
    openf1_base: str = "https://api.openf1.org/v1"

    # F1_XR_Server (리플레이/데이터 백엔드) — 있으면 여기 경유, 없으면 OpenF1 직접
    f1_server_url: str = ""

    # 기본 세션 (데모용 리플레이) — 2025 아부다비 GP(Yas Marina) Race
    # 실제 값은 .env 의 DEFAULT_SESSION_KEY 가 덮어쓴다.
    default_session_key: int = 9839


settings = Settings()
