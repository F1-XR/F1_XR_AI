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
    tts_provider: str = "melotts"    # melotts | cosyvoice2 | elevenlabs

    # 데이터 소스(직결 폴백용). 기본 경로는 F1_XR_Server 경유이며,
    # 커리어(Jolpica)도 서버가 담당하므로 AI는 Jolpica를 직접 호출하지 않는다.
    openf1_base: str = "https://api.openf1.org/v1"

    # F1_XR_Server (리플레이/데이터 백엔드) — 있으면 여기 경유, 없으면 OpenF1 직접
    f1_server_url: str = ""

    # 기본 세션 (데모용 리플레이) — 2025 아부다비 GP(Yas Marina) Race
    # 실제 값은 .env 의 DEFAULT_SESSION_KEY 가 덮어쓴다.
    default_session_key: int = 9839


settings = Settings()
