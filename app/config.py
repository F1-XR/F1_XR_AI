"""환경변수/설정 로딩. .env 파일에서 API 키와 모델명을 읽는다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (에이전트 두뇌) — tool calling 지원 모델
    llm_model: str = "gpt-4.1"
    openai_api_key: str = ""

    # 데이터 소스
    openf1_base: str = "https://api.openf1.org/v1"
    jolpica_base: str = "https://api.jolpi.ca/ergast/f1"

    # F1_XR_Server (리플레이/데이터 백엔드) — 있으면 여기 경유, 없으면 OpenF1 직접
    f1_server_url: str = ""

    # 기본 세션 (데모용 과거 경기 리플레이)
    default_session_key: int = 9839


settings = Settings()
