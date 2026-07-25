"""환경변수/설정 로딩. .env 파일에서 API 키와 모델명을 읽는다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (에이전트 두뇌) — tool calling 지원 모델
    llm_model: str = "gpt-4.1"
    openai_api_key: str = ""

    # 데이터 소스(직결 폴백용). 기본 경로는 F1_XR_Server 경유이며,
    # 커리어(Jolpica)도 서버가 담당하므로 AI는 Jolpica를 직접 호출하지 않는다.
    openf1_base: str = "https://api.openf1.org/v1"

    # F1_XR_Server (리플레이/데이터 백엔드) — 있으면 여기 경유, 없으면 OpenF1 직접
    f1_server_url: str = ""

    # 기본 세션 (데모용 과거 경기 리플레이) — 모나코 2024 Race
    default_session_key: int = 9523


settings = Settings()
