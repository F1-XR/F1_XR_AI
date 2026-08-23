"""환경변수/설정 로딩. .env 파일에서 API 키와 모델명을 읽는다."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (에이전트 두뇌) — tool calling 지원 모델.
    # 실제 값은 .env 의 LLM_MODEL 이 덮어쓴다. 이건 .env 가 없을 때의 기본값.
    llm_model: str = "gpt-5.4-mini"
    openai_api_key: str = ""
    # OpenAI 호환 엔드포인트 base_url. 비우면 OpenAI 본가.
    # 로컬/호스팅 모델(예: Gemma 4)로 바꿀 때 .env 의 LLM_BASE_URL 로 지정.
    llm_base_url: str = ""
    # 응답 토큰 상한. 답변은 음성으로 읽어주므로 짧게 유지(장황 방지). 추론형 모델은
    # '생각 토큰'도 여기 포함되니 너무 낮으면 답이 잘린다 → Gemma q4는 640~1024 권장.
    # .env: LLM_MAX_TOKENS
    llm_max_tokens: int = 1024
    # 로컬/양자화 모델은 샘플링을 낮게 둬야 툴 선택과 짧은 답변이 안정적이다.
    # .env: LLM_TEMPERATURE
    llm_temperature: float = 0.1
    # ⚠️ 이 모델은 추론(reasoning)형이라 답 전에 '생각'을 길게 하고, 그 생각 토큰이
    #    max_tokens를 소진하면 최종 답(content)이 빈 채로 잘린다(=안내 튜토리얼 먹통 원인).
    #    아래를 켜면 요청 단위로 thinking을 꺼 답이 즉시 나온다(공유 서버 설정은 안 건드림).
    #    vLLM/Qwen 계열: extra_body.chat_template_kwargs.enable_thinking=false 로 전달된다.
    llm_disable_thinking: bool = True
    # 로컬 모델이 도구 호출 후 최종 텍스트를 비우는 경우, 한 번 더 "최종 문장만" 강제 재요청.
    # 빈 답일 때만 발동(정상 답이면 추가 호출 없음). 지연이 부담되면 .env 로 끈다.
    empty_reply_retry: bool = True
    # 로컬(양자화) 모델이 도구콜 JSON을 깨뜨려 500이 나는 경우(비결정적)의 자동 재시도 횟수.
    # 예: 2 → 최대 3번 시도. 대부분 한 번쯤은 정상 JSON이 나와 성공한다.
    tool_error_retries: int = 2
    # 데모 핵심 발화를 LLM 전에 규칙 라우터로 안정 처리할지 여부.
    # 순수 에이전트 성능을 비교할 때만 .env/환경변수로 false.
    demo_rule_router_enabled: bool = True
    # 복합 리플레이/카메라 명령을 action plan으로 분해해 실행할지 여부.
    command_planner_enabled: bool = True

    # 음성 공급자 — .env 의 STT_PROVIDER/TTS_PROVIDER 로 교체(모델 교체 시 여기만 바꿈)
    stt_provider: str = "whisper"    # whisper | voxtral
    tts_provider: str = "qwen3"    # melotts | qwen3 | cosyvoice2 | elevenlabs
    stt_model: str = "large-v3-turbo"   # faster-whisper 모델 (large-v3-turbo | small | base)

    # Qwen3-TTS 전용 (tts_provider=qwen3 일 때만 사용)
    qwen_tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_speaker: str = "Sohee"       # 한국어 여성 화자(Qwen3). 다른 화자로 바꿔 실험 가능
    tts_instruct: str = ""           # 말투 지시(예: "밝고 친절한 톤으로"). 비우면 기본
    tts_enabled: bool = True         # WS 응답에 TTS 오디오 포함 여부(.env: TTS_ENABLED=false 로 끔)
    warmup_on_start: bool = True     # 서버 부팅 시 STT/TTS 모델 미리 로드(첫 요청 지연 제거)

    # 데이터 소스(직결 폴백용). 기본 경로는 F1_XR_Server 경유이며,
    # 커리어(Jolpica)도 서버가 담당하므로 AI는 Jolpica를 직접 호출하지 않는다.
    openf1_base: str = "https://api.openf1.org/v1"

    # F1_XR_Server (리플레이/데이터 백엔드) — 있으면 여기 경유, 없으면 OpenF1 직접
    f1_server_url: str = ""

    # 능동 안내(예측형) — 서버가 리플레이를 지켜보다 '곧 추월' 예측 시 스스로 안내.
    # ⚠️ 켜면 Unity의 규칙형 PointOutWatcher는 꺼야 한다(안내 겹침 방지). 기본 off.
    predict_watcher_enabled: bool = False
    # ⚠️ 확률은 isotonic 보정값이라 강한 접전도 ~0.3 안팎(희귀 사건). 0.5는 거의 안 울림.
    #    데모 기본값은 오탐을 줄이기 위해 보수적으로 둔다. 필요 시 .env에서 낮춘다.
    watcher_threshold: float = 0.3     # 이 확률 이상이면 안내(0~1)
    watcher_period_sec: float = 0.5    # 감시 주기(초)
    watcher_cooldown_sec: float = 10.0 # 연속 안내 최소 간격(초)
    # 포메이션 랩은 watcher가 공식 Lap 1 시작시각으로 별도 완전 차단한다.
    # 실제 레이스는 Lap 1부터 측정한다.
    watcher_ignore_lap1: bool = False
    watcher_require_closing: bool = True # 앞차와 간격이 벌어지는 후보는 억제
    # 순수 ML 예측 평가에서는 아래 세 보조 경로를 끈다. 각각을 켜면 모델 threshold와
    # 무관하게 알림이 발생할 수 있으므로 데모 연출/실험용으로만 명시적으로 활성화한다.
    watcher_replay_confirmation_enabled: bool = False
    watcher_hybrid_enabled: bool = False # ML 점수 + gap/closing 도메인 신호 결합
    watcher_hybrid_gap_sec: float = 0.85
    watcher_hybrid_closing_delta: float = -0.05
    watcher_hybrid_min_probability: float = 0.05
    watcher_fast_hybrid_enabled: bool = False
    watcher_fast_gap_sec: float = 0.15
    watcher_fast_closing_delta: float = -0.1
    watcher_fast_min_elapsed_sec: float = 300.0
    watcher_debug: bool = False       # WATCHER_DEBUG=true면 후보/skip/fire 이유를 터미널에 자세히 출력
    # 능동 안내 오탐 평가용 로그. 안내할 때마다 예측(차량·시각·확률)을 jsonl로 적재하고,
    # 나중에 scripts/eval_watcher.py 로 실제 추월 여부와 대조해 오탐률·Precision@K를 집계한다.
    watcher_eval_enabled: bool = True
    watcher_eval_log: str = "logs/watcher_eval.jsonl"

    # 기본 세션 (데모용 리플레이) — 2025 아부다비 GP(Yas Marina) Race
    # 실제 값은 .env 의 DEFAULT_SESSION_KEY 가 덮어쓴다.
    default_session_key: int = 9839


settings = Settings()
