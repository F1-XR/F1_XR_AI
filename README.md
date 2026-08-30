# F1_XR_AI

F1_XR_AI는 F1을 처음 보는 사용자가 경기 상황을 자연어로 물어볼 수 있게 만든 튜토리얼 에이전트입니다. 사용자가 드라이버, 피트스톱, 깃발 상황, 리플레이 제어 등을 말로 요청하면 서버가 OpenF1/Jolpica 데이터를 조회하고, 필요한 경우 Unity 클라이언트로 화면 강조나 리플레이 제어 명령을 보냅니다.

현재는 텍스트 입력과 WebSocket 기반 Unity 연동을 먼저 검증하는 단계입니다. STT, TTS, 능동형 하이라이트 감시는 인터페이스만 잡아두고 이후 단계에서 붙이는 구조로 두었습니다.

## 지금 들어간 것

- FastAPI 서버와 `/health`, `/ws` WebSocket 엔드포인트
- LangGraph 기반 ReAct 에이전트 파이프라인
- OpenF1 API를 통한 세션, 드라이버, 순위, 피트, 타이어, 레이스 컨트롤 조회
- Jolpica-F1 API를 통한 드라이버 커리어 정보 조회
- Unity로 보낼 명령 버퍼: 드라이버 강조, 리플레이 재생/정지/속도/이동, 이벤트 점프
- F1 입문자용 용어집 기반 설명 도구
- 텍스트 CLI 테스트 러너
- API 연결 확인용 smoke test
- 기본 pytest 테스트

## 프로젝트 구조

```text
app/
├─ main.py             # FastAPI 앱, WebSocket 엔드포인트
├─ cli.py              # 터미널에서 에이전트를 돌려보는 텍스트 러너
├─ config.py           # 환경변수와 기본 설정
├─ agent/
│  ├─ graph.py         # LangGraph 에이전트 생성 및 run_agent()
│  ├─ tools.py         # 에이전트가 호출하는 조회/제어 도구
│  ├─ commands.py      # Unity 명령을 요청 단위로 모으는 command sink
│  ├─ context.py       # 현재 세션과 리플레이 시각 관리
│  └─ watcher.py       # 능동형 pointOutMoment 감시 루프 자리
├─ data/
│  ├─ openf1.py        # OpenF1 비동기 클라이언트
│  ├─ jolpica.py       # Jolpica-F1 비동기 클라이언트
│  ├─ drivers.py       # 드라이버 번호와 Jolpica id 매핑
│  └─ glossary.json    # F1 용어 설명 데이터
├─ voice/
│  ├─ stt.py           # 음성 인식 래퍼 자리
│  └─ tts.py           # 음성 합성 래퍼 자리
└─ ws/
   └─ protocol.py      # Unity와 공유할 WebSocket 메시지 모델

scripts/
├─ list_models.py      # OpenAI 계정에서 사용 가능한 모델 확인
└─ smoke_test.py       # OpenF1/Jolpica 연결 점검

tests/
└─ test_tools.py       # 용어 설명 도구 기본 테스트
```

## 설치

Python 3.11 이상을 기준으로 작업했습니다.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows에서는 가상환경 활성화만 아래처럼 바꾸면 됩니다.

```bash
venv\Scripts\activate
```

루트에 `.env` 파일을 만들고 OpenAI API 키를 넣습니다.

```bash
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4.1
```

`LLM_MODEL`은 생략하면 `gpt-4.1`을 사용합니다. 계정에서 사용 가능한 모델을 확인하려면 아래 스크립트를 실행하면 됩니다.

```bash
python -m scripts.list_models
```

## 실행

텍스트만으로 에이전트를 먼저 확인할 수 있습니다.

```bash
python -m app.cli
```

Unity 연동까지 보려면 서버를 실행합니다.

```bash
uvicorn app.main:app --reload
```

확인용 엔드포인트는 다음과 같습니다.

```text
GET  http://localhost:8000/health
WS   ws://localhost:8000/ws
```

## WebSocket 메시지

Unity 클라이언트는 사용자 발화를 아래 형식으로 보냅니다.

```json
{
  "type": "utterance",
  "text": "해밀턴 어디 있어?",
  "session_key": 9839,
  "at_time": "2024-05-26T14:12:30Z"
}
```

서버는 에이전트가 만든 Unity 명령을 먼저 보내고, 마지막에 사용자에게 보여줄 답변을 보냅니다.

```json
{
  "type": "command",
  "name": "highlightDriver",
  "args": {
    "driver_number": 44
  }
}
```

```json
{
  "type": "assistant_text",
  "text": "44번은 루이스 해밀턴입니다. 지금 화면에서 표시해둘게요."
}
```

## 에이전트 도구

현재 에이전트가 사용할 수 있는 도구는 아래와 같습니다.

| 도구 | 역할 |
| --- | --- |
| `find_session` | 연도, 국가/서킷, 세션명을 기준으로 OpenF1 세션을 찾고 현재 세션을 바꿉니다. |
| `get_driver_info` | 드라이버 번호로 이름, 팀, 국적, 사진, 커리어 정보를 조회합니다. |
| `get_race_status` | 현재 세션의 레이스 컨트롤 이벤트와 상위 순위를 요약합니다. |
| `explain_concept` | 로컬 용어집에서 DRS, 언더컷, 세이프티카 같은 F1 용어를 설명합니다. |
| `explain_why` | 피트, 타이어 스틴트, 인터벌 데이터를 모아 특정 상황의 이유를 설명할 근거를 제공합니다. |
| `highlight_driver` | Unity 화면에서 특정 드라이버를 강조하도록 명령을 쌓습니다. |
| `control_replay` | Unity 리플레이 재생, 정지, 속도 조절, 이동 명령을 쌓습니다. |
| `jump_to_event` | 첫 피트스톱, 세이프티카, 옐로 플래그 같은 이벤트 시점으로 이동합니다. |

`pointOutMoment`는 사용자 질문으로 호출하는 도구가 아니라 서버가 경기 흐름을 지켜보다가 중요한 순간을 먼저 알려주는 기능으로 분리해두었습니다. 아직은 `watcher.py`에 구현 자리가 있는 상태입니다.

## 동작 흐름

```text
사용자 발화
  -> FastAPI WebSocket
  -> LangGraph 에이전트
  -> 필요한 도구 호출
     -> OpenF1/Jolpica 데이터 조회
     -> 또는 Unity 명령 버퍼에 command 적재
  -> 한국어 답변 생성
  -> Unity command 전송
  -> assistant_text 전송
```

## 점검

외부 API 연결을 확인합니다.

```bash
python -m scripts.smoke_test
```

오프라인 단위 테스트를 실행합니다.

```bash
pytest
```

현재 테스트는 용어 설명 도구 중심입니다. 네트워크가 필요한 OpenF1/Jolpica 조회와 WebSocket 통합 테스트는 별도 확장 예정입니다.

## 아직 남은 일

- `at_time` 기준으로 레이스 컨트롤, 순위, 피트, 인터벌 데이터를 잘라서 실제 리플레이 시점에 맞는 답변 만들기
- 드라이버 이름, 약어, 한국어 표기를 드라이버 번호로 바꾸는 resolver 강화
- WebSocket 메시지를 `protocol.py`의 Pydantic 모델로 검증하고 에러 응답 정리
- 연결별 세션 상태 분리. 지금 `context.py`는 단일 사용자 데모 기준입니다.
- STT/TTS 실제 모델 연결
- `watcher.py`의 pointOutMoment 감시 루프 구현
- Unity 쪽 명령 스키마와 end-to-end 테스트 맞추기

## 메모

기본 세션은 `DEFAULT_SESSION_KEY=9839`로 잡혀 있습니다. 다른 경기로 보고 싶으면 Unity 메시지에서 `session_key`를 넘기거나, 사용자 발화에서 연도와 경기명을 말해 `find_session`이 세션을 전환하게 하면 됩니다.
