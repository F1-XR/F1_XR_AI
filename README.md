# F1_XR_AI

F1 입문자용 튜토리얼 AI 에이전트 (LangGraph + FastAPI). 음성으로 선수·경기 상황을 묻고,
AI가 데이터 근거로 답하며 Unity(F1_XR_Visualizer)에서 마커·리플레이를 제어한다.

## 구조

```
app/
├─ main.py          # FastAPI + WebSocket 엔드포인트
├─ cli.py           # 텍스트 전용 테스트 러너
├─ config.py        # 설정(.env)
├─ agent/
│  ├─ graph.py      # LangGraph 에이전트(툴콜 루프) + run_agent()
│  ├─ tools.py      # 도구 8종(세션전환 1 / 조회형 4 / 명령형 3)
│  ├─ watcher.py    # pointOutMoment 능동형 감시 루프(P2, stub)
│  ├─ commands.py   # Unity 명령 싱크
│  └─ context.py    # 요청 컨텍스트(세션/시각)
├─ data/
│  ├─ openf1.py     # OpenF1 클라이언트
│  ├─ jolpica.py    # Jolpica(커리어) 클라이언트
│  ├─ drivers.py    # 번호↔Jolpica id 매핑
│  └─ glossary.json # 튜토리얼 용어집
├─ voice/
│  ├─ stt.py        # STT 래퍼(Day12, stub)
│  └─ tts.py        # TTS 래퍼(Day14, stub)
└─ ws/protocol.py   # WebSocket 메시지 스키마(Unity와 공유)
tests/
└─ test_tools.py    # 도구 기본 테스트
```

> 참고: `voice/`·`watcher.py`·`tests/`는 골격(stub)만 있고 해당 Day에 채웁니다.
> 도구는 7종(툴콜) + pointOutMoment 1종(능동형 감시 루프) = 기능상 8종.

## 실행

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # OPENAI_API_KEY 채우기

# 1) 텍스트 모드로 에이전트 검증 (Day 1~6)
python -m app.cli

# 2) 서버 실행 (Unity 연동)
uvicorn app.main:app --reload   # http://localhost:8000/health, ws://localhost:8000/ws
```

## 파이프라인

```
사용자 발화 → (STT) → LangGraph 에이전트(의도→도구 선택)
  → 조회형 도구: OpenF1/Jolpica 조회 → 근거
  → 명령형 도구: Unity 명령 적재(highlight/replay/jump)
  → LLM 한국어 응답 생성 → (TTS) → 사용자 / Unity 명령 전송
```

자세한 설계는 Notion "입문자용 튜토리얼 AI" 참고.
