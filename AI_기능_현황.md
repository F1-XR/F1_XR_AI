# F1 XR — AI 파트 기능 현황

_최종 업데이트: 2026-08-10_

AI 파트는 3-레포 구조(F1_XR_AI :8001 / F1_XR_Server :8000 / F1_XR_Visualizer)에서
**음성·텍스트로 F1 리플레이를 안내하는 튜터 + 실시간 추월 예측 능동 안내**를 담당한다.

---

## ✅ 완료된 기능

### 대화형 에이전트 (LangGraph ReAct)
- 사용자 발화(텍스트/음성) → LLM이 의도 판단 → 도구 호출 → 한국어 답변
- 도구 10종:
  - 조회형: `get_driver_info`, `get_race_status`, `explain_concept`, `explain_why`
  - 세션: `find_session` (말로 경기 전환)
  - 명령형: `highlight_driver`, `control_replay`, `jump_to_event`
  - 예측형: `predict_overtake`
  - 시점: `toggle_drone_view` (AI 쪽 완료, Unity 진입 메서드 대기)
- 대화 맥락 유지(최근 N턴), 우아한 실패 처리

### LLM / 음성
- **로컬 Gemma 4**(OpenAI 호환 엔드포인트)로 전환 — 무료, GPT 토큰 소진 대응
- **STT**: Whisper(faster-whisper) 다국어
- **TTS**: MeloTTS 한국어(CPU 실시간) + **응답 음성 LRU 캐시**(반복 안내 지연↓)
- 한국어 숫자 정규화(‘7번’→‘칠 번’)로 자연스러운 읽기

### 공간 지능 (Battle Lens Phase 1)
- `interaction_context`(선택 차량 번호) 전달 → **"이 선수/쟤/여기"** 지시어를 그 차로 해석
- 마우스 클릭 / 순위표 / AI highlight로 선택 → 라벨 + 온보드 뷰 자동 표시

### 추월 예측 (Battle Lens Phase 2)
- LightGBM + isotonic 보정, 30초 horizon, 26피처(18개 실시간 계산)
- 단일시점 피처 빌더: gap·position·tyre·weather·car_data(speed/drs)·track_progress
- OpenF1 데이터 게이트웨이(weather/car_data/location) 서버 경유 배선

### 예측형 능동 안내 (watcher)
- 서버가 리플레이 heartbeat를 지켜보다 **접전(gap<1.5s) 상위 6명**을 주기마다 예측
- 임계 이상이면 **"N번, 곧 추월할 것 같아요!"** 음성 + `predictOvertake` 명령 push
- 쿨다운·중복 방지, 설정(.env)으로 임계/주기/쿨다운 조절

### Unity 연동 (완료분)
- 텍스트/음성 명령 입력 (데스크톱 자동 포커스 + Enter 전송)
- **WAV 파서 개선** — MeloTTS 음성 재생 정상화(16bit/float 등 대응)
- **추월 접근 리본 VFX** ↔ 예측 연동 (메인 맵, 실시간) — 동작 확인됨
- replay_state heartbeat 송신, interaction_context 송신
- 폰트 글리프 복원(깨짐 수정)

### 버그 수정
- explainWhy 스포일러 차단(cutoff 필터)
- 세션 서버측 유지
- **WebSocket 동시 전송 Lock**(답변·능동안내 프레임 충돌 방지)
- watcher 세션 데이터 중복 조회 캐시(ReadTimeout 완화)
- TTS nltk 보안훅 오탐 해제(venv-in-project defusedxml 차단)

---

## 🔜 진행해야 할 것 (발표 전)

| 우선 | 항목 | 담당 | 비고 |
|---|---|---|---|
| 🔴 | **한글 폰트 fallback + 답변 자막 UI** | Unity | 지금 답변이 Debug.Log만 → 화면 자막 필요. 관객 임팩트 최대 |
| 🔴 | **실제 마이크 음성 테스트** | 공통 | STT는 워밍업만 확인, 실제 녹음→인식 미검증 |
| 🟡 | 능동 안내 타이밍 튜닝(`WATCHER_PERIOD_SEC`) | AI | 캐시 후 재측정, 필요시 1.5로 |
| 🟡 | 데모 서킷/threshold 확정 | 공통 | 스즈카는 추월 적음 → Monza/Bahrain 권장 |
| 🟡 | **드론 진입 코어(②)** `EnterVrCore` 분리 + `EnterVrFromCommand()` | Unity(드론 담당) | 열리면 `toggle_drone_view` 완성 |
| ⚪ | 커밋 정리 (AI: watcher/main/tools · Vis: dispatcher/handler/scene/font) | 공통 | 나머지 M은 CRLF 노이즈 |
| ⚪ | 임시 디버그 로그 제거 | 공통 | 검증 끝나면 |

---

## 💡 추가하면 좋을 기능

- **추월 팝아웃 ↔ "그 추월 다시 보여줘"** — `EventPopoutReplay.OpenNextOvertake()` 공개 API 존재. 명령+핸들러만 붙이면 극적 리플레이 연출.
- **리본 라벨** — 리본 위에 확률%·번호 빌보드 라벨(예측 강점 시각화). 앞차 지목 연결선도 옵션.
- **능동 안내 공간 음향** — 안내 음성을 그 차 방향에서 들리게(주의 유도 효과↑).
- **영어 버전(언어 토글)** — F1 영미권 대응. STT/TTS 언어 파라미터 + 영어 프롬프트. 한국어판 완성 후.
- **XR Ray 차 선택**(`ReplayXRRayInput`) — Quest 실기기용. 씬 배선 + 검증 필요.
- **드론 진입 시 그 차 위치로 스폰**(옵션 B) — "그 선수한테 드론으로" 시 바로 근처에서 시작.

---

## 📌 참고 — 역할 경계

- **AI(서버)**: 발화 이해, 도구 실행, 예측, 명령·음성 송신 (화면은 못 그림)
- **Unity**: 명령 수신 → 시각화·VFX·카메라·자막 (렌더링 전부)
- 새 XR 기능 붙일 때 원칙: AI는 "명령"만, Unity는 "핸들러 + 연출". 기존 로직은 **공유 메서드로 확장**(복붙 금지)해 충돌 방지.
