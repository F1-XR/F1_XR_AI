# F1_XR_AI — 실시간 XR 관람을 위한 멀티모달 AI 파이프라인

F1을 처음 보는 사람이 **음성으로 질문하면**, AI가 실제 경기 데이터를 근거로 답하고
Unity XR 화면의 리플레이·마커·카메라를 제어하는 시스템입니다. 여기에 더해, 학습된
**추월 예측 모델**이 리플레이를 백그라운드로 지켜보다 "곧 추월이 나올 것 같아요"를
스스로 안내합니다.

> 이 문서는 **AI 파트**(추월 예측 모델 + 실시간 에이전트 서버)의 설계·성능·한계를
> 정리한 포트폴리오용 README입니다. 팀 프로젝트이며, **Unity 시각화/VFX/카메라/자막
> 렌더링은 팀원 담당**입니다. 아래 [My Contribution](#my-contribution)에 경계를 명시했습니다.

---

## 한눈에

- **추월 예측**: 단순 순위 상승을 추월로 보지 않고, 피트/세이프티카/리타이어/랩드/페널티/재시작을
  `event_type` 8종으로 분리해 **"트랙 위 실제 추월"만 양성 라벨**로 정의. LightGBM + isotonic 보정.
  **미학습 서킷(Spa) held-out** ROC-AUC **0.91**, PR-AUC **0.43** (추월은 양성률 ~2.9%의 희귀 사건).
- **실시간 파이프라인**: `STT → LangGraph ReAct Agent(도구 12종) → TTS → Unity(WebSocket)`.
  각 단계 지연을 계측하는 하니스(`scripts/bench_latency.py`)로 p50/p95를 측정.
- **에이전트 평가**: 발화 31문항 평가셋으로 **Tool 선택 정확도**를 채점. GPT ↔ 경량 모델 교체 시
  tool-selection 성능 변화를 정량 비교(`scripts/eval_agent.py`).

---

## 시스템 아키텍처

```
                                   ┌────────────────────────────────────────┐
                                   │  데이터 소스                             │
                                   │   • OpenF1 (position/intervals/laps/…)   │
                                   │   • F1_XR_Server (리플레이·커리어 게이트웨이)│
                                   └───────────────┬────────────────────────┘
                                                   │ (도구가 조회)
   사용자 음성                                     │
      │                                            ▼
      ▼                          ┌──────────────────────────────────────────┐
 ┌─────────┐   text     ┌────────┤  LangGraph ReAct Agent (도구 12종)         │
 │  STT     │──────────▶│ 의도   │   조회형 : get_driver_info · get_race_status│
 │ Whisper  │           │ 판단   │            explain_concept · explain_why    │
 │ turbo    │           │  +     │   세션   : find_session                     │
 └─────────┘           │ 도구   │   명령형 : highlight_driver · control_replay │
                        │ 선택   │            jump_to_event                     │
      ▲ (선택 차량      │ (ReAct │   예측·해설: predict_overtake ·             │
      │  interaction    │  루프) │            show_battle_context ·            │
      │  context)       └───┬────┤            explain_situation                │
      │                     │    │   시점   : toggle_drone_view                │
      │                     │    └──────────────┬─────────────────────────────┘
      │                     │  Overtake Prediction Model (LightGBM+isotonic)   │
      │                     │  ▲ predict_overtake / 배경 watcher가 호출         │
      │                     ▼                    │
      │              ┌─────────────┐             │
      │              │ 응답 생성    │◀────────────┘  (도구 결과 → 한국어 문장)
      │              │ (한국어 2~3  │
      │              │  문장, TTS용)│
      │              └──────┬──────┘
      │                     ▼
      │              ┌─────────────┐        ┌──────────────────────────────┐
      └──────────────│    TTS       │───────▶│  Unity XR (F1_XR_Visualizer) │
        자막·오디오   │ Qwen3 / Melo │  WS    │  리플레이·마커·리본 VFX·카메라 │
                      └─────────────┘        │  · 자막  (렌더링 = 팀원 담당)  │
                                             └──────────────────────────────┘

    ── 예측형 능동 안내(watcher) ────────────────────────────────────────────
      Unity replay_state(heartbeat) ─▶ 접전 후보 추출 ─▶ 추월 예측 모델 ─▶
      임계 이상이면 "N번, 곧 추월할 것 같아요!" 음성 + predictOvertake 명령 push
```

전체 흐름 요약: **사용자 발화 → (STT) → 의도 파악·도구 선택 → 도구 실행(데이터 조회/예측/명령)
→ 응답 생성 → (TTS) → Unity XR**. 예측 모델은 사용자 요청(`predict_overtake`)과
백그라운드 감시 루프(`watcher`) 양쪽에서 쓰입니다.

---

## My Contribution

**AI 파트 전체**를 담당했습니다. (Unity 시각화/VFX/카메라/자막 렌더링은 팀원)

| 영역 | 내용 | 위치 |
| --- | --- | --- |
| 추월 예측 — 라벨 설계 | 단순 순위변화가 아닌 `event_type` 8종 분리, "트랙 위 추월"만 양성 정의 | `F1_XR_overtakeML/pipeline.py` |
| 추월 예측 — 전처리/필터 | 피트/세이프티카/리타이어/재시작/1랩 예외 윈도우 정의 및 학습 제외 | `pipeline.py`, `train_races.py` |
| 추월 예측 — 피처 엔지니어링 | 26개 피처(gap·추세·타이어·DRS·트랙 진행률·날씨·서킷) | `train_races.py` |
| 추월 예측 — 학습/보정/평가 | LightGBM + isotonic, 서킷·시즌 홀드아웃, 누수검증, held-out 평가 | `train_races.py`, `evaluate_external_race.py` |
| 실시간 에이전트 | LangGraph ReAct 에이전트, 도구 12종, 결정적 planner/rule-router, 빈 답 방어·복구 | `app/agent/` |
| 실시간 추론 배선 | 단일시점 26피처 빌더(스포일러 방지), Booster 로드 + 보정 추론 | `app/ml/features.py`, `app/ml/predict.py` |
| 음성 파이프라인 | STT/TTS 공급자 교체형 래퍼, 한국어 숫자 정규화, TTS LRU 캐시 | `app/voice/`, `app/main.py` |
| 예측형 능동 안내 | 리플레이 감시 + 추월 예측 push, 쿨다운/중복 방지, 오탐 평가 | `app/agent/watcher.py`, `scripts/eval_watcher.py` |
| 서버/프로토콜 | FastAPI + WebSocket, 동시 전송 Lock, OpenF1 게이트웨이 | `app/main.py`, `app/ws/`, `app/data/` |

---

## Part A. 추월 예측 — "라벨을 만드는 과정"이 핵심

### A.1 파이프라인

```
Raw F1 Data          Position Change        Exception              Overtake Label       Feature            Model               Prediction
(OpenF1 10종)   ──▶  Detection         ──▶  Filtering        ──▶   (event_type=1만)  ──▶ Engineering  ──▶  LightGBM       ──▶  overtake_prob
position/intervals   30초 내 순위 상승·      pit/control/           트랙 위 실제 추월    26 features        + isotonic          position_gain/loss
laps/stints/pit/     하락을 pairwise로       retirement/penalty/    만 양성(1),          (gap·추세·타이어·   (year-OOF 보정)     /change (0~1)
race_control/…       앞차와 비교해 탐지      restart/lap1 윈도우    나머지는 6개 예외    DRS·트랙위치·날씨)
                                            로 표시                event_type로 분리
```

### A.2 왜 단순 순위 상승 ≠ 추월인가 (라벨 설계)

경주 중 순위는 **추월이 아닌 이유로도** 바뀝니다. 피트인, 세이프티카/VSC, 상대 리타이어,
백마커 랩드(lapping), 페널티, 재시작 등입니다. 이걸 전부 "추월"로 라벨링하면 모델은
**추월이 아닌 패턴**을 학습합니다. 그래서 각 `(t, 드라이버)` 시점을 8종으로 분류하고,
**`on_track_overtake`(트랙 위, 같은 랩, 접전에서의 실제 추월)만 양성**으로 씁니다.

| code | event_type | 처리 |
| ---: | --- | --- |
| 0 | no_change | 음성 후보 |
| **1** | **on_track_overtake** | **양성 라벨(추월)** |
| 2 | pit_related_change | 제외(pit_window) |
| 3 | retirement_related_change | 제외 |
| 4 | lapping_pass | 제외(백마커, `not_same_lap`) |
| 5 | penalty_related_change | 제외 |
| 6 | restart_overtake | 제외(restart_phase) |
| 7 | uncertain | 제외 |

동시에 4개 타깃(`overtake` / `position_gain` / `position_loss` / `position_change`)을
같은 파이프라인에서 학습합니다. 누수 방지 원칙: **피처는 `t` 시점까지의 데이터만, 라벨은
미래(30초 창)만** 봅니다.

### A.3 데이터 · 실험 설정

| 항목 | 값 |
| --- | --- |
| 데이터 소스 | OpenF1 (무료 공개 데이터) |
| 학습 서킷 | 5종 — Sakhir(Bahrain), Monza, Monte Carlo, Silverstone, Singapore |
| 시즌 | 2023 · 2024 · 2025 |
| held-out 평가 | **Spa-Francorchamps** (학습에서 완전 제외, 155,617 rows) |
| 격자 / 지평 | 1초 grid / 30초 horizon |
| 배틀 필터 | 앞차 gap ≤ 2초 구간만 |
| 피처 | 26개 (스키마 `event_type_v1_26`) |
| 모델 | LightGBM(`n_estimators=400`, `lr=0.05`, `class_weight=balanced`) |
| 보정 | isotonic regression (leave-one-year-out OOF) |
| 분리(split) | 시즌 홀드아웃(2023·2024 학습 / 2025 검증) + 서킷 홀드아웃(Spa) |
| 누수 검증 | 학습/평가 세션 겹침 **0건**, 평가 서킷이 학습에 등장 **0건** |

**26개 피처**: 상황(`gap_ahead`, `gap_trend`, `position`, `position_delta`, `same_lap`),
속도·DRS(`speed`, `speed_delta`, `drs_range`, `drs_active`), 타이어(`tyre_age`, `tyre_age_delta`),
트랙 위치(`track_progress`, `track_progress_sin/cos`, `sector`, `segment`),
컨텍스트(`season`, `circuit_key`, `circuit_type_code`, `is_lap1`, `restart_phase`),
날씨(`air/track_temperature`, `humidity`, `rainfall`, `weather_regime_code`).

### A.4 결과 — 4개 타깃 (미학습 서킷 Spa held-out)

| 타깃 | OOF ROC-AUC | OOF PR-AUC | held-out ROC-AUC | held-out PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| **overtake** | 0.888 | 0.366 | **0.909** | **0.428** |
| position_gain | 0.894 | 0.387 | 0.873 | 0.393 |
| position_loss | 0.913 | 0.315 | 0.831 | 0.195 |
| position_change | 0.888 | 0.403 | 0.853 | 0.421 |

> OOF = 시즌 교차검증(out-of-fold), held-out = strict filtering 후 미학습 서킷 성능.
> 추월은 **양성률 ~2.9%의 희귀 사건**이라 정확도(accuracy)가 아니라 **PR-AUC**를 주지표로 봅니다.

### A.5 결과 — `overtake` threshold별 Precision / Recall / F1 + Confusion Matrix

Spa held-out(strict filtering 후 121,571 rows, 양성 4,503). 아래 표는 저장된 평가 리포트에서
`scripts/analyze_overtake_report.py`로 재현됩니다.

| threshold | TP | FP | FN | TN | Precision | Recall | **F1** |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 2,417 | 4,229 | 2,086 | 112,839 | 0.364 | 0.537 | **0.434** |
| 0.20 | 1,478 | 1,095 | 3,025 | 115,973 | 0.574 | 0.328 | 0.418 |
| 0.30 | 1,153 |   651 | 3,350 | 116,417 | 0.639 | 0.256 | 0.366 |
| 0.50 |   603 |   193 | 3,900 | 116,875 | 0.758 | 0.134 | 0.228 |

임계값으로 정밀도/재현율을 조절합니다. 능동 안내(watcher)는 **오탐을 줄이는 쪽(높은 정밀도)**이
중요하므로 보수적 임계값을 씁니다.

### A.6 재현

```bash
# F1_XR_overtakeML 레포에서 저장된 평가 리포트로부터 위 표를 재현
python -m scripts.analyze_overtake_report \
    --report ../F1_XR_overtakeML/results/external_eval_spa_2025_report.json \
    --target all
```

---

## Part B. 실시간 멀티모달 파이프라인 — 단계별 지연(latency)

"GPT로 안내하는 튜토리얼"이 아니라 **"각 구성요소의 지연을 측정한 실시간 멀티모달 AI
파이프라인"**임을 수치로 보이기 위해, 각 단계 지연을 계측합니다.

### B.1 계측 방법

`app/obs.py`(경량 contextvar 트레이스)가 발화 처리 중 단계별 시간과 도구 호출을 기록합니다.
**운영 동작은 바꾸지 않고(측정용)**, 벤치마크가 트레이스를 읽어 통계를 냅니다.

| 단계 | 내용 |
| --- | --- |
| `stt` | 음성 → 텍스트 (faster-whisper large-v3-turbo) |
| `agent_total` | run_agent 전체 (라우팅 + 도구 실행 + LLM 응답) |
| `agent_llm` | 그 중 LLM/툴 호출 구간 (ReAct 경로) |
| `tts` | 텍스트 → 음성 wav (Qwen3-TTS / MeloTTS) |
| `end_to_end` | stt + agent_total + tts (사용자 체감 지연) |

### B.2 실행

```bash
python -m scripts.bench_latency --repeats 30                 # 기본 7개 프롬프트 × 30회
python -m scripts.bench_latency --repeats 30 --stt-wav tts_out.wav   # STT까지 포함
python -m scripts.bench_latency --no-tts                     # 에이전트만
```

출력은 아래 형식의 표(평균/p50/p95, ms)와 `results/latency_report.json`입니다.
`.env`의 `LLM_MODEL`을 바꿔 재측정하면 모델별 지연을 비교할 수 있습니다.

| 단계 | n | 평균(ms) | p50(ms) | p95(ms) |
| --- | ---: | ---: | ---: | ---: |
| STT (음성→텍스트) | · | _측정_ | _측정_ | _측정_ |
| Agent 전체 | · | _측정_ | _측정_ | _측정_ |
| Agent LLM/툴 실행 | · | _측정_ | _측정_ | _측정_ |
| TTS (텍스트→음성) | · | _측정_ | _측정_ | _측정_ |
| **End-to-End** | · | _측정_ | _측정_ | _측정_ |

> 표는 하드웨어·모델·네트워크에 따라 달라지므로, 발표 환경에서 `bench_latency.py`를 돌려
> **그 환경의 수치**를 채웁니다. 예: "30회 기준 End-to-End 평균 1.8s / p95 2.4s" 형태로 서술.

---

## Part C. Agent 평가 — 모델 교체가 시스템 안정성에 미치는 영향

경량 모델(예: Gemma 계열)로 바꾸면 **Tool 선택 정확도가 떨어지는** 현상을 관찰했습니다.
이를 "모델이 안 좋았다"가 아니라 **실험 주제**로 만들었습니다: 복수 도구 기반 에이전트에서
모델 성능이 시스템 안정성에 미치는 영향을 정량화합니다.

### C.1 평가셋 · 채점 기준

`eval/agent_tool_eval.jsonl` — **발화 31문항**(개념/조회/제어/예측/상황해설/세션전환/복합).
각 문항은 기대 도구·필수 인자·허용 도구를 라벨로 갖습니다. 채점 5기준:

1. **Tool selection** — 기대한 도구를 골랐는가
2. **Missing** — 필요한 도구를 누락했는가
3. **Tool args** — 도구 인자(driver_number/action/event_type 등)가 맞는가
4. **Unnecessary** — 불필요한 도구를 호출했는가
5. **Final success** — 최종 한국어 답변을 냈는가

> 공정한 비교를 위해 기본적으로 결정적 라우터(planner/rule-router)를 끄고 **순수 LLM의
> tool-selection**을 측정합니다(`--with-router`로 켤 수 있음).

### C.2 실행 · 비교

```bash
# 모델 A(.env의 LLM_MODEL)로 평가
python -m scripts.eval_agent --label gpt --session-key 9839
# .env를 모델 B로 바꾼 뒤
python -m scripts.eval_agent --label gemma --session-key 9839
# 두 결과를 표로 비교
python -m scripts.eval_agent --compare results/agent_eval_gpt.json results/agent_eval_gemma.json
```

출력은 아래 형식의 비교표입니다(발표 환경에서 실제 모델로 측정해 채움):

| Model | Tool selection | Tool args | Missing | Unnecessary | Final success |
| --- | ---: | ---: | ---: | ---: | ---: |
| (model A) | _측정_ | _측정_ | _측정_ | _측정_ | _측정_ |
| (model B) | _측정_ | _측정_ | _측정_ | _측정_ | _측정_ |

결론 서술 예시: *"경량 모델로 교체 시 Tool selection이 크게 하락함을 확인했고, 복수 도구 기반
에이전트에서 모델 성능이 시스템 안정성(빈 답·오호출)에 미치는 영향을 분석했다. 이를 완화하기 위해
결정적 command planner / rule-router와 빈 답 재요청·도구 결과 복구(salvage)를 설계했다."*

능동 안내(watcher)의 **오탐 평가**도 별도로 제공합니다:

```bash
python -m scripts.eval_watcher --window 30 --k 5   # Precision, Precision@K, 분당 오탐 수
```

---

## Part D. 실패 사례 (Failure Cases)

라벨 설계에서 실제로 마주친 문제 → 원인 추적 → 데이터 재정의 과정입니다.

**Failure 01 — 피트스톱이 추월로 오염**
단순 position change를 추월로 보면, 피트인으로 인한 순위 변화가 양성에 섞입니다.
→ **해결:** 본인/앞차의 피트 이벤트 주변 `pit_window`를 정의해 학습에서 제외.
(Spa held-out에서 `pit_window` 683 rows 제거)

**Failure 02 — 세이프티카/VSC 구간 오염**
SC/VSC 동안의 비정상 순위 변화가 추월로 오인됩니다.
→ **해결:** race_control을 파싱해 SC/VSC/red-flag 구간을 `control_window`로, 그 직후를
`restart_phase`로 표시해 제외. (`restart_phase` 124 rows 제거)

**Failure 03 — 백마커 랩드(lapping)를 추월로 착각**
다른 랩에 있는 백마커를 지나치는 것은 배틀 추월이 아닙니다.
→ **해결:** `same_lap` 조건과 `lapping_pass`(event_type=4)를 분리, "같은 랩 접전"만 양성.
이 `not_same_lap` 제외가 **가장 많은 샘플(22,311 rows)** 을 걸러냈습니다.

**Failure 04 — 클래스 불균형**
strict filtering 후 양성률이 낮습니다(overtake ~3.7%). 정확도는 의미가 없습니다.
→ **해결:** `class_weight=balanced` + isotonic 보정 + **PR-AUC/threshold별 P·R·F1**로 평가.

**예외처리별 제거 샘플 (overtake, Spa held-out — 많이 제거한 순)**

| 예외처리 | 제거 rows |
| --- | ---: |
| `not_same_lap` (백마커 랩드) | 22,311 |
| `strict_on_track_event_type` (트랙 외 변화) | 10,928 |
| `pit_window` (피트 전후) | 683 |
| `restart_phase` (재시작 직후) | 124 |
| **합계** | **34,046** (155,617 → 121,571) |

---

## 레포 구조 (AI 파트)

```
app/
├─ main.py            # FastAPI + WebSocket (STT→에이전트→명령·자막·TTS)
├─ obs.py             # 경량 관측(단계 지연·툴콜 트레이스) — 계측 전용
├─ config.py          # 설정(.env: LLM/STT/TTS/watcher)
├─ agent/
│  ├─ graph.py        # LangGraph ReAct 에이전트 + run_agent()
│  ├─ tools.py        # 도구 12종
│  ├─ planner.py      # 복합 명령 결정적 플래너
│  ├─ watcher.py      # 예측형 능동 안내(추월 예측 감시 루프)
│  └─ watcher_eval.py # 능동 안내 예측 로깅(오탐 평가용)
├─ ml/
│  ├─ features.py     # 단일시점 26피처 빌더(스포일러 방지)
│  ├─ predict.py      # LightGBM Booster 로드 + isotonic 보정 추론
│  └─ models/         # 학습된 모델(.txt) + 보정 + Unity 계약(JSON)
├─ voice/{stt,tts}.py # STT/TTS 공급자 교체형 래퍼
└─ data/openf1.py     # OpenF1 데이터 게이트웨이
scripts/
├─ bench_latency.py           # 단계별 지연 p50/p95 (Part B)
├─ eval_agent.py              # 에이전트 Tool 선택 평가 (Part C)
├─ eval_watcher.py            # 능동 안내 오탐 평가
└─ analyze_overtake_report.py # 추월 예측 리서치 요약 재현 (Part A)
eval/
└─ agent_tool_eval.jsonl      # 에이전트 평가셋 31문항
```

## 실행

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # OPENAI_API_KEY / LLM_MODEL / LLM_BASE_URL 채우기

python -m app.cli                       # 텍스트 모드로 에이전트 검증
uvicorn app.main:app --reload           # 서버(Unity 연동): /health, ws://…/ws

python -m scripts.bench_latency --repeats 30           # 지연 측정 (Part B)
python -m scripts.eval_agent --label gpt --session-key 9839   # 에이전트 평가 (Part C)
```

## 관련 레포

- **F1_XR_overtakeML** — 추월 예측 모델 학습·평가 공장 (Part A의 학습 코드)
- **F1_XR_Server** — 리플레이·데이터 게이트웨이 백엔드
- **F1_XR_Visualizer** — Unity XR 시각화 (팀원 담당)

## 데이터 · 라이선스

데이터: [OpenF1](https://openf1.org/) (무료·키 불필요) · 커리어: [Jolpica-F1](https://github.com/jolpica/jolpica-f1).
코드: MIT.
