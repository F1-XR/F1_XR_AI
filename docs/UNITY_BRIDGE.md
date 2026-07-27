# F1 튜토리얼 AI ↔ Unity 연동 스펙 (WebSocket 브리지)

이 문서는 **F1_XR_AI(에이전트 서버)** 와 **F1_XR_Visualizer(Unity/Quest 클라이언트)** 가
주고받는 메시지 계약과 명령 매핑을 정의한다. 두 레포가 이 문서를 **공유 계약**으로 삼는다.

> ⚠️ **동기화 규칙**: 메시지 스키마의 원본(source of truth)은 F1_XR_AI의
> `app/ws/protocol.py`(Python)다. 여기 필드명을 바꾸면 Unity의 C# 클래스도 **동시에** 갱신해야 한다.

---

## 1. 아키텍처 개요

| 레포 | 역할 | 포트 |
|---|---|---|
| **F1_XR_AI** | 에이전트 + STT + TTS + WebSocket 서버 | `:8001` |
| **F1_XR_Server** | 데이터/리플레이 백엔드 (AI가 내부적으로 호출) | `:8000` |
| **F1_XR_Visualizer** | Unity/Quest 클라이언트 (이 문서의 구현 대상) | — |

```
[Quest 마이크]
   → (Unity) audio_utterance  ──WS──▶  [F1_XR_AI :8001]
                                          STT → 에이전트(LangGraph) → TTS
   ◀──WS── transcript / command / assistant_text / tts_audio
[Unity]  자막 표시 / 화면·리플레이 제어 / 답변 음성 재생
```

- 브리지의 **서버 절반**(STT·에이전트·TTS·명령 emit)은 F1_XR_AI에 **구현 완료**.
- 브리지의 **클라이언트 절반**(4개 컴포넌트, 아래 §6)은 **F1_XR_Visualizer에서 구현**한다.

---

## 2. 연결

- 엔드포인트: `ws://<AI_SERVER_HOST>:8001/ws`
  - 로컬 테스트: `ws://localhost:8001/ws`
  - 외부/Quest: 서버를 ngrok·Cloudflare 등으로 노출한 주소 (예: `wss://xxxx.ngrok.io/ws`)
- 프로토콜: **JSON 텍스트 프레임** (모든 메시지는 `type` 필드로 구분)
- 헬스체크: `GET http://<host>:8001/health` → `{"status":"ok"}`
- 하나의 WebSocket 연결이 **대화 세션 하나**. 연결이 유지되는 동안 최근 8턴의 대화 맥락이 서버에 유지된다.

---

## 3. 메시지 계약

### 3.1 Unity → AI (클라이언트가 보냄)

**(a) 텍스트 발화** — 디버그·키보드 입력용
```json
{
  "type": "utterance",
  "text": "해밀턴 왜 피트인했어?",
  "session_key": 9839,
  "at_time": "2025-12-07T15:20:00+00:00"
}
```

**(b) 음성 발화** — 실사용(마이크)
```json
{
  "type": "audio_utterance",
  "data": "<base64로 인코딩한 wav 전체(헤더 포함)>",
  "session_key": 9839,
  "at_time": "2025-12-07T15:20:00+00:00"
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `type` | string | ✅ | `"utterance"` 또는 `"audio_utterance"` |
| `text` | string | (a)에서 ✅ | 사용자가 말/입력한 내용 |
| `data` | string | (b)에서 ✅ | base64 wav (§5 참고) |
| `session_key` | int \| null | 권장 | **지금 보고 있는 경기 ID**. 없으면 서버 기본값으로 폴백 |
| `at_time` | string(ISO) \| null | 권장 | **리플레이 현재 시각**. 이 시각 이후 결과는 스포일러 방지로 언급 안 함 |

> 💡 `session_key`/`at_time`은 **말에서 뽑는 게 아니라 Unity가 매 발화에 실어 보내는 화면 상태**다.
> 안 보내면 에이전트가 어느 경기·언제인지 몰라 기본 세션으로 답한다. **매 발화마다 현재 값을 넣을 것.**

### 3.2 AI → Unity (서버가 보냄)

한 번의 발화에 대해 서버는 보통 이 순서로 여러 메시지를 보낸다:
**(transcript?) → command* → assistant_text → tts_audio?**

**(a) 인식 결과** — 음성 발화였을 때만, "무엇으로 알아들었는지"
```json
{ "type": "transcript", "text": "해밀턴 왜 피트인했어?" }
```

**(b) 명령** — 화면/리플레이 제어 (0개 이상, §4 매핑)
```json
{ "type": "command", "name": "highlightDriver", "args": { "driver_number": 44 } }
```

**(c) 답변 자막**
```json
{ "type": "assistant_text", "text": "해밀턴은 타이어가 닳아서 피트인했어요..." }
```

**(d) 답변 음성** — `TTS_ENABLED`일 때만
```json
{ "type": "tts_audio", "format": "wav", "data": "<base64 wav>" }
```

| type | 필드 | Unity 처리 |
|---|---|---|
| `transcript` | text | (선택) 사용자 발화 자막 표시 |
| `command` | name, args | **§4 매핑대로 실행** |
| `assistant_text` | text | 답변 자막 표시 |
| `tts_audio` | format, data | base64 디코드 → AudioSource 재생 |

---

## 4. 명령(command) ↔ Unity 메서드 매핑 ★핵심

서버가 보내는 `command.name`은 **3종**. `args`는 name마다 다르다.

| name | args | 의미 | Unity 동작(구현 대상) |
|---|---|---|---|
| `loadSession` | `{ "session_key": int }` | 경기 전환 | 해당 세션 리플레이 로드/전환 |
| `highlightDriver` | `{ "driver_number": int }` | 선수 강조 | 그 번호 차량에 마커/하이라이트 |
| `controlReplay` | `{ "action": string, "value": number\|string\|null }` | 재생 제어 | 아래 표대로 분기 |

### 4.1 `controlReplay`의 action별

| action | value | Unity 동작 |
|---|---|---|
| `play` | null | 재생 |
| `pause` | null | 일시정지 |
| `speed` | number (예: 0.5, 2.0) | 배속 변경 |
| `seek` | **시각/숫자** (아래 주의) | 그 지점으로 이동 |

> ⚠️ **seek의 value 두 형태 — 반드시 분기 처리**
> - `control_replay(action="seek", value=<숫자>)` → **상대 시간(초 등)**
> - `jump_to_event(...)`로 발생한 seek → **ISO 절대시각 문자열** (예: `"2025-12-07T15:23:10+00:00"`)
>
> Unity의 seek 핸들러는 **value가 문자열이면** 리플레이 시작 절대시각 기준으로 상대초를 계산해 이동하고,
> **숫자면** 그대로 사용한다. (이 환산 규칙이 AI↔Unity 합의 지점)

### 4.2 명령 발생 예시

- "해밀턴 강조해줘" → `highlightDriver {driver_number:44}`
- "천천히 보여줘" → `controlReplay {action:"speed", value:0.5}`
- "멈춰" → `controlReplay {action:"pause", value:null}`
- "첫 피트스톱 보여줘" → `controlReplay {action:"seek", value:"<ISO시각>"}`
- "2024 모나코 경기 보여줘" → `loadSession {session_key:<찾은 값>}`

---

## 5. 오디오 포맷

### 5.1 입력 (Unity → AI, `audio_utterance.data`)
- **wav 전체(RIFF 헤더 포함)** 를 base64로 인코딩한 문자열.
- 권장: **16kHz, mono, 16-bit PCM**. (STT가 자동 리샘플하지만 16k mono가 안정적)
- 마이크 캡처 → wav 바이트 직렬화 → `Convert.ToBase64String(bytes)`.

### 5.2 출력 (AI → Unity, `tts_audio.data`)
- **wav 전체(헤더 포함)** 의 base64. 샘플레이트 등은 **헤더에 들어 있으니** Unity wav 파서가 그대로 읽으면 된다.
- `Convert.FromBase64String(data)` → wav 파싱 → `AudioClip` 생성 → `AudioSource.PlayOneShot`.

---

## 6. Unity 구현 체크리스트 (F1_XR_Visualizer)

1. **WebSocket 클라이언트** — `ws://…:8001/ws` 연결, JSON 송수신, 재연결 처리.
   - 추천 패키지: `NativeWebSocket` (Quest/Android 호환).
2. **마이크 녹음기** — 발화 캡처 → wav → base64 → `audio_utterance` 전송.
   - 입력 방식: 버튼 눌러 말하기(push-to-talk) 또는 음성 감지 중 택1. 매 전송에 현재 `session_key`/`at_time` 포함.
3. **오디오 재생기** — `tts_audio` 디코드 → `AudioClip` → 재생.
4. **명령 디스패처** — `command.name`을 §4 매핑대로 실제 메서드에 연결. `seek` value 타입 분기 포함.

---

## 7. C# 스켈레톤 (참고용)

```csharp
// ── 메시지 클래스 (protocol.py와 필드명 일치 필수) ──
[Serializable] public class Utterance {
    public string type = "utterance";
    public string text;
    public int session_key;
    public string at_time;
}
[Serializable] public class AudioUtterance {
    public string type = "audio_utterance";
    public string data;          // base64 wav
    public int session_key;
    public string at_time;
}

// 수신 메시지는 먼저 type만 읽고 분기 → 해당 타입으로 다시 역직렬화
void OnMessage(string json) {
    var head = JsonUtility.FromJson<Head>(json);   // { public string type; }
    switch (head.type) {
        case "transcript":     /* 자막 표시 */                       break;
        case "assistant_text": /* 답변 자막 표시 */                   break;
        case "tts_audio":      PlayWavBase64(GetField(json,"data")); break;
        case "command":        OnCommand(json);                      break;
    }
}

void OnCommand(string json) {
    var cmd = ParseCommand(json);   // name + args
    switch (cmd.name) {
        case "loadSession":     replay.Load(cmd.SessionKey());              break;
        case "highlightDriver": carSet.Highlight(cmd.DriverNumber());       break;
        case "controlReplay":   HandleReplay(cmd.Action(), cmd.RawValue()); break;
    }
}

void HandleReplay(string action, object value) {
    switch (action) {
        case "play":  replay.Play();                break;
        case "pause": replay.Pause();               break;
        case "speed": replay.SetSpeed(ToFloat(value)); break;
        case "seek":
            if (value is string iso) replay.SeekAbsolute(DateTime.Parse(iso));
            else                     replay.SeekRelative(ToFloat(value));
            break;
    }
}
```
> `JsonUtility`는 `Dictionary`/동적 타입에 약하므로, `args`는 SimpleJSON·Newtonsoft(Json.NET) 등으로
> 파싱하거나 명령별 전용 DTO를 두는 방식을 권장한다.

---

## 8. 엣지 케이스 / 규칙

- **명령 0개일 수 있음**: 단순 질문("DRS가 뭐야?")은 `command` 없이 `assistant_text`(+`tts_audio`)만 온다.
- **명령 여러 개**: "그 장면 다시 천천히" 같은 복합 요청은 `command`가 여러 개 순서대로 온다. **받은 순서대로 실행**.
- **TTS 없음**: 서버가 `TTS_ENABLED=false`거나 합성 실패 시 `tts_audio`가 안 온다 → **자막만 표시**하면 됨.
- **STT 실패**: 음성이 무음/잡음이면 `transcript`가 안 오고 아무 응답도 없을 수 있다 → 재시도 유도.
- **오류 응답**: 서버 내부 오류 시 `assistant_text`로 "죄송해요, 잠시 문제가 있었어요…"가 온다(연결은 유지).

---

## 9. 계약 동기화 규칙

- 스키마 원본: `F1_XR_AI/app/ws/protocol.py`.
- 필드명/명령명 변경 시 **이 문서 + Python + C# 세 곳을 동시에** 갱신한다.
- 명령 이름은 camelCase (`loadSession`, `highlightDriver`, `controlReplay`).
