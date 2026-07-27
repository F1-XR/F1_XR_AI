"""서버 /ws WebSocket 왕복 테스트 클라이언트 (Unity 대역).

진짜 마이크·Unity 없이, wav 파일을 audio_utterance로 서버에 보내고
서버가 돌려주는 메시지(transcript / command / assistant_text / tts_audio)를 순서대로 출력한다.
tts_audio가 오면 ws_reply.wav 로 저장한다.

→ "서버 쪽 WS 프로토콜이 정상인지"를 Unity 붙이기 전에 미리 검증하는 용도.
   실기기에서 안 될 때 '서버 문제냐 Unity 문제냐'를 빨리 가르는 기준이 된다.

실행:
  # (기본) 음성 발화: tts_out.wav 를 STT→에이전트→TTS 왕복
  python -m scripts.ws_client_test
  python -m scripts.ws_client_test 내질문.wav

  # 텍스트 발화(마이크/STT 없이 에이전트만)
  python -m scripts.ws_client_test --text "해밀턴 왜 피트인했어?"

  # 옵션: 현재 보고 있는 경기/시각을 함께 전달
  python -m scripts.ws_client_test --session 9839 --at-time 2025-12-07T15:20:00+00:00

준비:
  pip install websockets          # 클라이언트 라이브러리
  (그리고 다른 터미널에서 서버 실행: uvicorn app.main:app --port 8001)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json

try:
    import websockets
except ImportError:
    raise SystemExit("websockets 미설치 → pip install websockets")


async def run(url: str, payload: dict, first_timeout: float, grace: float) -> None:
    print(f"연결: {url}")
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(payload))
        if payload["type"] == "utterance":
            print(f"→ 전송: utterance text={payload.get('text')!r}")
        else:
            print(f"→ 전송: audio_utterance (base64 {len(payload['data']):,} chars)")
        print("← 응답 대기… (첫 응답은 모델 로드로 느릴 수 있어요)\n")

        got_text = False
        n_cmd = 0
        while True:
            timeout = grace if got_text else first_timeout
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                print("\n(더 이상 메시지 없음 — 턴 종료)")
                break

            msg = json.loads(raw)
            t = msg.get("type")
            if t == "transcript":
                print(f"[transcript] 인식: {msg.get('text')!r}")
            elif t == "command":
                n_cmd += 1
                print(f"[command]    {msg.get('name')}  args={msg.get('args')}")
            elif t == "assistant_text":
                got_text = True
                print(f"[assistant]  {msg.get('text')}")
            elif t == "tts_audio":
                audio = base64.b64decode(msg.get("data", ""))
                out = "ws_reply.wav"
                with open(out, "wb") as f:
                    f.write(audio)
                print(f"[tts_audio]  {len(audio):,} bytes → {out} 저장 (재생: start {out})")
                break   # tts_audio 는 보통 그 턴의 마지막 메시지
            else:
                print(f"[{t}] {msg}")

        print(f"\n요약: 명령 {n_cmd}개 · 자막 {'수신' if got_text else '없음'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="서버 /ws 왕복 테스트 (Unity 대역)")
    ap.add_argument("wav", nargs="?", default="tts_out.wav",
                    help="보낼 wav 경로 (audio_utterance). 기본 tts_out.wav")
    ap.add_argument("--text", help="wav 대신 텍스트 발화(utterance)로 보냄")
    ap.add_argument("--url", default="ws://localhost:8001/ws", help="서버 WS 주소")
    ap.add_argument("--session", type=int, default=None, help="session_key(현재 보는 경기)")
    ap.add_argument("--at-time", default=None, help="리플레이 현재 시각(ISO)")
    ap.add_argument("--first-timeout", type=float, default=120.0,
                    help="첫 응답 대기(초). 모델 첫 로드로 느릴 수 있음")
    ap.add_argument("--grace", type=float, default=8.0,
                    help="자막 수신 후 tts_audio 등 추가 메시지 대기(초)")
    args = ap.parse_args()

    if args.text:
        payload: dict = {"type": "utterance", "text": args.text}
    else:
        try:
            with open(args.wav, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise SystemExit(
                f"wav 없음: {args.wav} — `python -m scripts.tts_test` 로 tts_out.wav 를 먼저 만드세요."
            )
        payload = {"type": "audio_utterance", "data": base64.b64encode(data).decode("ascii")}

    if args.session is not None:
        payload["session_key"] = args.session
    if args.at_time:
        payload["at_time"] = args.at_time

    asyncio.run(run(args.url, payload, args.first_timeout, args.grace))


if __name__ == "__main__":
    main()
