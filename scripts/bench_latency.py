"""실시간 멀티모달 파이프라인 지연(latency) 벤치마크 — 단계별 p50/p95.

STT → LLM Agent(툴 선택/호출) → TTS → End-to-End 각 단계의 지연을 여러 번 측정해
평균/중앙값(p50)/p95 를 표로 뽑는다. "GPT/LLM 안내 튜토리얼"을 "각 구성요소의 latency를
측정한 실시간 멀티모달 AI 파이프라인"으로 바꿔 말할 수 있게 하는 근거 수치를 만든다.

측정 단계:
  - stt         : 음성 → 텍스트 (--stt-wav 를 준 경우에만)
  - agent_total : run_agent 전체 (라우팅 + 도구 실행 + LLM 응답)
  - agent_llm   : 그 중 LLM/툴 호출 구간 (app/obs 트레이스, react 경로에서만 값 존재)
  - tts         : 텍스트 → 음성 wav
  - end_to_end  : stt + agent_total + tts (사용자 체감 지연)

실행:
  python -m scripts.bench_latency                          # 기본 프롬프트, 5회 반복
  python -m scripts.bench_latency --repeats 30             # 30회 반복(p95 신뢰도↑)
  python -m scripts.bench_latency --prompts-file q.txt     # 프롬프트를 파일에서(줄당 1개)
  python -m scripts.bench_latency --stt-wav tts_out.wav    # STT까지 포함해 측정
  python -m scripts.bench_latency --no-tts                 # TTS 제외(에이전트만)

주의:
  - 첫 호출은 STT/TTS 모델 로드로 느리다 → --warmup(기본 1)회는 통계에서 제외한다.
  - .env 의 LLM_MODEL/LLM_BASE_URL 이 현재 측정 대상 모델이다(모델 바꿔 재측정해 비교).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import app.obs as obs
from app.agent.graph import run_agent
from app.config import settings
from app.voice import stt as stt_mod
from app.voice import tts as tts_mod

# 여러 도구 경로를 골고루 자극하는 기본 프롬프트(개념/조회/제어/예측/상황해설).
DEFAULT_PROMPTS = [
    "DRS가 뭐야?",
    "지금 누가 1등이야?",
    "지금 상황 어때?",
    "앞차랑 얼마나 붙었어?",
    "저 선수 곧 추월할 것 같아?",
    "첫 추월 장면 보여줘",
    "천천히 보여줘",
]


def _percentile(samples: list[float], pct: float) -> float:
    """nearest-rank 백분위(정렬 후 위치). 표본이 적어도 안전하게 동작."""
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}"


async def _measure_once(
    prompt: str,
    session_key: int | None,
    stt_audio: bytes | None,
    measure_tts: bool,
) -> dict:
    """한 프롬프트 1회 측정 → 단계별 초(seconds) dict."""
    result: dict = {"prompt": prompt}

    stt_s = 0.0
    if stt_audio is not None:
        t0 = time.perf_counter()
        await stt_mod.transcribe(stt_audio, language="ko")
        stt_s = time.perf_counter() - t0
        result["stt"] = stt_s

    obs.start_trace()   # 이 발화의 계측 트레이스 시작(agent_llm/path/tool_calls 기록)
    t0 = time.perf_counter()
    reply, commands, ok = await run_agent(prompt, session_key=session_key)
    agent_total = time.perf_counter() - t0
    trace = obs.current()
    result["agent_total"] = agent_total
    result["path"] = trace.path if trace else None
    if trace and "agent_llm" in trace.timings:
        result["agent_llm"] = trace.timings["agent_llm"]
    result["tool_calls"] = trace.tool_names if trace else []
    result["reply_chars"] = len(reply)
    result["ok"] = ok

    tts_s = 0.0
    if measure_tts:
        t0 = time.perf_counter()
        await tts_mod.synthesize(reply)
        tts_s = time.perf_counter() - t0
        result["tts"] = tts_s

    result["end_to_end"] = stt_s + agent_total + tts_s
    return result


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts-file", default=None, help="프롬프트 파일(줄당 1개). 없으면 기본셋")
    ap.add_argument("--repeats", type=int, default=5, help="각 프롬프트 반복 횟수")
    ap.add_argument("--warmup", type=int, default=1, help="통계에서 제외할 워밍업 라운드 수")
    ap.add_argument("--session-key", type=int, default=None, help="run_agent에 넘길 세션(옵션)")
    ap.add_argument("--stt-wav", default=None, help="STT까지 측정할 입력 wav 경로(옵션)")
    ap.add_argument("--no-tts", action="store_true", help="TTS 단계 측정 제외")
    ap.add_argument("--out", default="results/latency_report.json", help="JSON 리포트 저장 경로")
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        lines = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()]
        prompts = [ln for ln in lines if ln]

    stt_audio = None
    if args.stt_wav:
        stt_audio = Path(args.stt_wav).read_bytes()

    measure_tts = (not args.no_tts) and settings.tts_enabled

    print(f"모델: LLM={settings.llm_model!r}  base_url={settings.llm_base_url or 'OpenAI'!r}")
    print(f"STT={settings.stt_provider!r}  TTS={settings.tts_provider!r} (measure_tts={measure_tts})")
    print(f"프롬프트 {len(prompts)}개 × 반복 {args.repeats} (워밍업 {args.warmup} 제외)\n")

    # 워밍업: 모델 로드/캐시가 통계를 오염시키지 않도록 몇 라운드 버린다.
    for _ in range(max(0, args.warmup)):
        await _measure_once(prompts[0], args.session_key, stt_audio, measure_tts)

    samples: dict[str, list[float]] = {}
    per_path: dict[str, int] = {}
    raw_rows: list[dict] = []
    for r in range(args.repeats):
        for prompt in prompts:
            row = await _measure_once(prompt, args.session_key, stt_audio, measure_tts)
            raw_rows.append(row)
            path = row.get("path") or "unknown"
            per_path[path] = per_path.get(path, 0) + 1
            for stage in ("stt", "agent_total", "agent_llm", "tts", "end_to_end"):
                if stage in row:
                    samples.setdefault(stage, []).append(row[stage])

    stage_order = [s for s in ("stt", "agent_total", "agent_llm", "tts", "end_to_end") if s in samples]
    stage_ko = {
        "stt": "STT (음성→텍스트)",
        "agent_total": "Agent 전체 (라우팅+툴+LLM)",
        "agent_llm": "Agent LLM/툴 실행",
        "tts": "TTS (텍스트→음성)",
        "end_to_end": "End-to-End",
    }

    print("| 단계 | n | 평균(ms) | p50(ms) | p95(ms) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    report_stages = {}
    for stage in stage_order:
        vals = samples[stage]
        mean = statistics.fmean(vals)
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        report_stages[stage] = {
            "n": len(vals),
            "mean_ms": mean * 1000,
            "p50_ms": p50 * 1000,
            "p95_ms": p95 * 1000,
        }
        print(f"| {stage_ko[stage]} | {len(vals)} | {_fmt_ms(mean)} | {_fmt_ms(p50)} | {_fmt_ms(p95)} |")

    print(f"\n경로 분포(path): {per_path}")
    if "end_to_end" in report_stages:
        e = report_stages["end_to_end"]
        print(f"\n요약: End-to-End 평균 {e['mean_ms'] / 1000:.2f}s / p95 {e['p95_ms'] / 1000:.2f}s "
              f"({report_stages['end_to_end']['n']}회 기준)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": settings.llm_model,
        "base_url": settings.llm_base_url or "openai",
        "stt_provider": settings.stt_provider,
        "tts_provider": settings.tts_provider,
        "measure_tts": measure_tts,
        "repeats": args.repeats,
        "prompts": prompts,
        "path_distribution": per_path,
        "stages": report_stages,
        "raw": raw_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
