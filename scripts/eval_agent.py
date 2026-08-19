"""에이전트 Tool 선택 평가 — LLM이 발화별로 '올바른 도구를 올바른 인자로' 부르는지 채점.

경량 모델(예: Gemma 계열)로 바꿨을 때 Tool selection 성능이 얼마나 떨어지는지를
숫자로 보여주기 위한 평가 하니스다. 평가셋(eval/agent_tool_eval.jsonl)의 각 발화를
run_agent로 실행하고, app/obs 트레이스에 기록된 실제 tool_calls 를 정답과 대조한다.

채점 항목(피드백 5기준):
  1. tool_selection : 기대한 primary 도구를 호출했는가
  2. missing_tool   : 기대 도구를 누락했는가 (= tool_selection 실패)
  3. tool_args      : 호출한 기대 도구의 필수 인자가 맞는가
  4. unnecessary    : 기대·허용 목록 밖의 도구를 불필요하게 호출했는가
  5. final_success  : 최종 한국어 답변을 냈는가(ok=True, 비어있지 않음)

⚠️ 공정한 '모델 tool-selection' 비교를 위해, 기본적으로 결정적 라우터(planner/rule_router)를
   끈다(--with-router 로 켤 수 있음). 라우터가 켜져 있으면 LLM이 아니라 규칙이 도구를
   고르므로 모델 성능 차이가 가려진다.

실행:
  # 모델 A(.env 의 LLM_MODEL) 로 평가 → 라벨 지정해 저장
  python -m scripts.eval_agent --label gpt --session-key 9839

  # .env 를 모델 B로 바꾼 뒤 다시
  python -m scripts.eval_agent --label gemma --session-key 9839

  # 두 결과를 표로 비교
  python -m scripts.eval_agent --compare results/agent_eval_gpt.json results/agent_eval_gemma.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import app.obs as obs
from app.agent.graph import run_agent
from app.config import settings

EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "agent_tool_eval.jsonl"


def _load_items(path: Path) -> list[dict]:
    items = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            items.append(json.loads(ln))
    return items


def _arg_matches(expected, actual) -> bool:
    """기대 인자 값 매칭. None=키 존재만 확인, 문자열=부분/대소문자 무시 일치, 그 외=동등."""
    if expected is None:
        return actual is not None
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() in actual.lower() or actual.lower() in expected.lower()
    try:
        return int(expected) == int(actual)
    except (TypeError, ValueError):
        return expected == actual


def _score_item(item: dict, called: list[dict], reply_ok: bool) -> dict:
    called_names = [c["name"] for c in called]
    expected = item.get("expected_tools", [])
    acceptable = set(item.get("acceptable_tools", [])) | set(expected)
    expected_args = item.get("expected_args", {})

    # 1) tool_selection / missing : 기대 도구가 모두 호출됐는가
    missing = [t for t in expected if t not in called_names]
    tool_selection = len(missing) == 0

    # 3) tool_args : 호출된 기대 도구의 필수 인자가 맞는가
    args_ok = True
    arg_detail = []
    for tool, want in expected_args.items():
        # 그 도구의 마지막 호출 인자를 검사
        calls = [c for c in called if c["name"] == tool]
        if not calls:
            args_ok = False
            arg_detail.append(f"{tool}:not_called")
            continue
        got = calls[-1].get("args") or {}
        for key, exp_val in want.items():
            if not _arg_matches(exp_val, got.get(key)):
                args_ok = False
                arg_detail.append(f"{tool}.{key}={got.get(key)!r}!=~{exp_val!r}")

    # 4) unnecessary : 기대·허용 밖 도구 호출
    unnecessary = sorted({n for n in called_names if n not in acceptable})

    return {
        "id": item["id"],
        "category": item.get("category"),
        "utterance": item["utterance"],
        "called_tools": called_names,
        "tool_selection": tool_selection,
        "missing_tools": missing,
        "tool_args_ok": args_ok,
        "arg_detail": arg_detail,
        "unnecessary_tools": unnecessary,
        "final_success": bool(reply_ok),
    }


async def _run_eval(items: list[dict], session_key: int | None) -> list[dict]:
    results = []
    for item in items:
        obs.start_trace()
        try:
            reply, commands, ok = await run_agent(
                item["utterance"],
                session_key=session_key,
                selected_driver=item.get("selected_driver"),
            )
            reply_ok = bool(ok and reply)
        except Exception as exc:   # 모델/데이터 오류도 '실패'로 기록(평가 중단 안 함)
            reply_ok = False
            print(f"  [error] {item['id']}: {type(exc).__name__}: {str(exc)[:120]}")
        trace = obs.current()
        called = trace.tool_calls if trace else []
        row = _score_item(item, called, reply_ok)
        results.append(row)
        mark = "O" if row["tool_selection"] and row["tool_args_ok"] and not row["unnecessary_tools"] else "X"
        print(f"  [{mark}] {row['id']:22s} called={row['called_tools']}")
    return results


def _aggregate(results: list[dict]) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "tool_selection_rate": sum(r["tool_selection"] for r in results) / n,
        "tool_args_rate": sum(r["tool_args_ok"] for r in results) / n,
        "missing_rate": sum(bool(r["missing_tools"]) for r in results) / n,
        "unnecessary_rate": sum(bool(r["unnecessary_tools"]) for r in results) / n,
        "final_success_rate": sum(r["final_success"] for r in results) / n,
    }


def _print_summary(label: str, agg: dict) -> None:
    print(f"\n=== 에이전트 Tool 선택 평가 — {label} (n={agg['n']}) ===")
    print(f"Tool selection  : {agg['tool_selection_rate'] * 100:5.1f}%")
    print(f"Tool args 정확  : {agg['tool_args_rate'] * 100:5.1f}%")
    print(f"누락(missing)   : {agg['missing_rate'] * 100:5.1f}%")
    print(f"불필요 호출     : {agg['unnecessary_rate'] * 100:5.1f}%")
    print(f"최종 답변 성공  : {agg['final_success_rate'] * 100:5.1f}%")


def _compare(paths: list[str]) -> None:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print("\n| Model | Tool selection | Tool args | Missing | Unnecessary | Final success |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for rep in reports:
        a = rep["aggregate"]
        print(
            f"| {rep['label']} | {a['tool_selection_rate'] * 100:.0f}% | "
            f"{a['tool_args_rate'] * 100:.0f}% | {a['missing_rate'] * 100:.0f}% | "
            f"{a['unnecessary_rate'] * 100:.0f}% | {a['final_success_rate'] * 100:.0f}% |"
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-set", default=str(EVAL_SET), help="평가셋 jsonl 경로")
    ap.add_argument("--label", default=None, help="이 실행의 모델 라벨(예: gpt / gemma). 저장 파일명에 사용")
    ap.add_argument("--session-key", type=int, default=None, help="run_agent에 넘길 세션(데이터 필요한 도구용)")
    ap.add_argument("--with-router", action="store_true",
                    help="결정적 planner/rule_router를 끄지 않는다(기본은 끔 — 순수 LLM tool selection 측정)")
    ap.add_argument("--out-dir", default="results", help="리포트 저장 폴더")
    ap.add_argument("--compare", nargs="+", default=None, help="여러 리포트 JSON을 비교표로 출력하고 종료")
    args = ap.parse_args()

    if args.compare:
        _compare(args.compare)
        return

    if not args.with_router:
        # 순수 LLM tool-selection 측정: 규칙 라우터를 끈다.
        settings.command_planner_enabled = False
        settings.demo_rule_router_enabled = False

    label = args.label or settings.llm_model
    items = _load_items(Path(args.eval_set))
    print(f"모델: {settings.llm_model!r}  base_url={settings.llm_base_url or 'OpenAI'!r}")
    print(f"라우터 사용: {args.with_router}  |  평가 {len(items)}문항  |  세션={args.session_key}\n")

    results = await _run_eval(items, args.session_key)
    agg = _aggregate(results)
    _print_summary(label, agg)

    out = Path(args.out_dir) / f"agent_eval_{label}.json".replace("/", "_")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": label,
        "model": settings.llm_model,
        "base_url": settings.llm_base_url or "openai",
        "with_router": args.with_router,
        "aggregate": agg,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    asyncio.run(main())
