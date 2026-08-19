"""추월 예측 리서치 요약 — held-out 평가 리포트(JSON)를 표로 재현한다.

`evaluate_external_race.py`가 만든 리포트(예: results/external_eval_spa_2025_report.json)를
읽어 포트폴리오/논문용 표를 뽑는다. 모델을 다시 돌리지 않고, 이미 저장된 평가 결과에서
아래를 계산·정리한다.

  - 전체 원본 샘플 수 / positive·negative / positive_rate
  - strict quality filtering 전후 positive 비율
  - 어떤 예외처리(exclusion)가 가장 많은 샘플을 제거했는지 (내림차순)
  - held-out(미학습 서킷) ROC-AUC / PR-AUC
  - threshold별 Precision / Recall / F1 + Confusion Matrix(TP/FP/FN/TN)
  - 누수 검증(leakage check) 요약

리포트는 F1_XR_overtakeML 레포에서 생성된다. 이 스크립트는 어느 레포에서 실행해도
되도록 경로만 인자로 받는다.

실행:
  python -m scripts.analyze_overtake_report \\
      --report ../F1_XR_overtakeML/results/external_eval_spa_2025_report.json \\
      [--target label_overtake] [--markdown out.md]

  # 4개 타깃 전부 요약
  python -m scripts.analyze_overtake_report --report <경로> --target all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = [
    "label_overtake",
    "label_position_gain",
    "label_position_loss",
    "label_position_change",
]


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _confusion_rows(target_block: dict) -> list[dict]:
    """threshold별 confusion matrix + F1 계산.

    리포트의 thresholds는 (predicted_positive, true_positive, false_positive,
    precision, recall)만 담는다. FN/TN은 전체 개수에서 복원한다.

    ⚠️ threshold 평가는 strict filtering '후'(external_quality_filtered) 집합에서
    이뤄진다(recall = TP / filtered_positive 로 검증됨). 따라서 FN/TN도 filtered
    집합의 positive/negative 로 복원해야 한다(raw 집합으로 하면 TN이 틀어진다).
    """
    filt = target_block["external_quality_filtered"]
    positive = filt["positive"]
    negative = filt["negative"]

    rows = []
    for th in target_block.get("thresholds", []):
        tp = th["true_positive"]
        fp = th["false_positive"]
        fn = positive - tp
        tn = negative - fp
        precision = th["precision"]
        recall = th["recall"]
        rows.append({
            "threshold": th["display_threshold"],
            "predicted_positive": th["predicted_positive"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        })
    return rows


def summarize_target(name: str, block: dict) -> list[str]:
    lines: list[str] = []
    allrows = block["external_all_rows"]
    filt = block["external_quality_filtered"]
    qf = block["quality_filter"]

    lines.append(f"## {name}")
    lines.append("")
    lines.append("### 데이터 규모 / 클래스 불균형")
    lines.append("")
    lines.append("| 구분 | 전체 rows | positive | negative | positive rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| 원본(raw) | {allrows['rows']:,} | {allrows['positive']:,} | "
        f"{allrows['negative']:,} | {_fmt_pct(allrows['positive_rate'])} |"
    )
    lines.append(
        f"| strict filtering 후 | {filt['rows']:,} | {filt['positive']:,} | "
        f"{filt['negative']:,} | {_fmt_pct(filt['positive_rate'])} |"
    )
    lines.append("")
    lines.append(
        f"strict filtering: {qf['train_rows_before']:,} → {qf['train_rows_after']:,} rows "
        f"({qf['excluded_rows']:,} 제거), positive rate "
        f"{_fmt_pct(qf['train_positive_rate_before'])} → {_fmt_pct(qf['train_positive_rate_after'])}"
    )
    lines.append("")

    excluded = sorted(qf.get("excluded_groups", []), key=lambda g: g["rows"], reverse=True)
    if excluded:
        lines.append("### 예외처리별 제거 샘플 (많이 제거한 순)")
        lines.append("")
        lines.append("| 예외처리(reason) | 제거 rows | 제거 positive |")
        lines.append("| --- | ---: | ---: |")
        for g in excluded:
            lines.append(f"| {g['reason']} | {g['rows']:,} | {g['positive']:,} |")
        lines.append("")

    oof = block.get("model_oof_raw_metrics", {})
    lines.append("### held-out(미학습 서킷) 성능")
    lines.append("")
    lines.append("| 지표 | OOF(교차검증) | held-out(strict filtering 후) |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| ROC-AUC | {oof.get('roc_auc', float('nan')):.3f} | {filt['raw_roc_auc']:.3f} |"
    )
    lines.append(
        f"| PR-AUC | {oof.get('pr_auc', float('nan')):.3f} | {filt['raw_pr_auc']:.3f} |"
    )
    lines.append("")

    cm = _confusion_rows(block)
    if cm:
        lines.append("### threshold별 Precision / Recall / F1 + Confusion Matrix")
        lines.append("")
        lines.append("| threshold | TP | FP | FN | TN | Precision | Recall | F1 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in cm:
            lines.append(
                f"| {r['threshold']:.2f} | {r['tp']:,} | {r['fp']:,} | {r['fn']:,} | "
                f"{r['tn']:,} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
            )
        best = max(cm, key=lambda r: r["f1"])
        lines.append("")
        lines.append(
            f"최고 F1: **{best['f1']:.3f}** (threshold={best['threshold']:.2f}, "
            f"Precision={best['precision']:.3f}, Recall={best['recall']:.3f})"
        )
        lines.append("")
    return lines


def summarize(report: dict, target: str) -> str:
    lines: list[str] = []
    lines.append(f"# 추월 예측 held-out 평가 요약 — `{report.get('model_run', '?')}`")
    lines.append("")

    leak = report.get("leakage_check", {})
    lines.append("## 누수 검증 (leakage check)")
    lines.append("")
    lines.append(
        f"- 평가 rows: {report.get('external_rows', 0):,} "
        f"({', '.join(report.get('external_circuits', []))}, {report.get('external_years', [])})"
    )
    lines.append(
        f"- 학습/평가 세션 겹침: {len(leak.get('overlap_session_keys', []))}건 "
        f"(학습 {len(leak.get('training_session_keys', []))}세션 / "
        f"평가 {len(leak.get('external_session_keys', []))}세션)"
    )
    lines.append(
        f"- 평가 서킷이 학습에 등장: "
        f"{len(leak.get('external_circuit_seen_in_training', []))}건 "
        f"(학습 서킷키 {leak.get('training_circuit_keys', [])}, "
        f"평가 서킷키 {leak.get('external_circuit_keys', [])})"
    )
    lines.append("")

    targets = TARGETS if target == "all" else [target]
    for name in targets:
        block = report.get("targets", {}).get(name)
        if block is None:
            lines.append(f"> (리포트에 {name} 없음)")
            continue
        lines.extend(summarize_target(name, block))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        default="../F1_XR_overtakeML/results/external_eval_spa_2025_report.json",
        help="evaluate_external_race.py가 만든 리포트 JSON 경로",
    )
    ap.add_argument("--target", default="label_overtake",
                    help="label_overtake | label_position_gain | label_position_loss | "
                         "label_position_change | all")
    ap.add_argument("--markdown", default=None, help="결과를 저장할 .md 경로(옵션)")
    args = ap.parse_args()

    path = Path(args.report)
    if not path.exists():
        raise SystemExit(f"리포트를 찾을 수 없어요: {path}\n"
                         f"→ F1_XR_overtakeML 에서 evaluate_external_race.py 를 먼저 실행하세요.")

    report = json.loads(path.read_text(encoding="utf-8"))
    md = summarize(report, args.target)
    print(md)
    if args.markdown:
        Path(args.markdown).write_text(md, encoding="utf-8")
        print(f"\n[저장] {args.markdown}")


if __name__ == "__main__":
    main()
