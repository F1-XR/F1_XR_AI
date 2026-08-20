"""Export a compact calibration summary for the overtake ML outputs.

This is a lightweight evidence artifact for the AI portfolio. It reads the model
metadata JSON files already shipped with the AI service and reports raw vs
display-calibrated Brier score changes.

Examples:
  python -m scripts.export_calibration_summary
  python -m scripts.export_calibration_summary --json-out calibration_summary.json
  python -m scripts.export_calibration_summary --markdown-out calibration_summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS_DIR = Path(__file__).resolve().parents[1] / "app" / "ml" / "models"
RUN = "races_initial_event_type_final"
TARGETS = [
    "label_overtake",
    "label_position_gain",
    "label_position_loss",
    "label_position_change",
]


def _pct_improvement(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return 100 * (1 - after / before)


def _load_target(target: str) -> dict[str, Any]:
    path = MODELS_DIR / f"{RUN}_{target}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    model = json.loads(path.read_text(encoding="utf-8"))
    raw = model.get("raw_metrics", model.get("model_oof_raw_metrics", {}))
    display = model.get("display_metrics", model.get("model_oof_display_metrics", {}))
    calibration = model.get("calibration", {})
    raw_brier = raw.get("brier")
    display_brier = display.get("brier")
    return {
        "target": target,
        "metadata_file": str(path),
        "calibration_method": calibration.get("method"),
        "raw": {
            "roc_auc": raw.get("roc_auc"),
            "pr_auc": raw.get("pr_auc"),
            "brier": raw_brier,
        },
        "display_calibrated": {
            "roc_auc": display.get("roc_auc"),
            "pr_auc": display.get("pr_auc"),
            "brier": display_brier,
        },
        "brier_improvement_pct": _pct_improvement(raw_brier, display_brier),
    }


def build_summary(targets: list[str]) -> dict[str, Any]:
    outputs = [_load_target(target) for target in targets]
    return {
        "run": RUN,
        "models_dir": str(MODELS_DIR),
        "outputs": outputs,
    }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Overtake ML Calibration Summary",
        "",
        f"- run: `{summary['run']}`",
        "- method: compare raw model probability against display-calibrated probability",
        "- metric: Brier score, lower is better",
        "",
        "| output | method | raw Brier | calibrated Brier | Brier improvement | raw ROC-AUC | calibrated ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for output in summary["outputs"]:
        improvement = output["brier_improvement_pct"]
        lines.append(
            "| {target} | {method} | {raw_brier} | {display_brier} | {improvement} | {raw_auc} | {display_auc} |".format(
                target=output["target"],
                method=output.get("calibration_method") or "-",
                raw_brier=_fmt(output["raw"]["brier"]),
                display_brier=_fmt(output["display_calibrated"]["brier"]),
                improvement="-" if improvement is None else f"{improvement:+.1f}%",
                raw_auc=_fmt(output["raw"]["roc_auc"], 3),
                display_auc=_fmt(output["display_calibrated"]["roc_auc"], 3),
            )
        )
    lines.extend(
        [
            "",
            "Recommended use: this is enough for a portfolio/debug dashboard capture. A live XR calibration dashboard can be deferred unless the project needs a dedicated AI diagnostics screen.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="all", help="One target name or all.")
    parser.add_argument("--json-out", default=None, help="Write summary JSON.")
    parser.add_argument("--markdown-out", default=None, help="Write summary markdown.")
    args = parser.parse_args()

    targets = TARGETS if args.target == "all" else [args.target]
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise SystemExit(f"unknown target(s): {', '.join(unknown)}")

    summary = build_summary(targets)
    print(to_markdown(summary))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[json] {out}")
    if args.markdown_out:
        out = Path(args.markdown_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_markdown(summary), encoding="utf-8")
        print(f"[markdown] {out}")


if __name__ == "__main__":
    main()
