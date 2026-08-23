"""Evaluate runtime feature parity on labeled on-track overtakes and hard negatives."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.features import build_features
from app.ml.predict import coverage, predict


def _positive_samples(frame: pd.DataFrame, limit: int) -> list[dict]:
    positives = frame[
        (frame["event_type"] == 1) & (frame["is_lap1"] == 0)
    ].sort_values(["driver", "t"])
    samples: list[dict] = []
    for driver, group in positives.groupby("driver"):
        group = group.sort_values("t")
        blocks = (group["t"].diff().dt.total_seconds().fillna(999) > 1.5).cumsum()
        for _, block in group.groupby(blocks):
            # Roughly ten seconds before the labeled window ends (the pass boundary).
            row = block.iloc[max(0, len(block) - 11)]
            samples.append({"kind": "positive", "driver": int(driver), "t": row["t"]})
    samples.sort(key=lambda row: row["t"])
    if len(samples) <= limit:
        return samples
    indices = np.linspace(0, len(samples) - 1, limit).astype(int)
    return [samples[i] for i in indices]


def _negative_samples(frame: pd.DataFrame, limit: int) -> list[dict]:
    negatives = frame[
        (frame["event_type"] == 0)
        & (frame["gap_ahead"] > 0)
        & (frame["gap_ahead"] <= 1.0)
        & (frame["gap_trend"] < 0)
        & (frame["same_lap"] == 1)
        & (frame["pit_window"] == 0)
        & (frame["is_lap1"] == 0)
    ].sort_values("t")
    if negatives.empty:
        return []
    indices = np.linspace(0, len(negatives) - 1, min(limit, len(negatives))).astype(int)
    return [
        {"kind": "negative", "driver": int(negatives.iloc[i]["driver"]), "t": negatives.iloc[i]["t"]}
        for i in indices
    ]


async def evaluate_one(session: int, sample: dict, semaphore: asyncio.Semaphore) -> dict:
    at_time = sample["t"].isoformat() + "+00:00"
    async with semaphore:
        feats = await build_features(session, at_time, sample["driver"])
    probs = predict(feats)
    return {
        "kind": sample["kind"],
        "driver": sample["driver"],
        "at_time": at_time,
        "feature_count": coverage(feats)["computed_count"],
        "probability": probs["overtake_probability"],
    }


async def run(args: argparse.Namespace) -> dict:
    frame = pd.read_parquet(args.samples)
    if "session_key" in frame.columns:
        frame = frame[frame["session_key"] == args.session].copy()
    chosen = _positive_samples(frame, args.positives) + _negative_samples(frame, args.negatives)
    semaphore = asyncio.Semaphore(args.concurrency)
    rows = await asyncio.gather(*(evaluate_one(args.session, row, semaphore) for row in chosen))

    summary: dict = {"session": args.session, "samples": rows, "thresholds": {}}
    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    summary["positive_mean"] = float(np.mean([row["probability"] for row in positives])) if positives else None
    summary["negative_mean"] = float(np.mean([row["probability"] for row in negatives])) if negatives else None
    for threshold in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        summary["thresholds"][str(threshold)] = {
            "event_recall": sum(row["probability"] >= threshold for row in positives) / len(positives) if positives else None,
            "negative_fire_rate": sum(row["probability"] >= threshold for row in negatives) / len(negatives) if negatives else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--positives", type=int, default=12)
    parser.add_argument("--negatives", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
