"""Compare one offline training row with features rebuilt through the runtime data path."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from app.ml.features import build_features
from app.ml.predict import predict

FEATURE_ORDER = json.loads(
    (Path(__file__).parents[1] / "app/ml/models/races_initial_event_type_final_unity_contract.json")
    .read_text(encoding="utf-8")
)["feature_order"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--driver", type=int, required=True)
    parser.add_argument("--time", required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.samples)
    target = pd.Timestamp(args.time).tz_localize(None)
    rows = frame[(frame["session_key"] == args.session) & (frame["driver"] == args.driver)].copy()
    rows["distance"] = (pd.to_datetime(rows["t"]) - target).abs()
    row = rows.sort_values("distance").iloc[0]
    runtime = await build_features(args.session, pd.Timestamp(args.time).isoformat(), args.driver)
    comparison = []
    for feature in FEATURE_ORDER:
        offline = row.get(feature)
        online = runtime.get(feature)
        comparison.append({
            "feature": feature,
            "offline": None if pd.isna(offline) else float(offline),
            "runtime": online,
            "absolute_delta": None if online is None or pd.isna(offline) else abs(float(offline) - online),
        })
    comparison.sort(key=lambda item: item["absolute_delta"] or 0, reverse=True)
    print(json.dumps({"prediction": predict(runtime), "features": comparison}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
