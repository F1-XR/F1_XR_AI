"""추월 예측 스모크 테스트 — 실제 (session, at_time, driver)로 features+predict를 돌려 확인.

내 샌드박스에선 lightgbm 설치가 막혀 절대확률까지는 못 냈다. 이 스크립트는
네 컴퓨터(=lightgbm·numpy 설치 + OpenF1 접근 가능)에서 **한 줄로 end-to-end 검증**하게 한다.
피처가 제대로 뽑히고(18/26), 4개 확률이 나오면 predict_overtake 도구도 정상이다.

사전:
    pip install lightgbm numpy          # (voice venv 말고, AI 서버 venv에)
    # F1_XR_Server가 떠 있으면 그 게이트웨이로, 없으면 OpenF1 직결(.env의 f1_server_url)

실행(F1_XR_AI 루트에서):
    python -m scripts.smoke_predict --session 9939 --driver 4
    python -m scripts.smoke_predict --session 9939 --driver 4 --at 2025-07-27T14:40:00+00:00

인자:
    --session  경기 session_key (예: 9939 = 2025 Spa)
    --driver   차량 번호
    --at       리플레이 현재 시각 ISO. 생략하면 최신(스포일러 방지는 at 줄 때만 의미).
"""
from __future__ import annotations

import argparse
import asyncio
import json

from app.ml import features, predict


async def run(session: int, at: str | None, driver: int) -> None:
    feats = await features.build_features(session, at, driver)
    cov = predict.coverage(feats)
    print(f"session={session} driver={driver} at={at or '(최신)'}")
    print(f"계산된 피처: {cov['computed_count']}/{cov['total']}  (나머지는 -1.0 결측)")
    print(json.dumps(feats, indent=2, ensure_ascii=False))

    probs = predict.predict(feats)   # ← 여기서 lightgbm 필요
    print("\n예측 (30초 내 확률):")
    for name, value in probs.items():
        print(f"  {name:32} {value}")


def main() -> None:
    ap = argparse.ArgumentParser(description="추월 예측 스모크 테스트")
    ap.add_argument("--session", type=int, required=True, help="경기 session_key")
    ap.add_argument("--driver", type=int, required=True, help="차량 번호")
    ap.add_argument("--at", default=None, help="리플레이 현재 시각 ISO(생략=최신)")
    args = ap.parse_args()
    asyncio.run(run(args.session, args.at, args.driver))


if __name__ == "__main__":
    main()
