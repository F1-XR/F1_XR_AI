"""능동 안내(예측형) — 서버 백그라운드 감시 루프.

7개 도구와 달리 사용자 발화로 발동하지 않는다. 리플레이 상태(heartbeat)를 지켜보다
'곧 추월할 것 같은' 순간을 **추월 예측 모델로** 감지해, Unity로 강조 + "곧 추월 나올 것 같아요"를 push한다.

Unity 규칙형(PointOutWatcher)과의 관계:
  둘 다 켜면 안내가 겹친다. 이걸 켜면(settings.predict_watcher_enabled) Unity 것은 끈다(스위치).
  규칙형=이미 일어난 이벤트 '감지', 예측형=아직 안 일어난 추월 '예측' — 후자가 진짜 예측.

비용:
  - period(기본 3초)마다 1회만 확인(매 프레임 아님).
  - 접전(gap<max_gap) 드라이버로 후보를 좁힌 뒤에만 예측 → 예측 호출 수 최소화.
  - cutoff(at_time) 이하 데이터만 봄 → 스포일러 방지.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from ..config import settings
from ..data import openf1
from . import watcher_eval

logger = logging.getLogger(__name__)

_MAX_GAP = 1.5        # 이 값 이하 gap(초)인 드라이버만 예측 후보(접전)
_MAX_CANDIDATES = 6   # 매 틱 예측할 최대 인원(가장 붙은 순). 과도한 HTTP·예측 방지.


async def _battle_drivers(
    session_key: int, cutoff: str | None,
    max_gap: float = _MAX_GAP, max_candidates: int = _MAX_CANDIDATES,
) -> list[int]:
    """지금(cutoff) 시점에 앞차와 max_gap 이내로 붙은 드라이버 중 '가장 붙은 상위 N명'.
    전원 예측하면 매 틱 HTTP가 폭증(ReadTimeout)하므로 접전 상위 N명으로 제한한다."""
    rows = await openf1.get_intervals(session_key)
    latest: dict[int, tuple[str, float]] = {}
    for r in rows:
        dn, dt, iv = r.get("driver_number"), r.get("date"), r.get("interval")
        if dn is None or not dt:
            continue
        if cutoff and dt > cutoff:          # 미래 데이터 제외(스포일러 방지)
            continue
        try:
            iv = float(iv)
        except (TypeError, ValueError):
            continue
        if dn not in latest or dt > latest[dn][0]:
            latest[dn] = (dt, iv)
    battles = [(dn, iv) for dn, (dt, iv) in latest.items() if 0 < iv < max_gap]
    battles.sort(key=lambda x: x[1])        # 가장 붙은(gap 작은) 순
    return [dn for dn, _iv in battles[:max_candidates]]


async def watch(
    get_state: Callable[[], dict | None],
    announce: Callable[[int, float, str], Awaitable[None]],
) -> None:
    """예측형 능동 안내 루프.

    get_state(): 최신 replay_state dict(없으면 None). heartbeat로 갱신됨.
    announce(driver_number, probability, message): 리본 표시 + TTS 안내를 실제로 보내는
        콜백(main이 주입). probability(0~1)는 Unity가 리본 강도로 쓴다.
    """
    # 예측 모듈은 lightgbm 필요 → 지연 import(미설치여도 서버는 뜬다).
    from ..ml import features as _feat
    from ..ml import predict as _pred

    threshold = settings.watcher_threshold
    period = settings.watcher_period_sec
    cooldown = settings.watcher_cooldown_sec

    last_fire = 0.0
    announced: dict[int, float] = {}   # driver -> 마지막 안내 시각(중복 방지)
    logger.info("[watcher] 예측형 능동 안내 시작 (threshold=%.2f, period=%.1fs)", threshold, period)

    while True:
        await asyncio.sleep(period)
        st = get_state()
        if not st or not st.get("is_playing", True):
            continue
        session = st.get("session_key")
        cutoff = st.get("at_time")
        if session is None or not cutoff:
            continue

        try:
            candidates = await _battle_drivers(session, cutoff)
            best: tuple[int, float] | None = None
            for dn in candidates:
                feats = await _feat.build_features(session, cutoff, dn)
                prob = _pred.predict(feats).get("overtake_probability", 0.0) or 0.0
                if best is None or prob > best[1]:
                    best = (dn, prob)

            if best is None or best[1] < threshold:
                continue

            dn, prob = best
            now = time.monotonic()
            if now - last_fire < cooldown:                     # 전체 쿨다운
                continue
            if now - announced.get(dn, -1e9) < cooldown * 2:   # 같은 차 반복 방지
                continue

            last_fire = now
            announced[dn] = now
            logger.info("[watcher] 능동 안내: %s번 추월확률 %.2f", dn, prob)
            # 오탐 평가용: 이 예측(차량·시각·확률)을 기록 → 나중에 실제 추월 여부와 대조.
            watcher_eval.log_prediction(session, cutoff, dn, prob)
            await announce(dn, prob, f"{dn}번, 곧 추월할 것 같아요!")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[watcher] 예측/안내 실패(계속)")
