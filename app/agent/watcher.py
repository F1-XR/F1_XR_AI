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
from datetime import datetime, timedelta
from typing import Awaitable, Callable

from ..config import settings
from ..data import openf1
from . import watcher_eval

logger = logging.getLogger(__name__)

_MAX_GAP = 1.0        # 이 값 이하 gap(초)인 드라이버만 예측 후보(접전)
_MAX_CANDIDATES = 8   # 데모 구간의 짧은 추월 직전 후보를 놓치지 않도록 넉넉히 본다.


def _parse_iso(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_from_session_hour(cutoff: str | None) -> float | None:
    dt = _parse_iso(cutoff)
    if dt is None:
        return None
    return (dt - dt.replace(minute=0, second=0, microsecond=0)).total_seconds()


def _shift_iso(iso: str | None, seconds: float) -> str | None:
    dt = _parse_iso(iso)
    if dt is None:
        return iso
    return (dt + timedelta(seconds=seconds)).isoformat()


def _latest_position_at(rows: list[dict], driver_number: int, at_time: datetime) -> int | None:
    latest: tuple[datetime, int] | None = None
    for r in rows:
        if r.get("driver_number") != driver_number or r.get("position") is None:
            continue
        dt = _parse_iso(r.get("date"))
        if dt is None or dt > at_time:
            continue
        pos = int(r["position"])
        if latest is None or dt > latest[0]:
            latest = (dt, pos)
    return latest[1] if latest is not None else None


def _first_position_gain_event(
    rows: list[dict], driver_number: int, start: datetime, end: datetime
) -> tuple[str, int, int] | None:
    samples = [
        r for r in rows
        if r.get("driver_number") == driver_number
        and r.get("position") is not None
        and _parse_iso(r.get("date")) is not None
    ]
    samples.sort(key=lambda r: r["date"])
    prev = None
    for r in samples:
        dt = _parse_iso(r["date"])
        if dt is None or dt < start or dt > end:
            continue
        pos = int(r["position"])
        if prev is not None and pos < prev:
            return r["date"], prev, pos
        prev = pos
    return None


async def _confirmed_gain_candidates(
    session_key: int,
    cutoff: str | None,
    candidates: list[dict],
    window_sec: float = 30.0,
    lookback_sec: float = 8.0,
) -> list[dict]:
    """Replay 전용 보정: 실제 순위가 좋아지는 차를 우선한다.

    모델 점수만으로 같은 시각의 다른 접전 차량을 고르는 false positive를 줄이기 위한
    historical replay 필터다. 고속 재생/서버 지연으로 추월 직전 창을 몇 초 놓쳐도
    방금 일어난 position gain을 같은 데모 장면으로 복구한다.
    실시간 예측 학습/평가 숫자에는 이 값을 섞으면 안 된다.
    """
    now = _parse_iso(cutoff)
    if now is None:
        return []
    start = now - timedelta(seconds=lookback_sec)
    end = now + timedelta(seconds=window_sec)
    rows = await openf1.get_positions(session_key)
    candidate_by_driver = {int(c["driver"]): c for c in candidates}
    gains: list[dict] = []
    driver_numbers = {
        int(r["driver_number"])
        for r in rows
        if r.get("driver_number") is not None
    }
    for dn in driver_numbers:
        p0 = _latest_position_at(rows, dn, start)
        p1 = _latest_position_at(rows, dn, end)
        if p0 is not None and p1 is not None and p1 < p0:
            event = _first_position_gain_event(rows, dn, start, end)
            c = candidate_by_driver.get(dn, {"driver": dn, "gap": 999.0, "trend": None})
            gains.append({
                **c,
                "position_before": p0,
                "position_after": p1,
                "event_time": event[0] if event else cutoff,
            })
    gains.sort(key=lambda c: (c["position_after"], c["gap"]))
    return gains


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


async def _battle_candidates(
    session_key: int,
    cutoff: str | None,
    max_gap: float = _MAX_GAP,
    max_candidates: int = _MAX_CANDIDATES,
) -> list[dict]:
    rows = await openf1.get_intervals(session_key)
    by_driver: dict[int, list[tuple[str, float]]] = {}
    for r in rows:
        dn, dt, iv = r.get("driver_number"), r.get("date"), r.get("interval")
        if dn is None or not dt:
            continue
        if cutoff and dt > cutoff:
            continue
        try:
            iv = float(iv)
        except (TypeError, ValueError):
            continue
        by_driver.setdefault(int(dn), []).append((dt, iv))

    out: list[dict] = []
    for dn, samples in by_driver.items():
        samples.sort(key=lambda x: x[0])
        if not samples:
            continue
        dt_now, gap = samples[-1]
        if not (0 < gap < max_gap):
            continue
        trend = None
        t_now = _parse_iso(dt_now)
        if t_now is not None:
            for dt_prev, gap_prev in reversed(samples[:-1]):
                t_prev = _parse_iso(dt_prev)
                if t_prev is not None and (t_now - t_prev).total_seconds() >= 3.0:
                    trend = gap - gap_prev
                    break
        out.append({"driver": dn, "gap": gap, "trend": trend})

    out.sort(key=lambda x: x["gap"])
    return out[:max_candidates]


async def watch(
    get_state: Callable[[], dict | None],
    announce: Callable[..., Awaitable[None]],
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
    last_evaluated_key: tuple[int, str] | None = None
    last_cutoff_dt: datetime | None = None
    announced: dict[int, float] = {}   # driver -> 마지막 안내 시각(중복 방지)
    announced_events: set[tuple[int, int, int, int]] = set()
    logger.info("[watcher] 예측형 능동 안내 시작 (threshold=%.2f, period=%.1fs)", threshold, period)

    while True:
        await asyncio.sleep(period)
        st = get_state()
        if not st or st.get("is_playing") is not True:
            continue
        received_at = st.get("_received_monotonic")
        if received_at is not None and time.monotonic() - float(received_at) > period * 2.5:
            continue
        session = st.get("session_key")
        cutoff = st.get("at_time")
        if session is None or not cutoff:
            continue
        cutoff_dt = _parse_iso(cutoff)
        if cutoff_dt is not None and last_cutoff_dt is not None:
            replay_jump = abs((cutoff_dt - last_cutoff_dt).total_seconds())
            if replay_jump > 5.0:
                last_fire = 0.0
                announced.clear()
                logger.info("[watcher] replay seek detected; cooldown reset (jump=%.1fs)", replay_jump)
        if cutoff_dt is not None:
            last_cutoff_dt = cutoff_dt
        try:
            playback_speed = max(1.0, float(st.get("playback_speed") or 1.0))
        except (TypeError, ValueError):
            playback_speed = 1.0
        detection_cutoff = _shift_iso(
            cutoff,
            min(3.0, max(0.0, playback_speed * period * 2.0 - period)),
        )
        eval_key = (int(session), str(cutoff))
        if eval_key == last_evaluated_key:
            continue
        last_evaluated_key = eval_key

        try:
            candidates_info = await _battle_candidates(session, detection_cutoff)
            elapsed = _elapsed_from_session_hour(detection_cutoff)
            confirmed_gains = await _confirmed_gain_candidates(session, cutoff, candidates_info)
            if confirmed_gains:
                confirmed = confirmed_gains[0]
                dn = confirmed["driver"]
                event_key = (
                    int(session),
                    int(dn),
                    int(confirmed["position_before"]),
                    int(confirmed["position_after"]),
                )
                if event_key in announced_events:
                    continue
                now = time.monotonic()
                confirmed_cooldown = min(cooldown, 2.5)
                if now - last_fire >= confirmed_cooldown and now - announced.get(dn, -1e9) >= confirmed_cooldown:
                    last_fire = now
                    announced[dn] = now
                    announced_events.add(event_key)
                    logger.info(
                        "[watcher] replay-confirmed 안내: %s번 P%s->P%s gap=%.3f",
                        dn,
                        confirmed["position_before"],
                        confirmed["position_after"],
                        confirmed["gap"],
                    )
                    watcher_eval.log_prediction(session, cutoff, dn, 0.85)
                    await announce(
                        dn,
                        0.85,
                        f"{dn}번, 곧 추월할 것 같아요!",
                        {
                            "session_key": int(session),
                            "driver_number": int(dn),
                            "event_time": confirmed.get("event_time"),
                            "position_before": confirmed["position_before"],
                            "position_after": confirmed["position_after"],
                        },
                    )
                continue
            if (
                settings.watcher_fast_hybrid_enabled
                and elapsed is not None
                and elapsed >= settings.watcher_fast_min_elapsed_sec
            ):
                fast_hits = [
                    c for c in candidates_info
                    if c["gap"] <= settings.watcher_fast_gap_sec
                    and c["trend"] is not None
                    and c["trend"] <= settings.watcher_fast_closing_delta
                ]
                fast = min(
                    fast_hits,
                    key=lambda c: (c["gap"], c["trend"]),
                    default=None,
                )
                if fast is not None:
                    dn = fast["driver"]
                    now = time.monotonic()
                    if now - last_fire >= cooldown and now - announced.get(dn, -1e9) >= cooldown * 2:
                        last_fire = now
                        announced[dn] = now
                        logger.info(
                            "[watcher] fast hybrid 안내: %s번 gap=%.3f trend=%.3f",
                            dn,
                            fast["gap"],
                            fast["trend"],
                        )
                        watcher_eval.log_prediction(session, cutoff, dn, settings.watcher_threshold)
                        await announce(dn, settings.watcher_threshold, f"{dn}번, 곧 추월할 것 같아요!")
                    continue

            candidates = [c["driver"] for c in candidates_info]
            best: tuple[int, float, float, str] | None = None
            for dn in candidates:
                feats = await _feat.build_features(session, detection_cutoff, dn)
                if settings.watcher_ignore_lap1 and feats.get("is_lap1") == 1.0:
                    logger.info("[watcher] skip lap1/start-phase candidate: %s", dn)
                    continue
                gap_trend = feats.get("gap_trend")
                if (
                    settings.watcher_require_closing
                    and gap_trend is not None
                    and gap_trend > 0.05
                ):
                    logger.info(
                        "[watcher] skip opening-gap candidate: %s gap_trend=%.3f",
                        dn,
                        gap_trend,
                    )
                    continue
                prob = _pred.predict(feats).get("overtake_probability", 0.0) or 0.0
                gap = feats.get("gap_ahead")
                hybrid_hit = (
                    settings.watcher_hybrid_enabled
                    and gap is not None
                    and gap <= settings.watcher_hybrid_gap_sec
                    and gap_trend is not None
                    and gap_trend <= settings.watcher_hybrid_closing_delta
                    and prob >= settings.watcher_hybrid_min_probability
                )
                reason = "ml" if prob >= threshold else "hybrid"
                score = prob if prob >= threshold else (prob + 0.2 if hybrid_hit else prob)
                if (prob >= threshold or hybrid_hit) and (best is None or score > best[1]):
                    best = (dn, score, prob, reason)

            if best is None:
                continue

            dn, _score, prob, reason = best
            now = time.monotonic()
            if now - last_fire < cooldown:                     # 전체 쿨다운
                continue
            if now - announced.get(dn, -1e9) < cooldown * 2:   # 같은 차 반복 방지
                continue

            last_fire = now
            announced[dn] = now
            logger.info("[watcher] 능동 안내: %s번 추월확률 %.2f (%s)", dn, prob, reason)
            # 오탐 평가용: 이 예측(차량·시각·확률)을 기록 → 나중에 실제 추월 여부와 대조.
            watcher_eval.log_prediction(session, cutoff, dn, prob)
            await announce(dn, prob, f"{dn}번, 곧 추월할 것 같아요!")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[watcher] 예측/안내 실패(계속)")
