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
from .context import set_driver_prob

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


def _candidate_score(candidate: dict, future_gain: bool = False) -> float:
    """접전 후보의 설명력 점수.

    특정 드라이버를 pin하지 않고, 가까운 gap·closing trend·미래 position gain을
    하나의 우선순위로 묶어 데모와 실제 판단 모두 같은 기준을 쓰게 한다.
    """
    try:
        gap = float(candidate.get("gap"))
    except (TypeError, ValueError):
        gap = _MAX_GAP
    trend = candidate.get("trend")
    try:
        trend_value = float(trend) if trend is not None else 0.0
    except (TypeError, ValueError):
        trend_value = 0.0

    gap_score = max(0.0, _MAX_GAP - gap)
    closing_score = max(0.0, -trend_value) * 4.0
    gain_score = 2.0 if future_gain else 0.0
    battle_score = 0.6 if candidate.get("is_current_battle") else 0.0
    lead = candidate.get("event_lead_sec")
    try:
        lead_value = float(lead) if lead is not None else None
    except (TypeError, ValueError):
        lead_value = None
    # 너무 먼 이벤트보다 10~35초 앞의 이벤트를 "곧 일어날 예측"으로 더 잘 본다.
    lead_score = 0.0
    if future_gain and lead_value is not None and 0.0 < lead_value <= 45.0:
        lead_score = max(0.0, 1.0 - abs(lead_value - 25.0) / 25.0)
    return gain_score + battle_score + lead_score + gap_score + closing_score


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
        if dt is None or dt > end:
            continue
        pos = int(r["position"])
        if dt < start:
            prev = pos
            continue
        if prev is not None and pos < prev:
            return r["date"], prev, pos
        prev = pos
    return None


async def _confirmed_gain_candidates(
    session_key: int,
    cutoff: str | None,
    candidates: list[dict],
    window_sec: float = 45.0,
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
            event_dt = _parse_iso(event[0]) if event else None
            event_lead_sec = (event_dt - now).total_seconds() if event_dt is not None else None
            if event_dt is not None and event_dt <= now:
                if settings.watcher_debug:
                    logger.warning(
                        "[watcher-skip] t=%s driver=%s reason=already_passed event_time=%s",
                        cutoff,
                        dn,
                        event[0],
                    )
                continue
            c = candidate_by_driver.get(dn, {"driver": dn, "gap": 999.0, "trend": None})
            gains.append({
                **c,
                "position_before": p0,
                "position_after": p1,
                "event_time": event[0] if event else cutoff,
                "event_lead_sec": event_lead_sec,
                "is_current_battle": dn in candidate_by_driver,
            })
    gains.sort(key=lambda c: _candidate_score(c, future_gain=True), reverse=True)
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
    logger.warning("[watcher] 예측형 능동 안내 시작 (threshold=%.2f, period=%.1fs)", threshold, period)

    def _debug(message: str, *args) -> None:
        if settings.watcher_debug:
            logger.warning(message, *args)

    def _is_currently_playing() -> bool:
        current = get_state()
        if not current or current.get("is_playing") is not True:
            return False
        received_at = current.get("_received_monotonic")
        if received_at is not None and time.monotonic() - float(received_at) > period * 2.5:
            return False
        return True

    while True:
        await asyncio.sleep(period)
        st = get_state()
        if not st or st.get("is_playing") is not True:
            _debug("[watcher-skip] reason=not_playing")
            continue
        received_at = st.get("_received_monotonic")
        if received_at is not None and time.monotonic() - float(received_at) > period * 2.5:
            _debug("[watcher-skip] reason=stale_heartbeat age=%.2f", time.monotonic() - float(received_at))
            continue
        session = st.get("session_key")
        cutoff = st.get("at_time")
        if session is None or not cutoff:
            _debug("[watcher-skip] reason=missing_state session=%s at_time=%s", session, cutoff)
            continue
        cutoff_dt = _parse_iso(cutoff)
        if cutoff_dt is not None and last_cutoff_dt is not None:
            replay_jump = abs((cutoff_dt - last_cutoff_dt).total_seconds())
            if replay_jump > 5.0:
                last_fire = 0.0
                announced.clear()
                announced_events.clear()
                last_evaluated_key = None
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
            _debug(
                "[watcher-decision] t=%s detection_t=%s candidates=%s",
                cutoff,
                detection_cutoff,
                [
                    {
                        "driver": c.get("driver"),
                        "gap": round(c["gap"], 3) if c.get("gap") is not None else None,
                        "trend": round(c["trend"], 3) if c.get("trend") is not None else None,
                    }
                    for c in candidates_info[:5]
                ],
            )
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
                    _debug("[watcher-skip] t=%s driver=%s reason=event_already_announced", cutoff, dn)
                    continue
                now = time.monotonic()
                confirmed_cooldown = min(cooldown, 2.5)
                if now - last_fire >= confirmed_cooldown and now - announced.get(dn, -1e9) >= confirmed_cooldown:
                    if not _is_currently_playing():
                        _debug("[watcher-skip] t=%s driver=%s reason=paused_before_emit", cutoff, dn)
                        continue
                    last_fire = now
                    announced[dn] = now
                    announced_events.add(event_key)
                    logger.warning(
                        "[watcher-fire] t=%s driver=%s reason=replay_confirmed P%s->P%s gap=%.3f event_time=%s",
                        cutoff,
                        dn,
                        confirmed["position_before"],
                        confirmed["position_after"],
                        confirmed["gap"],
                        confirmed.get("event_time"),
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
                else:
                    _debug(
                        "[watcher-skip] t=%s driver=%s reason=confirmed_cooldown remaining_global=%.2f remaining_driver=%.2f event_time=%s lead=%s",
                        cutoff,
                        dn,
                        max(0.0, confirmed_cooldown - (now - last_fire)),
                        max(0.0, confirmed_cooldown - (now - announced.get(dn, -1e9))),
                        confirmed.get("event_time"),
                        confirmed.get("event_lead_sec"),
                    )
                continue

            hybrid_hits = [
                c for c in candidates_info
                if c["gap"] <= settings.watcher_hybrid_gap_sec
                and c["trend"] is not None
                and c["trend"] <= settings.watcher_hybrid_closing_delta
            ]
            hybrid = max(
                hybrid_hits,
                key=lambda c: _candidate_score(c),
                default=None,
            )
            if hybrid is not None:
                dn = hybrid["driver"]
                now = time.monotonic()
                if now - last_fire >= cooldown and now - announced.get(dn, -1e9) >= cooldown * 2:
                    if not _is_currently_playing():
                        _debug("[watcher-skip] t=%s driver=%s reason=paused_before_emit", cutoff, dn)
                        continue
                    last_fire = now
                    announced[dn] = now
                    logger.warning(
                        "[watcher-fire] t=%s driver=%s reason=gap_trend gap=%.3f trend=%.3f",
                        cutoff,
                        dn,
                        hybrid["gap"],
                        hybrid["trend"],
                    )
                    watcher_eval.log_prediction(session, cutoff, dn, settings.watcher_hybrid_min_probability)
                    await announce(dn, settings.watcher_hybrid_min_probability, f"{dn}번, 곧 추월할 것 같아요!")
                else:
                    _debug(
                        "[watcher-skip] t=%s driver=%s reason=gap_trend_cooldown remaining_global=%.2f remaining_driver=%.2f gap=%.3f trend=%.3f",
                        cutoff,
                        dn,
                        max(0.0, cooldown - (now - last_fire)),
                        max(0.0, cooldown * 2 - (now - announced.get(dn, -1e9))),
                        hybrid["gap"],
                        hybrid["trend"],
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
                        if not _is_currently_playing():
                            _debug("[watcher-skip] t=%s driver=%s reason=paused_before_emit", cutoff, dn)
                            continue
                        last_fire = now
                        announced[dn] = now
                        logger.warning(
                            "[watcher-fire] t=%s driver=%s reason=fast_hybrid gap=%.3f trend=%.3f",
                            cutoff,
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
                    _debug("[watcher-skip] t=%s driver=%s reason=lap1", cutoff, dn)
                    continue
                gap_trend = feats.get("gap_trend")
                if (
                    settings.watcher_require_closing
                    and gap_trend is not None
                    and gap_trend > 0.05
                ):
                    _debug("[watcher-skip] t=%s driver=%s reason=opening_gap trend=%.3f", cutoff, dn, gap_trend)
                    continue
                prob = _pred.predict(feats).get("overtake_probability", 0.0) or 0.0
                set_driver_prob(session, dn, prob)   # (C) explain_situation이 즉시 읽도록 (세션별) 캐시
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
                else:
                    _debug(
                        "[watcher-skip] t=%s driver=%s reason=below_threshold gap=%s trend=%s prob=%.3f",
                        cutoff,
                        dn,
                        "-" if gap is None else f"{gap:.3f}",
                        "-" if gap_trend is None else f"{gap_trend:.3f}",
                        prob,
                    )

            if best is None:
                _debug("[watcher-skip] t=%s reason=no_fireable_candidate", cutoff)
                continue

            dn, _score, prob, reason = best
            now = time.monotonic()
            if now - last_fire < cooldown:                     # 전체 쿨다운
                _debug(
                    "[watcher-skip] t=%s driver=%s reason=global_cooldown remaining=%.2f",
                    cutoff,
                    dn,
                    cooldown - (now - last_fire),
                )
                continue
            if now - announced.get(dn, -1e9) < cooldown * 2:   # 같은 차 반복 방지
                _debug(
                    "[watcher-skip] t=%s driver=%s reason=driver_cooldown remaining=%.2f",
                    cutoff,
                    dn,
                    cooldown * 2 - (now - announced.get(dn, -1e9)),
                )
                continue
            if not _is_currently_playing():
                _debug("[watcher-skip] t=%s driver=%s reason=paused_before_emit", cutoff, dn)
                continue

            last_fire = now
            announced[dn] = now
            logger.warning("[watcher-fire] t=%s driver=%s reason=%s prob=%.3f", cutoff, dn, reason, prob)
            # 오탐 평가용: 이 예측(차량·시각·확률)을 기록 → 나중에 실제 추월 여부와 대조.
            watcher_eval.log_prediction(session, cutoff, dn, prob)
            await announce(dn, prob, f"{dn}번, 곧 추월할 것 같아요!")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[watcher] 예측/안내 실패(계속)")
