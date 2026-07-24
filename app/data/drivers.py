"""driver_number ↔ Jolpica driver_id 매핑.

getDriverInfo의 커리어 조회 정확도를 위해 사용. 현재 tools.py는 성(last_name)
근사 매칭을 쓰는데, 대상 경기의 드라이버를 이 표로 채우면 정확해진다.
(대상 리플레이 경기가 고정이므로 20명 내외면 충분)
"""
from __future__ import annotations

NUMBER_TO_JOLPICA: dict[int, str] = {
    1: "max_verstappen",
    44: "hamilton",
    16: "leclerc",
    4: "norris",
    63: "russell",
    # TODO: 대상 경기 드라이버 번호 → Jolpica id 채우기
}


def jolpica_id(driver_number: int, fallback_last_name: str | None = None) -> str | None:
    """번호로 Jolpica id를 찾고, 없으면 성(last_name) 근사값을 반환."""
    return NUMBER_TO_JOLPICA.get(driver_number) or (
        fallback_last_name.lower() if fallback_last_name else None
    )
