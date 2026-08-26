import pytest

from app.ml.features import _closing_ratio, _num


def test_chronological_closing_ratio_treats_falling_gap_as_closing():
    assert _closing_ratio([0.9, 0.7, 0.5, 0.3]) == pytest.approx(1.0)
    assert _closing_ratio([0.3, 0.5, 0.7, 0.9]) == pytest.approx(0.0)


def test_gap_parser_rejects_lapped_interval_text():
    assert _num("+1 LAP") is None
