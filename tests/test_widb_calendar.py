from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from sanos.widb import (
    NMS,
    NMS5,
    add_business_days,
    business_dates,
    count_business_days_between,
    half_days,
    next_business_day,
    previous_business_day,
    session_open_close,
)


def test_business_dates_and_half_days_match_reference_holiday_cases() -> None:
    assert business_dates(date(2024, 12, 20), date(2025, 1, 3)) == (
        date(2024, 12, 20),
        date(2024, 12, 23),
        date(2024, 12, 24),
        date(2024, 12, 26),
        date(2024, 12, 27),
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    )
    assert business_dates(
        date(2024, 12, 20),
        date(2025, 1, 3),
        exclude_half_days=True,
    ) == (
        date(2024, 12, 20),
        date(2024, 12, 23),
        date(2024, 12, 26),
        date(2024, 12, 27),
        date(2024, 12, 30),
        date(2024, 12, 31),
        date(2025, 1, 2),
        date(2025, 1, 3),
    )
    assert half_days(date(2024, 12, 20), date(2025, 1, 3)) == (date(2024, 12, 24),)


def test_business_day_counting_is_vectorized_and_signed() -> None:
    ends = np.asarray(["2024-12-24", "2024-12-27"], dtype="datetime64[D]")
    observed = count_business_days_between(date(2024, 12, 23), ends)
    np.testing.assert_array_equal(observed, np.asarray([1, 3], dtype=np.int32))
    starts = np.asarray(["2024-12-23", "2024-12-27"], dtype="datetime64[D]")
    observed = count_business_days_between(starts, date(2024, 12, 24))
    np.testing.assert_array_equal(observed, np.asarray([1, -2], dtype=np.int32))
    assert count_business_days_between(date(2024, 12, 27), date(2024, 12, 23)) == -3
    assert count_business_days_between(date(2024, 12, 25), date(2024, 12, 24)) == -1
    assert (
        count_business_days_between(
            date(2024, 12, 23),
            date(2024, 12, 24),
            exclude_half_days=True,
        )
        == 0
    )


def test_navigation_and_session_bounds_match_reference() -> None:
    assert add_business_days(date(2024, 12, 24), 1) == date(2024, 12, 26)
    assert add_business_days(date(2024, 12, 24), 1, count_start=True) == date(2024, 12, 24)
    assert previous_business_day(date(2024, 12, 25)) == date(2024, 12, 23)
    assert next_business_day(date(2024, 12, 24)) == date(2024, 12, 26)

    timezone = ZoneInfo("America/New_York")
    assert session_open_close(date(2024, 12, 23)) == (
        datetime(2024, 12, 23, 9, 30, tzinfo=timezone),
        datetime(2024, 12, 23, 16, 0, tzinfo=timezone),
        False,
    )
    assert session_open_close(date(2024, 12, 24)) == (
        datetime(2024, 12, 24, 9, 30, tzinfo=timezone),
        datetime(2024, 12, 24, 13, 0, tzinfo=timezone),
        True,
    )
    assert session_open_close(date(2024, 12, 25)) == (None, None, False)


def test_normalized_strike_round_trip_for_both_reference_scales() -> None:
    strikes = np.asarray([0.8, 1.0, 1.2])
    times = np.asarray([0.01, 0.1, 1.0])
    for scale in (NMS, NMS5):
        normalized = scale.to_nms(strikes, times)
        np.testing.assert_allclose(scale.from_nms(normalized, times), strikes, atol=1e-15)
