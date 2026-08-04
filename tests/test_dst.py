"""Regression tests for the DST handling in eisbach.data.

The autumn fold used to crash the whole pipeline whenever the gauge dropped a single
sample during the repeated hour: `ambiguous='infer'` raises when it cannot find both
repeats. Because the input window is 40 days wide, one missing sample would have taken
the forecast down for the following six weeks.
"""

import pandas as pd
import pytest

from eisbach.data import localize_local_time

BERLIN = "Europe/Berlin"


def series(*stamps):
    return pd.Series(pd.to_datetime(list(stamps)))


def test_autumn_fold_with_both_repeats_is_inferred():
    """The happy path: both 02:00 rows present, so the offsets differ correctly."""
    result = localize_local_time(series(
        "2026-10-25 00:00", "2026-10-25 01:00",
        "2026-10-25 02:00", "2026-10-25 02:00",
        "2026-10-25 03:00",
    ))

    assert result.isna().sum() == 0
    offsets = [ts.utcoffset() for ts in result]
    assert offsets[2] == pd.Timedelta(hours=2), "first 02:00 is still summer time"
    assert offsets[3] == pd.Timedelta(hours=1), "second 02:00 is winter time"


def test_autumn_fold_with_a_gap_does_not_raise():
    """The regression: one missing repeat used to raise and kill the run."""
    stamps = series(
        "2026-10-25 00:00", "2026-10-25 01:00",
        "2026-10-25 02:00",  # only one of the two
        "2026-10-25 03:00",
    )

    # Confirm the old behaviour really was fatal, so this test cannot silently rot.
    with pytest.raises(ValueError, match="Cannot infer dst time"):
        stamps.dt.tz_localize(BERLIN, ambiguous="infer")

    result = localize_local_time(stamps)

    assert len(result) == 4
    assert result.isna().sum() == 1, "only the ambiguous hour is dropped"
    assert result.notna().sum() == 3, "every unambiguous hour survives"


def test_spring_gap_is_shifted_forward():
    """02:00 does not exist on the spring transition; it must not become NaT."""
    result = localize_local_time(series(
        "2026-03-29 01:00", "2026-03-29 02:00", "2026-03-29 03:00",
    ))

    assert result.isna().sum() == 0
    assert result.iloc[1].utcoffset() == pd.Timedelta(hours=2)


def test_ordinary_days_are_untouched():
    stamps = series("2026-07-01 00:00", "2026-07-01 01:00", "2026-07-01 02:00")
    result = localize_local_time(stamps)

    assert result.isna().sum() == 0
    assert [ts.strftime("%H:%M") for ts in result] == ["00:00", "01:00", "02:00"]


def test_unsorted_input_is_sorted_before_inferring():
    """`ambiguous='infer'` needs chronological order; the gauge is not guaranteed to
    deliver it."""
    result = localize_local_time(series(
        "2026-10-25 03:00", "2026-10-25 02:00",
        "2026-10-25 00:00", "2026-10-25 02:00",
        "2026-10-25 01:00",
    ))

    assert result.isna().sum() == 0
    assert result.is_monotonic_increasing
