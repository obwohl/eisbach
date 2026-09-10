import numpy as np
import pandas as pd
import pytest

from eisbach import archive, verification
from eisbach.model import QUANTILES

#: Offsets from the median, one per quantile level, roughly normal-shaped.
_SPREAD = [-2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0]


@pytest.fixture
def root(tmp_path):
    return tmp_path / "archive"


def make_forecast(reference_time, hours=96, median=15.0, spread=1.0):
    """A forecast shaped like the model's real output: 96 hours, seven quantiles."""
    reference_time = pd.Timestamp(reference_time)
    index = pd.date_range(reference_time + pd.Timedelta(hours=1), periods=hours, freq="1h")
    centre = np.full(hours, float(median))
    return pd.DataFrame(
        {f"wassertemp_q{q}": centre + offset * spread
         for q, offset in zip(QUANTILES, _SPREAD, strict=True)},
        index=index,
    )


def make_actuals(reference_time, hours=97, value=15.0):
    """Measurements covering the anchor hour and the whole horizon after it."""
    reference_time = pd.Timestamp(reference_time)
    index = pd.date_range(reference_time, periods=hours, freq="1h")
    return pd.Series(float(value), index=index)


REF = pd.Timestamp("2026-08-10 06:00", tz="UTC")


# --- the scoring arithmetic -------------------------------------------------------

def test_crps_of_a_point_forecast_is_the_absolute_error_over_the_reported_range():
    # With every quantile at the same value the pinball loss is linear in tau, so the
    # integral collapses to |error| times the width of the reported range. The 0.98
    # rather than 1.0 is the truncation the docstring warns about: the tails beyond
    # q0.01 and q0.99 are not reported, so they are not integrated.
    values = np.full((1, len(QUANTILES)), 15.0)
    assert verification.crps(values, np.array([17.0]))[0] == pytest.approx(0.98 * 2.0)
    assert verification.crps(values, np.array([13.0]))[0] == pytest.approx(0.98 * 2.0)


def test_crps_rewards_a_sharp_forecast_that_is_right():
    truth = np.array([15.0])
    sharp = np.array([[15 + o * 0.1 for o in _SPREAD]])
    wide = np.array([[15 + o * 2.0 for o in _SPREAD]])
    assert verification.crps(sharp, truth)[0] < verification.crps(wide, truth)[0]


def test_crps_punishes_a_sharp_forecast_that_is_wrong():
    truth = np.array([19.0])
    sharp = np.array([[15 + o * 0.1 for o in _SPREAD]])
    wide = np.array([[15 + o * 2.0 for o in _SPREAD]])
    assert verification.crps(sharp, truth)[0] > verification.crps(wide, truth)[0]


def test_pit_is_the_level_the_observation_sits_at():
    values = np.array([[10.0, 11.0, 13.0, 15.0, 17.0, 19.0, 20.0]])
    assert verification.pit(values, np.array([15.0]))[0] == pytest.approx(0.5)
    assert verification.pit(values, np.array([13.0]))[0] == pytest.approx(0.25)


def test_pit_clamps_rather_than_extrapolating_beyond_the_reported_quantiles():
    # The model said nothing about where in the far tail a value sits, so neither does
    # this: an observation below q0.01 reads as 0.01, not as 0.
    values = np.array([[10.0, 11.0, 13.0, 15.0, 17.0, 19.0, 20.0]])
    assert verification.pit(values, np.array([-5.0]))[0] == pytest.approx(0.01)
    assert verification.pit(values, np.array([99.0]))[0] == pytest.approx(0.99)


def test_crossed_quantiles_are_repaired_before_scoring():
    forecast = make_forecast(REF, hours=24)
    forecast["wassertemp_q0.75"], forecast["wassertemp_q0.95"] = (
        forecast["wassertemp_q0.95"].copy(), forecast["wassertemp_q0.75"].copy(),
    )
    scores = verification.score_forecast(
        forecast, make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
    )
    # Swapping two columns is not new information, so a sorted read is unmoved by it.
    straight = verification.score_forecast(
        make_forecast(REF, hours=24), make_actuals(REF),
        reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert scores["crps"].iloc[0] == pytest.approx(straight["crps"].iloc[0])


# --- one scored run --------------------------------------------------------------

def test_a_full_run_scores_one_row_per_lead_bucket():
    scores = verification.score_forecast(
        make_forecast(REF), make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert list(scores["lead_lo"]) == [0, 24, 48, 72]
    assert list(scores["lead_hi"]) == [24, 48, 72, 96]
    assert list(scores["n"]) == [24, 24, 24, 24]
    assert list(scores["n_forecast"]) == [24, 24, 24, 24]
    assert list(scores.columns) == list(verification.SCORE_COLUMNS)


def test_lead_buckets_do_not_overlap_and_cover_the_whole_horizon():
    buckets = verification.lead_buckets()
    assert buckets[0][0] == 0
    assert buckets[-1][1] == verification.COVARIATE_SHIFT_HOURS
    for (_, high), (low, _) in zip(buckets, buckets[1:], strict=False):
        assert high == low


def test_a_perfect_forecast_scores_zero_error():
    scores = verification.score_forecast(
        make_forecast(REF, median=15.0), make_actuals(REF, value=15.0),
        reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert (scores["mae"] == 0).all()
    assert (scores["rmse"] == 0).all()
    assert (scores["bias"] == 0).all()


def test_bias_keeps_its_sign():
    warm = verification.score_forecast(
        make_forecast(REF, median=16.0), make_actuals(REF, value=15.0),
        reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert warm["bias"].to_numpy() == pytest.approx(1.0)
    assert warm["mae"].to_numpy() == pytest.approx(1.0)


def test_a_bucket_without_observations_yields_no_row():
    # Only the first thirty hours were ever measured.
    actuals = make_actuals(REF, hours=31)
    scores = verification.score_forecast(
        make_forecast(REF), actuals, reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert list(scores["lead_lo"]) == [0, 24]
    assert list(scores["n"]) == [24, 6]
    # The forecast still predicted 24 hours in that bucket; six of them were measured.
    assert list(scores["n_forecast"]) == [24, 24]


def test_a_missing_quantile_column_is_refused():
    forecast = make_forecast(REF).drop(columns=["wassertemp_q0.99"])
    with pytest.raises(ValueError, match="wassertemp_q0.99"):
        verification.score_forecast(
            forecast, make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
        )


# --- the baselines ---------------------------------------------------------------

def test_persistence_is_the_anchor_observation_held_flat():
    actuals = make_actuals(REF, value=15.0)
    actuals.iloc[0] = 12.0  # the anchor reading, three degrees below everything after
    scores = verification.score_forecast(
        make_forecast(REF), actuals, reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert scores["mae_persistence"].to_numpy() == pytest.approx(3.0)


def test_persistence_is_missing_rather_than_guessed_when_the_anchor_was_not_measured():
    actuals = make_actuals(REF, hours=97).iloc[6:]  # nothing until six hours after
    scores = verification.score_forecast(
        make_forecast(REF), actuals, reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert scores["mae_persistence"].isna().all()
    assert not scores["mae"].isna().any()


def test_the_diurnal_baseline_repeats_the_last_whole_day_the_run_had_seen():
    # A pure 24-hour square wave: flat persistence is wrong half the time, repeating
    # yesterday is right always.
    index = pd.date_range(REF - pd.Timedelta(hours=24), periods=121, freq="1h")
    actuals = pd.Series(np.where(index.hour < 12, 10.0, 20.0), index=index)
    scores = verification.score_forecast(
        make_forecast(REF), actuals, reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert scores["mae_diurnal"].to_numpy() == pytest.approx(0.0)
    assert (scores["mae_persistence"] > 0).all()


def test_the_diurnal_baseline_never_reads_past_the_anchor():
    # Everything from the anchor onwards is poisoned; a causal baseline cannot see it,
    # so it must still score exactly zero against the quiet day before.
    index = pd.date_range(REF - pd.Timedelta(hours=24), periods=121, freq="1h")
    actuals = pd.Series(7.0, index=index)
    actuals.loc[actuals.index > REF] = 7.0
    scores = verification.score_forecast(
        make_forecast(REF, median=7.0), actuals, reference_time=REF, kind=archive.KIND_LIVE,
    )
    assert scores["mae_diurnal"].to_numpy() == pytest.approx(0.0)


# --- the store -------------------------------------------------------------------

def test_scores_round_trip_through_the_archive(root):
    scores = verification.score_forecast(
        make_forecast(REF), make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
    )
    archive.write_verification(scores, root=root)

    stored = archive.read_verification(root=root)
    assert len(stored) == 4
    assert (stored["schema_version"] == archive.VERIFICATION_SCHEMA_VERSION).all()
    assert stored["reference_time"].iloc[0] == REF
    for column in verification.SCORE_COLUMNS:
        assert column in stored.columns


def test_a_stored_row_is_never_restated(root):
    scores = verification.score_forecast(
        make_forecast(REF), make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
    )
    archive.write_verification(scores, root=root)
    # A later scorer that disagrees must not be able to rewrite history it disagrees
    # with: the window has closed, so the first answer is the answer.
    archive.write_verification(scores.assign(mae=99.0), root=root)

    stored = archive.read_verification(root=root)
    assert len(stored) == 4
    assert (stored["mae"] != 99.0).all()


def test_scored_rows_without_a_reference_time_are_refused(root):
    scores = verification.score_forecast(
        make_forecast(REF), make_actuals(REF), reference_time=REF, kind=archive.KIND_LIVE,
    )
    with pytest.raises(ValueError, match="reference_time"):
        archive.write_verification(scores.assign(reference_time=pd.NaT), root=root)


def test_reading_an_empty_verification_store_is_not_an_error(root):
    assert archive.read_verification(root=root).empty
    assert verification.read_scores(root=root).empty


def test_observations_round_trip(root):
    measured = pd.DataFrame(
        {"wassertemp": [15.0, 15.5], "lufttemperatur_c": [20.0, 21.0]},
        index=pd.date_range("2026-08-01", periods=2, freq="1h", tz="UTC"),
    )
    archive.write_observations(measured, root=root)

    loaded = archive.read_observations(root=root)
    assert list(loaded["wassertemp"]) == [15.0, 15.5]
    assert loaded.index.name == "timestamp"


def test_observations_spanning_two_months_load_as_one_series(root):
    measured = pd.DataFrame(
        {"wassertemp": [15.0, 15.5]},
        index=pd.DatetimeIndex(["2026-07-31 23:00", "2026-08-01 00:00"], tz="UTC"),
    )
    archive.write_observations(measured, root=root)
    assert len(archive.read_observations(root=root)) == 2


# --- scoring the archive ---------------------------------------------------------

def _archive_a_run(root, reference_time, kind=archive.KIND_LIVE, median=15.0):
    archive.write_forecast(
        make_forecast(reference_time, median=median),
        reference_time=reference_time,
        kind=kind,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
        model_id="testmodel",
        version="test",
        root=root,
    )


def test_score_archive_scores_a_closed_window(root):
    _archive_a_run(root, REF)
    archive.write_observations(make_actuals(REF).to_frame("wassertemp"), root=root)

    written = verification.score_archive(root=root)
    assert len(written) == 4
    assert written["model_id"].unique().tolist() == ["testmodel"]
    assert written["code_version"].unique().tolist() == ["test"]


def test_score_archive_leaves_an_open_window_alone(root):
    _archive_a_run(root, REF)
    # Only half the horizon has happened, so the run's score is not yet final.
    archive.write_observations(make_actuals(REF, hours=49).to_frame("wassertemp"), root=root)

    assert verification.score_archive(root=root).empty
    assert archive.read_verification(root=root).empty


def test_score_archive_is_idempotent(root):
    _archive_a_run(root, REF)
    archive.write_observations(make_actuals(REF).to_frame("wassertemp"), root=root)

    verification.score_archive(root=root)
    assert verification.score_archive(root=root).empty
    assert len(archive.read_verification(root=root)) == 4


def test_score_archive_without_observations_scores_nothing(root):
    _archive_a_run(root, REF)
    assert verification.score_archive(root=root).empty


def test_score_archive_picks_up_a_run_once_its_window_closes(root):
    _archive_a_run(root, REF)
    archive.write_observations(make_actuals(REF, hours=49).to_frame("wassertemp"), root=root)
    assert verification.score_archive(root=root).empty

    archive.write_observations(make_actuals(REF).to_frame("wassertemp"), root=root)
    assert len(verification.score_archive(root=root)) == 4


# --- reading it back -------------------------------------------------------------

def test_read_scores_leaves_out_the_kinds_that_flatter_the_model(root):
    _archive_a_run(root, REF, kind=archive.KIND_LIVE)
    _archive_a_run(root, REF - pd.Timedelta(hours=96), kind=archive.KIND_ORACLE)
    archive.write_observations(
        make_actuals(REF - pd.Timedelta(hours=96), hours=193).to_frame("wassertemp"), root=root,
    )
    verification.score_archive(root=root)

    honest = verification.read_scores(root=root)
    assert set(honest["kind"]) == {archive.KIND_LIVE}
    everything = verification.read_scores(root=root, kinds=None)
    assert set(everything["kind"]) == {archive.KIND_LIVE, archive.KIND_ORACLE}


def test_read_scores_materialises_interval_coverage(root):
    _archive_a_run(root, REF)
    archive.write_observations(make_actuals(REF).to_frame("wassertemp"), root=root)
    verification.score_archive(root=root)

    scores = verification.read_scores(root=root)
    # Every observation sits exactly on the median, so it is at or below q0.5 upwards
    # and inside every interval.
    assert scores["cov_50"].to_numpy() == pytest.approx(1.0)
    assert scores["cov_90"].to_numpy() == pytest.approx(1.0)
    assert scores["cov_98"].to_numpy() == pytest.approx(1.0)


def test_pool_weights_by_the_hours_each_row_covers():
    rows = pd.DataFrame({
        "reference_time": [REF, REF],
        "n": [90, 10],
        "n_forecast": [96, 96],
        "mae": [1.0, 11.0],
        "rmse": [1.0, 11.0],
        "bias": [0.0, 0.0],
        "crps": [1.0, 1.0],
        "mae_persistence": [2.0, 2.0],
        "mae_diurnal": [2.0, 2.0],
        "pit_mean": [0.5, 0.5],
        "lead_lo": [0, 0],
        **{c: [0.5, 0.5] for c in verification.PIT_COLUMNS},
    })
    pooled = verification.pool(rows, by=[])
    assert pooled["n"].iloc[0] == 100
    assert pooled["mae"].iloc[0] == pytest.approx((90 * 1.0 + 10 * 11.0) / 100)
    # RMSE averages in squares, so it is not the weighted mean of the two.
    assert pooled["rmse"].iloc[0] == pytest.approx(np.sqrt((90 * 1.0 + 10 * 121.0) / 100))
    assert pooled["runs"].iloc[0] == 1


def test_pool_of_one_bucket_is_that_bucket(root):
    _archive_a_run(root, REF, median=16.0)
    archive.write_observations(make_actuals(REF, value=15.0).to_frame("wassertemp"), root=root)
    verification.score_archive(root=root)

    scores = verification.read_scores(root=root)
    pooled = verification.pool(scores, by=[])
    assert pooled["mae"].iloc[0] == pytest.approx(1.0)
    assert pooled["n"].iloc[0] == 96


def test_pool_of_an_empty_table_is_empty():
    assert verification.pool(pd.DataFrame()).empty


def test_pooling_by_era_keeps_runs_whose_model_was_never_recorded(root):
    # The legacy half of this archive carries a blank model_id. A default groupby drops
    # a NaN key, which would answer for one era while looking like it answered for both.
    _archive_a_run(root, REF)
    archive.write_forecast(
        make_forecast(REF - pd.Timedelta(hours=96)),
        reference_time=REF - pd.Timedelta(hours=96),
        kind=archive.KIND_LIVE,
        covariate_source=archive.COVARIATE_DWD_FORECAST,
        model_id="",
        version="test",
        root=root,
    )
    archive.write_observations(
        make_actuals(REF - pd.Timedelta(hours=96), hours=193).to_frame("wassertemp"), root=root,
    )
    verification.score_archive(root=root)

    scores = verification.read_scores(root=root)
    assert (scores["model_id"] == "").any()
    pooled = verification.pool(scores, by=["model_id"])
    assert pooled["n"].sum() == scores["n"].sum()
    assert len(pooled) == 2


def test_pool_gives_each_column_its_own_denominator():
    # One of the two rows has no persistence baseline — its anchor hour was never
    # measured. That must not shrink the pooled baseline towards zero.
    rows = pd.DataFrame({
        "reference_time": [REF, REF - pd.Timedelta(hours=96)],
        "n": [50, 50],
        "n_forecast": [96, 96],
        "mae": [1.0, 1.0],
        "rmse": [1.0, 1.0],
        "bias": [0.0, 0.0],
        "crps": [1.0, 1.0],
        "mae_persistence": [2.0, float("nan")],
        "mae_diurnal": [2.0, 2.0],
        "pit_mean": [0.5, 0.5],
        "lead_lo": [0, 0],
        **{c: [0.5, 0.5] for c in verification.PIT_COLUMNS},
    })
    pooled = verification.pool(rows, by=[])
    assert pooled["mae"].iloc[0] == pytest.approx(1.0)
    assert pooled["mae_persistence"].iloc[0] == pytest.approx(2.0)
