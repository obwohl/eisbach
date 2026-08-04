import pandas as pd
import pytest

from eisbach import archive


@pytest.fixture
def root(tmp_path):
    return tmp_path / "archive"


def make_forecast(reference_time, periods=4, base=15.0):
    """A minimal forecast shaped like the model's real output."""
    index = pd.date_range(pd.Timestamp(reference_time) + pd.Timedelta(hours=1),
                          periods=periods, freq="1h")
    return pd.DataFrame(
        {
            "wassertemp_q0.5": [base + i * 0.1 for i in range(periods)],
            "wassertemp_q0.01": [base - 1 + i * 0.1 for i in range(periods)],
            "wassertemp_q0.99": [base + 1 + i * 0.1 for i in range(periods)],
        },
        index=index,
    )


def write(root, reference_time, kind, base=15.0, **kwargs):
    covariates = (
        archive.COVARIATE_DWD_OBSERVED
        if kind == archive.KIND_ORACLE
        else archive.COVARIATE_DWD_FORECAST
    )
    return archive.write_forecast(
        make_forecast(reference_time, base=base),
        reference_time=reference_time,
        kind=kind,
        covariate_source=kwargs.pop("covariate_source", covariates),
        version="test",
        root=root,
        **kwargs,
    )


def test_roundtrip_preserves_values_and_provenance(root):
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE, base=15.0)

    loaded, meta = archive.load_forecast(ref, root=root)

    assert meta["kind"] == archive.KIND_LIVE
    assert meta["covariate_source"] == archive.COVARIATE_DWD_FORECAST
    assert meta["code_version"] == "test"
    pd.testing.assert_frame_equal(
        loaded, make_forecast(ref), check_names=False, check_freq=False,
    )


@pytest.mark.parametrize("kind", [archive.KIND_LIVE, archive.KIND_REPLAY, archive.KIND_ORACLE])
def test_every_kind_roundtrips(root, kind):
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, kind)
    loaded, meta = archive.load_forecast(ref, root=root)
    assert meta["kind"] == kind
    assert len(loaded) == 4


def test_oracle_cannot_overwrite_live(root):
    """The whole point of the precedence rule: real data is never lost to a replay."""
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE, base=15.0)
    write(root, ref, archive.KIND_ORACLE, base=99.0)

    loaded, meta = archive.load_forecast(ref, root=root)

    assert meta["kind"] == archive.KIND_LIVE
    assert loaded["wassertemp_q0.5"].iloc[0] == pytest.approx(15.0)


def test_live_upgrades_oracle(root):
    """Precedence works in the other direction too — a real forecast wins."""
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_ORACLE, base=99.0)
    write(root, ref, archive.KIND_LIVE, base=15.0)

    loaded, meta = archive.load_forecast(ref, root=root)

    assert meta["kind"] == archive.KIND_LIVE
    assert loaded["wassertemp_q0.5"].iloc[0] == pytest.approx(15.0)


def test_replay_beats_oracle_but_loses_to_live(root):
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_ORACLE, base=99.0)
    write(root, ref, archive.KIND_REPLAY, base=50.0)
    assert archive.load_forecast(ref, root=root)[1]["kind"] == archive.KIND_REPLAY

    write(root, ref, archive.KIND_LIVE, base=15.0)
    assert archive.load_forecast(ref, root=root)[1]["kind"] == archive.KIND_LIVE


def test_rerunning_live_refreshes_it(root):
    """Same kind, newer run: the fresher values should win rather than be ignored."""
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE, base=15.0,
          issued_at=pd.Timestamp("2026-05-10 12:05", tz="UTC"))
    write(root, ref, archive.KIND_LIVE, base=16.0,
          issued_at=pd.Timestamp("2026-05-10 12:30", tz="UTC"))

    loaded, _ = archive.load_forecast(ref, root=root)

    assert len(loaded) == 4, "a refresh must not duplicate rows"
    assert loaded["wassertemp_q0.5"].iloc[0] == pytest.approx(16.0)


def test_partitioned_by_reference_month(root):
    write(root, pd.Timestamp("2026-05-31 23:00", tz="UTC"), archive.KIND_LIVE)
    write(root, pd.Timestamp("2026-06-01 01:00", tz="UTC"), archive.KIND_LIVE)

    partitions = sorted(p.name for p in (root / "forecasts").glob("*.csv"))
    assert partitions == ["2026-05.csv", "2026-06.csv"]


def test_forecast_crossing_month_boundary_stays_in_one_partition(root):
    """Partitioning is by reference time, so a horizon may cross into the next month."""
    ref = pd.Timestamp("2026-05-31 22:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE)

    assert not (root / "forecasts" / "2026-06.csv").exists()
    stored = pd.read_csv(root / "forecasts" / "2026-05.csv", parse_dates=["target_time"])
    assert stored["target_time"].max().month == 6


def test_load_across_a_month_boundary(root):
    """A hit within tolerance can sit in the neighbouring partition."""
    ref = pd.Timestamp("2026-05-31 23:30", tz="UTC")
    write(root, ref, archive.KIND_LIVE)

    _, meta = archive.load_forecast(
        pd.Timestamp("2026-06-01 00:30", tz="UTC"), tolerance_hours=2, root=root,
    )
    assert meta["reference_time"] == ref


def test_tolerance_is_enforced(root):
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE)

    assert archive.load_forecast(ref + pd.Timedelta(hours=1),
                                 tolerance_hours=2, root=root) is not None
    assert archive.load_forecast(ref + pd.Timedelta(hours=3),
                                 tolerance_hours=2, root=root) is None


def test_kind_filter_can_demand_honesty(root):
    """Plotting an honest-only backtest must be able to exclude oracle rows."""
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_ORACLE)

    assert archive.load_forecast(ref, root=root) is not None
    assert archive.load_forecast(
        ref, kinds=[archive.KIND_LIVE, archive.KIND_REPLAY], root=root,
    ) is None


def test_load_returns_none_on_empty_archive(root):
    assert archive.load_forecast(pd.Timestamp("2026-05-10 12:00", tz="UTC"), root=root) is None


def test_naive_timestamps_are_treated_as_utc(root):
    write(root, pd.Timestamp("2026-05-10 12:00"), archive.KIND_LIVE)
    assert archive.load_forecast(pd.Timestamp("2026-05-10 12:00", tz="UTC"), root=root) is not None


def test_unknown_kind_is_rejected(root):
    with pytest.raises(ValueError, match="unknown kind"):
        write(root, pd.Timestamp("2026-05-10 12:00", tz="UTC"), "wishful")


def test_empty_forecast_is_rejected(root):
    with pytest.raises(ValueError, match="empty forecast"):
        archive.write_forecast(
            pd.DataFrame(),
            reference_time=pd.Timestamp("2026-05-10 12:00", tz="UTC"),
            kind=archive.KIND_LIVE,
            covariate_source=archive.COVARIATE_DWD_FORECAST,
            root=root,
        )


def test_read_forecasts_spans_partitions_and_filters(root):
    write(root, pd.Timestamp("2026-05-10 12:00", tz="UTC"), archive.KIND_LIVE)
    write(root, pd.Timestamp("2026-06-10 12:00", tz="UTC"), archive.KIND_ORACLE)

    assert len(archive.read_forecasts(root=root)) == 8
    assert len(archive.read_forecasts(root=root, kinds=[archive.KIND_LIVE])) == 4


def test_weather_snapshot_roundtrip_and_idempotence(root):
    issued = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    weather = pd.DataFrame({
        "timestamp": pd.date_range("2026-05-10 13:00", periods=3, freq="1h"),
        "temperature": [18.0, 19.0, 20.0],
        "precipitation": [0.0, 0.4, 0.0],
    })

    archive.write_weather_snapshot(weather, archived_at=issued, root=root)
    archive.write_weather_snapshot(weather, archived_at=issued, root=root)

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert len(stored) == 3, "re-archiving the same instant must replace, not duplicate"
    assert "precipitation" in stored.columns


def test_weather_snapshots_accumulate_within_a_month(root):
    weather = pd.DataFrame({
        "timestamp": pd.date_range("2026-05-10 13:00", periods=3, freq="1h"),
        "temperature": [18.0, 19.0, 20.0],
    })
    archive.write_weather_snapshot(weather, archived_at=pd.Timestamp("2026-05-10 12:00", tz="UTC"), root=root)
    archive.write_weather_snapshot(weather, archived_at=pd.Timestamp("2026-05-10 22:00", tz="UTC"), root=root)

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert len(stored) == 6
    assert stored["archive_timestamp"].nunique() == 2


def test_observations_update_in_place(root):
    index = pd.date_range("2026-05-10 12:00", periods=3, freq="1h", tz="UTC")
    archive.write_observations(pd.DataFrame({"wassertemp": [15.0, 15.1, 15.2]}, index=index), root=root)
    archive.write_observations(pd.DataFrame({"wassertemp": [99.0]}, index=index[:1]), root=root)

    stored = pd.read_csv(root / "observations" / "2026-05.csv")
    assert len(stored) == 3, "a correction must overwrite, not append"
    assert stored["wassertemp"].iloc[0] == pytest.approx(99.0)


def test_observations_split_across_months(root):
    index = pd.date_range("2026-05-31 22:00", periods=4, freq="1h", tz="UTC")
    archive.write_observations(pd.DataFrame({"wassertemp": [1.0, 2.0, 3.0, 4.0]}, index=index), root=root)

    assert len(pd.read_csv(root / "observations" / "2026-05.csv")) == 2
    assert len(pd.read_csv(root / "observations" / "2026-06.csv")) == 2


def test_migration_preserves_legacy_rows_as_live(tmp_path, root):
    """The legacy archive is irreplaceable real data — migration must not lose or
    mislabel any of it."""
    legacy_path = tmp_path / "water_temp_predictions_archive.csv"
    rows = []
    for ref in pd.date_range("2026-05-10 12:00", periods=3, freq="12h", tz="UTC"):
        forecast = make_forecast(ref).reset_index(names="target_time")
        forecast.insert(0, "reference_time", ref)
        rows.append(forecast)
    pd.concat(rows, ignore_index=True).to_csv(legacy_path, index=False)

    migrated = archive.migrate_legacy_forecasts(legacy_path, root=root)

    assert migrated == 12
    stored = archive.read_forecasts(root=root)
    assert len(stored) == 12
    assert (stored["kind"] == archive.KIND_LIVE).all()
    assert (stored["covariate_source"] == archive.COVARIATE_DWD_FORECAST).all()
    assert stored["reference_time"].nunique() == 3


def test_migration_is_idempotent(tmp_path, root):
    legacy_path = tmp_path / "legacy.csv"
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    forecast = make_forecast(ref).reset_index(names="target_time")
    forecast.insert(0, "reference_time", ref)
    forecast.to_csv(legacy_path, index=False)

    archive.migrate_legacy_forecasts(legacy_path, root=root)
    archive.migrate_legacy_forecasts(legacy_path, root=root)

    assert len(archive.read_forecasts(root=root)) == 4


def test_migration_of_missing_file_is_a_noop(tmp_path, root):
    assert archive.migrate_legacy_forecasts(tmp_path / "not_here.csv", root=root) == 0


def weather_rows(start, periods=3):
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=periods, freq="1h"),
        "temperature": [18.0, 19.0, 20.0][:periods],
    })


def test_a_runs_own_snapshot_is_eligible_however_late_it_was_fetched(root):
    """A run reads the gauge, then fetches the weather. The gap is not a leak — that
    forecast is literally the one the run used."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    archive.write_weather_snapshot(
        weather_rows(anchor), archived_at=anchor + pd.Timedelta(minutes=17),
        reference_time=anchor, root=root,
    )

    assert archive.load_weather_snapshot(anchor, root=root) is not None


def test_a_stale_anchor_with_a_late_fetch_is_rejected(root):
    """The dangerous case: a delayed gauge.

    A run at 16:00 still anchored to a 09:00 reading holds a weather forecast issued at
    16:00. Accepting it for a 10:00 replay would leak six hours of forecast while
    labelling the result honest.
    """
    archive.write_weather_snapshot(
        weather_rows(pd.Timestamp("2026-05-10 09:00", tz="UTC")),
        archived_at=pd.Timestamp("2026-05-10 16:00", tz="UTC"),
        reference_time=pd.Timestamp("2026-05-10 09:00", tz="UTC"),
        root=root,
    )

    assert archive.load_weather_snapshot(
        pd.Timestamp("2026-05-10 10:00", tz="UTC"), root=root,
    ) is None


def test_an_earlier_anchor_fetched_in_time_is_accepted(root):
    """The same fallback is fine when the fetch really did precede the request."""
    archive.write_weather_snapshot(
        weather_rows(pd.Timestamp("2026-05-10 09:00", tz="UTC")),
        archived_at=pd.Timestamp("2026-05-10 09:10", tz="UTC"),
        reference_time=pd.Timestamp("2026-05-10 09:00", tz="UTC"),
        root=root,
    )

    hit = archive.load_weather_snapshot(pd.Timestamp("2026-05-10 10:00", tz="UTC"), root=root)

    assert hit is not None
    _rows, anchor = hit
    assert anchor == pd.Timestamp("2026-05-10 09:00", tz="UTC")


def test_a_retry_before_the_gauge_advances_replaces_the_snapshot(root):
    """Two runs can share an anchor — a retry, or a manual run before the gauge ticks.

    They differ only in fetch time, so deduplicating on that would keep both and leave a
    replay choosing arbitrarily between two sets of identical timestamps.
    """
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    first = weather_rows(anchor)
    second = weather_rows(anchor)
    second["temperature"] = [30.0, 31.0, 32.0]

    archive.write_weather_snapshot(
        first, archived_at=anchor + pd.Timedelta(minutes=5), reference_time=anchor, root=root)
    archive.write_weather_snapshot(
        second, archived_at=anchor + pd.Timedelta(minutes=40), reference_time=anchor, root=root)

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert len(stored) == 3, "an anchor must hold exactly one snapshot"

    rows, _anchor = archive.load_weather_snapshot(anchor, root=root)
    assert rows["temperature"].tolist() == [30.0, 31.0, 32.0], "the refreshed run wins"


def test_distinct_anchors_still_accumulate(root):
    """Collapsing by anchor must not collapse genuinely different runs."""
    for hour in (10, 22):
        anchor = pd.Timestamp(f"2026-05-10 {hour}:00", tz="UTC")
        archive.write_weather_snapshot(
            weather_rows(anchor), archived_at=anchor + pd.Timedelta(minutes=5),
            reference_time=anchor, root=root,
        )

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert len(stored) == 6
    assert stored["reference_time"].nunique() == 2


def test_legacy_snapshots_without_an_anchor_still_load(root):
    """The five-daily archive predates anchors; those rows fall back to their fetch."""
    issued = pd.Timestamp("2026-05-10 09:00", tz="UTC")
    legacy = weather_rows(issued)
    legacy["archive_timestamp"] = issued.isoformat()
    (root / "weather").mkdir(parents=True, exist_ok=True)
    legacy.to_csv(root / "weather" / "2026-05.csv", index=False)

    hit = archive.load_weather_snapshot(pd.Timestamp("2026-05-10 10:00", tz="UTC"), root=root)

    assert hit is not None
    _rows, anchor = hit
    assert anchor == issued
