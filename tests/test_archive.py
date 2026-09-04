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


def test_a_blank_issued_at_is_rejected(root):
    """`_resolve_precedence` breaks same-kind ties on `issued_at` with
    `na_position="first"`, so a blank does not merely lose information — it loses every
    tie it enters, silently."""
    with pytest.raises(ValueError, match="issued_at"):
        write(root, pd.Timestamp("2026-05-10 12:00", tz="UTC"),
              archive.KIND_LIVE, issued_at=pd.NaT)


def test_an_omitted_issued_at_still_defaults_to_now(root):
    """Refusing a blank must not turn the field into a required argument."""
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE)

    _, meta = archive.load_forecast(ref, root=root)
    assert pd.notna(meta["issued_at"])


def test_archived_rows_with_a_blank_issued_at_are_left_alone(root):
    """The guard is at the door, not a migration.

    1 824 rows already carry a blank, and appending to their partition rewrites the whole
    file — which must preserve them rather than fail on, or repair, irreplaceable data.
    """
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    write(root, ref, archive.KIND_LIVE)

    path = root / "forecasts" / "2026-05.csv"
    damaged = pd.read_csv(path)
    damaged["issued_at"] = ""
    damaged.to_csv(path, index=False)

    write(root, pd.Timestamp("2026-05-11 12:00", tz="UTC"), archive.KIND_LIVE)

    stored = pd.read_csv(path)
    assert len(stored) == 8
    assert stored["issued_at"].isna().sum() == 4


def test_migration_records_an_explicit_unknown_model(tmp_path, root):
    """`model_id=""` reads as "no model", which is a different and false claim."""
    legacy_path = tmp_path / "legacy.csv"
    ref = pd.Timestamp("2026-05-10 12:00", tz="UTC")
    forecast = make_forecast(ref).reset_index(names="target_time")
    forecast.insert(0, "reference_time", ref)
    forecast.to_csv(legacy_path, index=False)

    archive.migrate_legacy_forecasts(legacy_path, root=root)

    stored = archive.read_forecasts(root=root)
    assert (stored["model_id"] == archive.LEGACY_MODEL_ID).all()


def full_fetch(anchor, history_days=40, forecast_days=8):
    """A weather frame shaped like a real fetch: 40 days back, 8 days on."""
    index = pd.date_range(
        anchor - pd.Timedelta(days=history_days),
        anchor + pd.Timedelta(days=forecast_days),
        freq="1h",
    )
    return pd.DataFrame({
        "timestamp": index,
        "lufttemperatur_c": range(len(index)),
        "niederschlag_mm": 0.0,
        "pressure": 1013.0,
    })


def stored_timestamps(root, anchor):
    rows, _ = archive.load_weather_snapshot(anchor, root=root)
    return pd.to_datetime(rows["timestamp"], utc=True).sort_values().reset_index(drop=True)


def test_trimming_keeps_every_row_an_eligible_anchor_could_read(root):
    """The trim's whole contract, stated as an assertion.

    A snapshot anchored at R can be selected for any backtest anchor in
    ``[R, R + SNAPSHOT_MAX_AGE_HOURS]``, and the splice then reads only the rows strictly
    after that anchor. So for every anchor in that window, what was stored must match
    what was fetched exactly. Anything tighter silently shortens a replay's weather.
    """
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    fetched = full_fetch(anchor)
    archive.write_weather_snapshot(
        fetched, archived_at=anchor + pd.Timedelta(minutes=17),
        reference_time=anchor, root=root,
    )

    stored = stored_timestamps(root, anchor)
    original = pd.to_datetime(fetched["timestamp"], utc=True)

    for hours in range(archive.SNAPSHOT_MAX_AGE_HOURS + 1):
        cut = anchor + pd.Timedelta(hours=hours)
        assert (
            stored[stored > cut].tolist() == original[original > cut].tolist()
        ), f"a replay anchored {hours}h after the snapshot would see a shortened forecast"


def test_a_snapshot_keeps_exactly_one_age_window_of_run_up(root):
    """The margin before the anchor is defensive — nothing reads those rows today.

    It is tied to SNAPSHOT_MAX_AGE_HOURS rather than picked so it stays one eligibility
    window wide if that window ever moves, instead of decaying into a magic number.
    """
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    archive.write_weather_snapshot(
        full_fetch(anchor), archived_at=anchor + pd.Timedelta(minutes=17),
        reference_time=anchor, root=root,
    )

    stored = stored_timestamps(root, anchor)
    margin = pd.Timedelta(hours=archive.SNAPSHOT_HISTORY_MARGIN_HOURS)

    assert stored.min() == anchor - margin + pd.Timedelta(hours=1)
    assert stored.max() == anchor + pd.Timedelta(days=8)


def test_trimming_drops_the_unreadable_history(root):
    """83 % of a full fetch was 40 days of past weather, re-archived three times a day."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    fetched = full_fetch(anchor)
    archive.write_weather_snapshot(
        fetched, archived_at=anchor, reference_time=anchor, root=root,
    )

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert len(stored) < len(fetched) / 4


def test_a_snapshot_with_nothing_a_replay_could_use_is_rejected(root):
    """Entirely-past weather is not worth a partition, and reaching here means the
    fetch or the anchor is wrong."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    stale = pd.DataFrame({
        "timestamp": pd.date_range(anchor - pd.Timedelta(days=5), periods=3, freq="1h"),
        "lufttemperatur_c": [18.0, 19.0, 20.0],
    })

    with pytest.raises(ValueError, match="no rows after"):
        archive.write_weather_snapshot(stale, archived_at=anchor,
                                       reference_time=anchor, root=root)


def test_a_snapshot_without_a_timestamp_column_is_rejected(root):
    """Seven pre-August snapshots reached the archive with no usable index. The values
    were there, the timestamps were gone, and they are unreplayable and unrecoverable."""
    with pytest.raises(ValueError, match="timestamp"):
        archive.write_weather_snapshot(
            pd.DataFrame({"lufttemperatur_c": [18.0, 19.0]}),
            archived_at=pd.Timestamp("2026-05-10 10:00", tz="UTC"), root=root,
        )


def test_a_snapshot_with_empty_timestamps_is_rejected(root):
    """The same failure in its historical shape: the column is there, the values are not."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    blank = pd.DataFrame({"timestamp": [None, None], "lufttemperatur_c": [18.0, 19.0]})

    with pytest.raises(ValueError, match="no rows after"):
        archive.write_weather_snapshot(blank, archived_at=anchor,
                                       reference_time=anchor, root=root)


def test_a_normalised_snapshot_declares_its_schema(root):
    """`data/archive/weather/` holds two incompatible schemas; new rows say which."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    archive.write_weather_snapshot(
        full_fetch(anchor), archived_at=anchor, reference_time=anchor, root=root,
    )

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert (stored["schema_version"] == archive.WEATHER_SCHEMA_VERSION).all()


def test_a_snapshot_in_another_shape_is_stored_without_a_schema_claim(root):
    """The stamp is earned, not asserted, so a reader may trust it."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    raw = pd.DataFrame({
        "timestamp": pd.date_range(anchor, periods=3, freq="1h"),
        "temperature": [18.0, 19.0, 20.0],
        "pressure_msl": [1013.0, 1013.5, 1014.0],
    })
    archive.write_weather_snapshot(raw, archived_at=anchor, reference_time=anchor, root=root)

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert "schema_version" not in stored.columns


def test_both_weather_schemas_coexist_in_one_partition(root):
    """Old rows keep their silence; only the new ones are labelled."""
    anchor = pd.Timestamp("2026-05-10 10:00", tz="UTC")
    legacy = pd.DataFrame({
        "timestamp": pd.date_range(anchor, periods=3, freq="1h"),
        "temperature": [18.0, 19.0, 20.0],
    })
    legacy["archive_timestamp"] = anchor.isoformat()
    (root / "weather").mkdir(parents=True, exist_ok=True)
    legacy.to_csv(root / "weather" / "2026-05.csv", index=False)

    later = anchor + pd.Timedelta(hours=6)
    archive.write_weather_snapshot(
        full_fetch(later), archived_at=later, reference_time=later, root=root,
    )

    stored = pd.read_csv(root / "weather" / "2026-05.csv")
    assert stored["schema_version"].isna().sum() == 3
    assert (stored["schema_version"].dropna() == archive.WEATHER_SCHEMA_VERSION).all()


def test_observations_carry_the_measured_covariates(root):
    """Without observed air temperature and pressure there is no way to tell "the model
    was wrong about the river" from "DWD was wrong about the air"."""
    index = pd.date_range("2026-05-10 12:00", periods=2, freq="1h", tz="UTC")
    archive.write_observations(pd.DataFrame({
        "wassertemp": [15.0, 15.1],
        "lufttemperatur_c": [18.0, 18.5],
        "pressure": [1013.0, 1013.5],
    }, index=index), root=root)

    stored = pd.read_csv(root / "observations" / "2026-05.csv")
    assert list(stored.columns) == ["timestamp", "wassertemp", "lufttemperatur_c", "pressure"]
    assert stored["pressure"].tolist() == pytest.approx([1013.0, 1013.5])


def test_an_old_shape_observations_partition_still_appends(root):
    """Every partition written so far holds only `timestamp,wassertemp`.

    Appending must neither rewrite that history nor invent weather for it.
    """
    index = pd.date_range("2026-05-10 12:00", periods=3, freq="1h", tz="UTC")
    (root / "observations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": index, "wassertemp": [15.0, 15.1, 15.2]}).to_csv(
        root / "observations" / "2026-05.csv", index=False,
    )

    archive.write_observations(pd.DataFrame({
        "wassertemp": [15.3], "lufttemperatur_c": [18.0], "pressure": [1013.0],
    }, index=index[-1:] + pd.Timedelta(hours=1)), root=root)

    stored = pd.read_csv(root / "observations" / "2026-05.csv")
    assert len(stored) == 4, "the old rows must survive"
    assert stored["wassertemp"].tolist() == pytest.approx([15.0, 15.1, 15.2, 15.3])
    assert stored["lufttemperatur_c"].isna().tolist() == [True, True, True, False]


def test_an_old_shape_partition_gains_covariates_for_hours_rewritten(root):
    """A run archives 40 days of observations, so old-shape rows do get revisited."""
    index = pd.date_range("2026-05-10 12:00", periods=2, freq="1h", tz="UTC")
    (root / "observations").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": index, "wassertemp": [15.0, 15.1]}).to_csv(
        root / "observations" / "2026-05.csv", index=False,
    )

    archive.write_observations(pd.DataFrame({
        "wassertemp": [15.0, 15.1],
        "lufttemperatur_c": [18.0, 18.5],
        "pressure": [1013.0, 1013.5],
    }, index=index), root=root)

    stored = pd.read_csv(root / "observations" / "2026-05.csv")
    assert len(stored) == 2
    assert stored["lufttemperatur_c"].tolist() == pytest.approx([18.0, 18.5])


def test_a_narrow_write_does_not_blank_the_columns_it_omits(root):
    """`drop_duplicates(keep="last")` would let a water-only correction erase the
    archived weather for that hour; merging column by column does not."""
    index = pd.date_range("2026-05-10 12:00", periods=2, freq="1h", tz="UTC")
    archive.write_observations(pd.DataFrame({
        "wassertemp": [15.0, 15.1],
        "lufttemperatur_c": [18.0, 18.5],
        "pressure": [1013.0, 1013.5],
    }, index=index), root=root)

    archive.write_observations(
        pd.DataFrame({"wassertemp": [99.0]}, index=index[:1]), root=root,
    )

    stored = pd.read_csv(root / "observations" / "2026-05.csv")
    assert stored["wassertemp"].iloc[0] == pytest.approx(99.0), "the correction must land"
    assert stored["lufttemperatur_c"].iloc[0] == pytest.approx(18.0), "and take nothing with it"
