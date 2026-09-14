"""Station-pinned, lossless covariate archive; quality flags are not deletions.

Only this new store is mutable. Original successful responses are retained compressed,
monthly hourly partitions keep unfiltered values, and model reads apply explicit policy.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from eisbach.archive import DEFAULT_ROOT, _as_utc, _read_partition, _write_partition
from eisbach.data import (
    BRIGHTSKY_URL,
    MAX_REJECTED_FRACTION,
    ImplausibleGaugeData,
    _spikes,
)

logger = logging.getLogger(__name__)
POLICY_VERSION = "v1"
STORE = DEFAULT_ROOT / "covariates"
GKD = "https://www.gkd.bayern.de/de/fluesse/wassertemperatur/isar/{station}"


@dataclass(frozen=True)
class SeriesSpec:
    station: str
    field: str
    unit: str
    bounds: tuple[float, float]
    jump: float
    floor: float
    fill_hours: int


SPECS = {
    "eisbach": SeriesSpec("muenchen-himmelreichbruecke-16515005", "water", "degC", (-5, 40), 2, 2, 1),
    "isar_toelz": SeriesSpec("bad-toelz-b472-16003207", "water", "degC", (-5, 40), 2, 2, 1),
    "loisach_beuerberg": SeriesSpec("beuerberg-16408504", "water", "degC", (-5, 40), 2, 2, 1),
    "airtemp": SeriesSpec("03379", "temperature", "degC", (-50, 50), 10, 6, 1),
    "t_catchment": SeriesSpec("01550", "temperature", "degC", (-50, 50), 10, 6, 1),
    "rain_toelz": SeriesSpec("05262", "precipitation", "mm", (0, 200), 30, 10, 0),
    "rain_lenggries": SeriesSpec("06257", "precipitation", "mm", (0, 200), 30, 10, 0),
    "rain_kochel": SeriesSpec("06342", "precipitation", "mm", (0, 200), 30, 10, 0),
    "rain_garmisch": SeriesSpec("01550", "precipitation", "mm", (0, 200), 30, 10, 0),
    "solar_hohenpeissenberg": SeriesSpec("02290", "solar", "kWh/m2", (0, 1.5), 0.8, 0.8, 0),
    # Already a trained production channel: must be cached and audited as well.
    "pressure": SeriesSpec("03379", "pressure_msl", "hPa", (850, 1100), 10, 8, 1),
}


def request(url: str, params: dict | None = None) -> requests.Response:
    """Failures never become successful empty chunks."""
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, timeout=180,
                                    headers={"User-Agent": "eisbach-covariate-archive/1"})
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def save_response(response, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(gzip.compress(response.content, mtime=0))
    tmp.replace(path)


def runs(mask: pd.Series):
    """Inclusive contiguous true intervals on an hourly grid."""
    groups = mask.ne(mask.shift()).cumsum()
    for _, part in mask[mask].groupby(groups[mask]):
        yield part.index[0], part.index[-1], len(part)


def quality(values: pd.Series, name: str) -> pd.DataFrame:
    """Report all six tests without changing a single value.

    Water uses the existing two-pass residual-MAD gate verbatim on an hourly grid.
    Other variables use a local Hampel statistic with a physical floor: weather fronts,
    rain and sunrise are less smooth than river temperature; these are review flags.
    """
    spec = SPECS[name]
    s = values.astype(float)
    measured = s.notna()
    out = pd.DataFrame(index=s.index)
    out["range"] = measured & ~s.between(*spec.bounds)
    if spec.field == "water":
        first, _ = _spikes(s, measured, exclude=out["range"])
        out["hampel"], _ = _spikes(s, measured, exclude=out["range"] | first)
    else:
        clean = s.mask(out["range"])
        neighbours = pd.concat([clean.shift(i) for i in range(-3, 4) if i], axis=1)
        median = neighbours.median(axis=1)
        mad = neighbours.sub(median, axis=0).abs().median(axis=1)
        out["hampel"] = measured & ((s - median).abs() > (12 * 1.4826 * mad).clip(lower=spec.floor))
    out["jump"] = s.diff().abs() > spec.jump
    out["frozen"] = False
    groups = s.ne(s.shift()).cumsum()
    lengths = s.groupby(groups).transform("size")
    out["frozen"] = measured & (lengths >= 24)
    out["missing"] = ~measured
    return out


def fill_short_gaps(values: pd.Series, name: str) -> pd.Series:
    """Fill entire bounded gaps only, never the first hour of a long outage."""
    limit = SPECS[name].fill_hours
    if not limit or values.empty:
        return values.copy()
    missing = values.isna()
    lengths = missing.groupby(missing.ne(missing.shift()).cumsum()).transform("size")
    interpolated = values.interpolate(method="time", limit_area="inside")
    return values.mask(missing & (lengths <= limit), interpolated)


def classify(name: str, test: str, value: float) -> str:
    spec = SPECS[name]
    if test in {"range", "subhourly_range"}:
        return ("instrument_or_encoding_fault" if spec.field in {"water", "temperature"}
                else "review_extreme_or_fault")
    if test == "frozen" and value == 0 and spec.field in {"precipitation", "solar"}:
        return "plausible_physical_zero_not_rejected"
    if test == "missing":
        return "unknown_outage_or_physical_event" if spec.field == "water" else "unknown_no_observation"
    return "undecidable_without_independent_evidence"


def hourly_gkd(html: str) -> tuple[pd.DataFrame, list[dict]]:
    """Parse before aggregation; ambiguous wall-clock hours remain explicit gaps.

    Original HTML retains every local timestamp/value, including unresolvable DST folds.
    Never shift nonexistent spring readings onto a real measurement.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.tblsort, table.datentabelle")
    if table is None:
        raise ValueError("GKD response has no measurement table")
    headers = [c.get_text(strip=True) for c in table.select("thead th")]
    value_col = next((i for i, h in enumerate(headers) if "wassertemp" in h.lower()), None)
    if value_col is None:
        raise ValueError("GKD response has no water temperature column")
    rows = []
    for row in table.select("tbody tr"):
        cells = [c.get_text(strip=True) for c in row.select("td, th")]
        if len(cells) > value_col:
            stamp = " ".join(cells[:value_col]).replace(" Uhr", "")
            rows.append((stamp, cells[value_col]))
    if not rows:
        return pd.DataFrame(), []
    raw = pd.DataFrame(rows, columns=["local_timestamp", "value"])
    raw["value"] = pd.to_numeric(raw["value"].str.replace(",", "."), errors="coerce")
    naive = pd.to_datetime(raw.local_timestamp, format="%d.%m.%Y %H:%M", errors="coerce")
    stamps = naive.dt.tz_localize("Europe/Berlin", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    issues = raw.loc[stamps.isna()].to_dict("records")
    raw["timestamp"] = stamps
    raw = raw.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    duplicates = raw.index.duplicated(keep=False)
    raw["duplicate_count"] = duplicates.astype(int)
    # Conflicting duplicates cannot safely be averaged; originals remain in the payload.
    conflicts = raw.groupby(level=0).value.nunique().gt(1)
    raw["conflict"] = raw.index.map(conflicts).astype(bool)
    raw.loc[raw["conflict"], "value"] = np.nan
    grouped = raw.resample("1h")
    # First of the hour, not the mean. GKD delivers four quarter-hourly samples, and the
    # production path this replaced used `resample("1h").first()`: every observation
    # already in `data/archive/observations/` is a first-of-hour reading, and DUET is a
    # fixed checkpoint trained on series prepared that way. Averaging instead moved 83 %
    # of September's hours, by up to 0.375 °C — a quiet discontinuity in both the model's
    # input and the target it is scored against. The mean is kept beside it, so nothing
    # is lost and the convention stays the one the record was built on.
    out = grouped.value.agg(raw_value="first", raw_mean="mean", raw_min="min",
                            raw_max="max", samples="count")
    out["duplicate_count"] = grouped.duplicate_count.sum()
    out["conflict"] = grouped.conflict.max().fillna(0).astype(int)
    return out, issues


def hourly_weather(payload: dict, station: str, field: str) -> pd.DataFrame:
    sources = {s["id"]: s for s in payload.get("sources", [])}
    rows = payload.get("weather", [])
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    for sid in raw.source_id.unique():
        source = sources.get(sid)
        if source is None or str(source.get("dwd_station_id")).zfill(5) != station:
            raise ValueError(f"Station substitution or unknown source {sid}; expected {station}")
    # A past timestamp alone does not prove that a value was measured.
    observed_ids = [sid for sid, s in sources.items() if s.get("observation_type") in
                    {"historical", "current", "synop"}]
    raw = raw[raw.source_id.isin(observed_ids)].copy()
    if raw.empty:
        return pd.DataFrame()
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    raw["value"] = pd.to_numeric(raw.get(field, pd.Series(np.nan, index=raw.index)), errors="coerce")
    raw = raw.set_index("timestamp").sort_index()
    # Bright Sky weather is hourly. Do not silently change accumulation semantics.
    if (raw.index != raw.index.floor("h")).any():
        raise ValueError("Unexpected subhourly Bright Sky /weather response")
    grouped = raw.groupby(level=0)
    out = grouped.value.agg(raw_value="first", raw_min="min", raw_max="max", samples="count")
    out["duplicate_count"] = grouped.size().where(grouped.size() > 1, 0)
    out["conflict"] = (grouped.value.nunique() > 1).astype(int)
    out.loc[out.conflict.eq(1), "raw_value"] = np.nan
    out["source_id"] = grouped.source_id.agg(lambda x: "|".join(map(str, sorted(set(x)))))
    return out


def write_hourly(name: str, incoming: pd.DataFrame, *, root: Path = STORE) -> None:
    if incoming.empty:
        return
    frame = incoming.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    if frame.index.has_duplicates or (frame.index != frame.index.floor("h")).any():
        raise ValueError("Expected unique UTC hourly timestamps")
    frame["station"] = SPECS[name].station
    frame["unit"] = SPECS[name].unit
    frame["conflict"] = frame.conflict.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    if "source_id" in frame:
        frame["source_id"] = frame.source_id.astype("string")
    for month, group in frame.groupby(frame.index.strftime("%Y-%m")):
        path = root / "hourly" / name / f"{month}.csv"
        old = _read_partition(path)
        if not old.empty:
            old = old.set_index("timestamp")
            old["station"] = SPECS[name].station
            old["conflict"] = old.conflict.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
            if "source_id" in old:
                old["source_id"] = old.source_id.astype("string")
            # Re-fetches may fill old missing cells, never blank a genuine old value.
            # All revisions remain recoverable from immutable payloads.
            retained = old.index.intersection(group.index)
            retained = retained[old.loc[retained, "raw_value"].notna()
                                & group.loc[retained, "raw_value"].isna()]
            group = group.combine_first(old)
            group.loc[retained, old.columns] = old.loc[retained]
        group["station"] = SPECS[name].station
        group["conflict"] = group.conflict.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
        _write_partition(path, group.sort_index().reset_index())


def read_hourly(name: str, *, root: Path = STORE, start=None, end=None) -> pd.DataFrame:
    start = _as_utc(start) if start is not None else None
    end = _as_utc(end) if end is not None else None
    paths = sorted((root / "hourly" / name).glob("*.csv"))
    if start is not None:
        paths = [p for p in paths if p.stem >= pd.Timestamp(start).strftime("%Y-%m")]
    if end is not None:
        paths = [p for p in paths if p.stem <= pd.Timestamp(end).strftime("%Y-%m")]
    frames = [_read_partition(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames).set_index("timestamp").sort_index()
    frame["conflict"] = frame.conflict.astype(str).str.lower().isin(["true", "1", "1.0"])
    frame["station"] = SPECS[name].station
    frame = frame.loc[start:end]
    if frame.empty:
        return frame
    return frame.reindex(pd.date_range(start or frame.index.min(), end or frame.index.max(), freq="h"))


def assess_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Identical hourly and ingestion checks for backfill reports and live runs."""
    flags = quality(frame.raw_value, name)
    flags["duplicate"] = frame.duplicate_count.fillna(0).gt(0)
    flags["conflicting_duplicate"] = frame.conflict.fillna(False).astype(bool)
    spec = SPECS[name]
    flags["subhourly_range"] = frame.raw_min.lt(spec.bounds[0]) | frame.raw_max.gt(spec.bounds[1])
    return flags


def record_quality(flags: pd.DataFrame, name: str, *, root: Path) -> None:
    """Persist every live suspicion/missing hour, including unresolved physical events."""
    for month, group in flags.groupby(flags.index.strftime("%Y-%m")):
        path = root / "quality" / name / f"{month}.csv"
        old = _read_partition(path)
        if not old.empty:
            old = old[~old.timestamp.isin(group.index)]
        current = group[group.any(axis=1)].copy()
        current["policy"] = POLICY_VERSION
        current.index.name = "timestamp"
        current = current.reset_index()
        combined = pd.concat([old, current], ignore_index=True) if not old.empty else current
        if not combined.empty or path.exists():
            _write_partition(path, combined.sort_values("timestamp"))


def model_values(name: str, *, root: Path = STORE, start=None, end=None) -> pd.Series:
    """Reviewed/automatic rejection is a separate, logged view; raw values never change."""
    frame = read_hourly(name, root=root, start=start, end=end)
    if frame.empty:
        raise RuntimeError(f"Archive has no {name}; run experiments/timesfm/build_archive.py")
    s = frame.raw_value.copy()
    flags = assess_frame(frame, name)
    record_quality(flags, name, root=root)
    # Conservative: for air and rain, hard bounds only. Hampel/jump and plateaux are
    # review findings there, not proof of a broken instrument or a dry river — a front,
    # a downpour and sunrise are all genuinely abrupt.
    reject = (flags["range"] if SPECS[name].field in {"water", "temperature"}
              else pd.Series(False, index=s.index))
    reasons = pd.Series("outside_physical_range", index=s.index)
    if SPECS[name].field == "water":
        # Check subhourly extremes too: an hourly mean must not dilute a bad sample.
        reject |= frame.raw_min.lt(-5) | frame.raw_max.gt(40)
        # And the neighbourhood test, which is the whole point of the gate. Bounds alone
        # catch 154.4 °C and miss 30 °C between neighbours of 19 °C — in range, and no
        # river did it. `quality` runs the same two-pass residual-MAD test as the legacy
        # `reject_implausible_readings`, calibrated over sixteen years: the largest
        # genuine residual is 1.33 °C, the one instrument fault 46.7 °C, and the 2.0 °C
        # floor flags exactly one reading in the whole record. Computing that verdict and
        # then not acting on it left production with no spike protection at all.
        spikes = flags["hampel"] & ~reject
        reasons = reasons.mask(spikes, "neighbourhood_spike")
        reject |= spikes
    measured = s.notna()
    # Below this many readings the fraction cannot mean anything: at a 1 % budget a
    # single rejection already exceeds it in any window shorter than a hundred hours,
    # so the test would condemn a healthy gauge for one spike. The legacy gate was only
    # ever called on a full 384-hour window and never met the case.
    enough = measured.sum() >= round(1 / MAX_REJECTED_FRACTION)
    if enough:
        share = float((reject & measured).sum()) / float(measured.sum())
        if share > MAX_REJECTED_FRACTION:
            # Past this point the series is not a good signal with a spike in it, and a
            # run that interpolated over it would forecast from mostly guesses.
            raise ImplausibleGaugeData(
                f"{name}: {int((reject & measured).sum())} of {int(measured.sum())} "
                f"readings are implausible ({share:.1%} > {MAX_REJECTED_FRACTION:.1%}); "
                f"the instrument is broken, not spiky"
            )
    for ts in s.index[reject]:
        record = {"timestamp": ts, "raw_value": s.loc[ts], "value": np.nan,
                  "status": "never_measured", "reason": reasons.loc[ts],
                  "policy": POLICY_VERSION, "series": name}
        path = root / "decisions" / name / f"{ts:%Y-%m}.csv"
        previous = _read_partition(path)
        combined = pd.concat([previous, pd.DataFrame([record])], ignore_index=True)
        combined = combined.drop_duplicates(["timestamp", "policy", "reason"], keep="last")
        _write_partition(path, combined)
        logger.warning("%s %s: raw=%s -> never_measured (%s, %s)",
                       name, ts, s.loc[ts], reasons.loc[ts], POLICY_VERSION)
    return s.mask(reject).rename(name)


def fingerprint(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def refresh(*, root: Path = STORE, now=None) -> None:
    """Fetch only the delta plus 72 h for late reports; persist before forecasting.

    Each station has its own cursor. A failure cannot advance it or poison another
    station's checkpoint. Long downtime is caught up in calendar-year chunks.
    """
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC").floor("h")
    cursor_path = root / "cursors.json"
    cursors = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    for station in sorted({s.station for s in SPECS.values()}):
        names = [n for n, s in SPECS.items() if s.station == station]
        water = SPECS[names[0]].field == "water"
        if station in cursors:
            last = pd.Timestamp(cursors[station])
        else:
            # Read the newest partition only, not sixteen years on every live run.
            paths = sorted((root / "hourly" / names[0]).glob("*.csv"))
            if not paths:
                raise RuntimeError(f"Archive missing {station}; run build_archive.py first")
            last = _read_partition(paths[-1]).timestamp.max()
        start = min(last, now) - pd.Timedelta(hours=72)
        zone = "Europe/Berlin" if water else "UTC"
        for year in range(start.tz_convert(zone).year, now.tz_convert(zone).year + 1):
            begin = max(start, pd.Timestamp(f"{year}-01-01", tz=zone).tz_convert("UTC"))
            end = min(now, (pd.Timestamp(f"{year + 1}-01-01", tz=zone)
                            - pd.Timedelta(hours=1)).tz_convert("UTC"))
            if water:
                url = GKD.format(station=station) + "/messwerte/tabelle"
                params = {"beginn": begin.tz_convert("Europe/Berlin").strftime("%d.%m.%Y"),
                          "ende": end.tz_convert("Europe/Berlin").strftime("%d.%m.%Y")}
            else:
                url = BRIGHTSKY_URL
                params = {"dwd_station_id": station, "date": begin.isoformat(), "last_date": end.isoformat()}
            response = request(url, params)
            digest = fingerprint(response.content)
            path = root / "raw" / station / "live" / f"{begin:%Y%m%d}-{end:%Y%m%d}-{digest[:16]}.gz"
            if not path.exists():
                save_response(response, path)
            for name in names:
                if water:
                    frame, issues = hourly_gkd(response.text)
                    if issues:
                        write_json(root / "ingestion" / name / f"{digest[:16]}.json", issues)
                else:
                    frame = hourly_weather(response.json(), station, SPECS[name].field)
                if not frame.empty:
                    frame = frame.loc[begin:end]
                    write_hourly(name, frame, root=root)
            write_json(path.with_suffix(".json"), {"url": url, "params": params, "fetched_at": now,
                                                   "sha256": digest})
        cursors[station] = now.isoformat()
        write_json(cursor_path, cursors)
    # Persist automated decisions for *all* covariates, even those not yet model channels.
    for name in SPECS:
        model_values(name, root=root, start=now - pd.Timedelta(days=4), end=now)


def prepare_live(*, root: Path = STORE):
    """Production entry: archived observations plus a short live forecast request."""
    from eisbach.data import (
        FORECAST_DAYS,
        HISTORY_DAYS,
        assemble_long_frame,
        get_prepared_weather_data,
    )

    now = pd.Timestamp.now(tz="UTC").floor("h")
    refresh(root=root, now=now)
    start = now - pd.Timedelta(days=HISTORY_DAYS)
    water = model_values("eisbach", root=root, start=start, end=now).rename("wassertemp")
    water = water.loc[:water.last_valid_index()]
    if water.empty or water.notna().sum() == 0:
        raise RuntimeError("No usable recent gauge measurements")
    # Historical weather is unfilled here: observation archives must never receive an
    # interpolated weather value. assemble_long_frame creates the separate model view.
    observed = pd.concat([model_values(n, root=root, start=start, end=now)
                          for n in ("airtemp", "pressure")], axis=1)
    observed = observed.rename(columns={"airtemp": "lufttemperatur_c"})
    live = get_prepared_weather_data(start_date=now.to_pydatetime(),
                                     end_date=(now + pd.Timedelta(days=FORECAST_DAYS)).to_pydatetime())
    live = live.tz_convert("UTC")
    # Never substitute a forecast for an unreported past observation.
    weather = pd.concat([observed.loc[:now], live.loc[live.index > now]]).sort_index()
    water.index.name = "timestamp"
    df_wt = water.reset_index()
    return assemble_long_frame(df_wt, weather), weather, df_wt


def load_window(names: list[str], start, end, *, root: Path = STORE) -> pd.DataFrame:
    """A usable model/research window, or an explicit refusal; never patch long gaps."""
    parts = [fill_short_gaps(model_values(n, root=root, start=start, end=end), n) for n in names]
    frame = pd.concat(parts, axis=1)
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Unusable window: missing or rejected values exceed the per-series gap policy")
    return frame
