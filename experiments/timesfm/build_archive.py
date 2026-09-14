"""Resumable full-source backfill and exhaustive machine-readable quality report.

Run from the repository root: python experiments/timesfm/build_archive.py
Successful chunks are cached; failed requests abort and can be retried safely.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eisbach.covariates import (  # noqa: E402
    GKD,
    SPECS,
    STORE,
    assess_frame,
    classify,
    fingerprint,
    hourly_gkd,
    hourly_weather,
    read_hourly,
    request,
    runs,
    save_response,
    write_hourly,
    write_json,
)

logger = logging.getLogger(__name__)


def cached(url, params, path):
    if not path.exists():
        response = request(url, params)
        save_response(response, path)
    return gzip.decompress(path.read_bytes())


def build(root=STORE):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    manifest = []
    stations = sorted({s.station for s in SPECS.values()})
    for station in stations:
        names = [n for n, s in SPECS.items() if s.station == station]
        water = SPECS[names[0]].field == "water"
        meta_url = GKD.format(station=station) if water else "https://api.brightsky.dev/sources"
        params = None if water else {"dwd_station_id": station}
        meta_path = root / "raw" / station / "metadata.gz"
        meta = cached(meta_url, params, meta_path)
        if water:
            text = BeautifulSoup(meta, "html.parser").get_text(" ", strip=True)
            match = re.search(r"Beobachtet seit\s+(\d{4})", text)
            if match is None:
                raise ValueError(f"No source start date for {station}")
            first = int(match[1])
        else:
            sources = json.loads(meta)["sources"]
            starts = [pd.Timestamp(s["first_record"]) for s in sources
                      if s.get("first_record") and s["observation_type"] != "forecast"
                      and str(s["dwd_station_id"]).zfill(5) == station]
            first = min(starts).year
        for year in range(first, now.year + 1):
            logger.info("%s: %d (source start %d)", station, year, first)
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = min(pd.Timestamp(f"{year}-12-31 23:00", tz="UTC"), now)
            if water:
                url = GKD.format(station=station) + "/messwerte/tabelle"
                last_day = (now.tz_convert("Europe/Berlin").strftime("%d.%m.%Y")
                            if year == now.year else f"31.12.{year}")
                params = {"beginn": f"01.01.{year}", "ende": last_day}
            else:
                url = "https://api.brightsky.dev/weather"
                params = {"dwd_station_id": station, "date": start.isoformat(), "last_date": end.isoformat()}
            key = f"{year}-calendar" if water else str(year)
            if year == now.year:
                key += f"-through-{now:%Y%m%dT%H}"
            path = root / "raw" / station / f"{key}.gz"
            content = cached(url, params, path)
            for name in names:
                if water:
                    frame, issues = hourly_gkd(content.decode())
                    write_json(root / "ingestion" / name / f"{year}.json", issues)
                else:
                    frame = hourly_weather(json.loads(content), station, SPECS[name].field)
                # GKD dates are local, so Jan 1 can start in the preceding UTC year.
                if not frame.empty:
                    frame = frame.loc[:now]
                    write_hourly(name, frame, root=root)
                manifest.append({"series": name, "station": station, "year": year,
                                 "source_start_year": first, "url": url, "params": params,
                                 "sha256": fingerprint(content), "raw_path": str(path),
                                 "hours": len(frame), "present": int(frame.raw_value.notna().sum())
                                 if not frame.empty else 0,
                                 "retrieved_at": pd.Timestamp(
                                     path.stat().st_mtime, unit="s", tz="UTC").isoformat()})
            write_json(root / "manifest.json", manifest)
    report(root)


def report(root=STORE):
    directory = root / "audit"
    directory.mkdir(parents=True, exist_ok=True)
    summary, findings, gaps, yearly = [], [], [], []
    manifest = json.loads((root / "manifest.json").read_text())
    audit_end = max(pd.Timestamp(r["params"]["last_date"]) for r in manifest
                    if "last_date" in r["params"])
    for name in SPECS:
        frame = read_hourly(name, root=root)
        if frame.empty:
            raise ValueError(f"Missing entire series {name}")
        if audit_end > frame.index.max():
            frame = frame.reindex(pd.date_range(frame.index.min(), audit_end, freq="h"))
        s = frame.raw_value
        for year, part in s.groupby(s.index.year):
            spacing = part.dropna().index.to_series().diff().dt.total_seconds().div(3600)
            yearly.append({"series": name, "year": year, "grid_hours": len(part),
                           "measured": int(part.notna().sum()), "missing": int(part.isna().sum()),
                           "median_observation_spacing_h": spacing.median()})
        valid = s.dropna()
        flags = assess_frame(frame, name)
        before, after = s.shift(1), s.shift(-1)
        for test in flags:
            if test == "missing":
                continue
            # Runs give every timestamp without listing identical plateau rows thousands of times.
            intervals = runs(flags[test])
            if test == "frozen":
                groups = s.ne(s.shift()).cumsum()
                intervals = ((part.index[0], part.index[-1], len(part))
                             for _, part in s[flags[test]].groupby(groups[flags[test]]))
            for start, end, count in intervals:
                part = s.loc[start:end]
                findings.append({"series": name, "test": test, "start": start, "end": end,
                                 "hours": count, "min": part.min(), "max": part.max(),
                                 "before": before.loc[start], "after": after.loc[end],
                                 "classification": classify(name, test, part.iloc[0])})
        these_gaps = []
        for start, end, count in runs(s.isna()):
            gap = {"series": name, "start": start, "end": end, "hours": count,
                   "location": "edge" if start == s.index.min() or end == s.index.max() else "internal",
                   "classification": classify(name, "missing", float("nan"))}
            if name == "loisach_beuerberg" and end.year < 2008 and count in {22, 23, 24}:
                gap["classification"] = "daily_sampling_pattern_not_an_hourly_instrument_outage"
            if name in {"eisbach", "isar_toelz", "loisach_beuerberg"} and count == 2:
                local = start.tz_convert("Europe/Berlin")
                if local.month == 10 and local.dayofweek == 6 and local.hour == 2:
                    gap["classification"] = "DST_fold_unresolvable_local_timestamps"
            gaps.append(gap)
            these_gaps.append(gap)
        summary.append({"series": name, "station": SPECS[name].station,
                        "grid_start": s.index.min(), "grid_end": s.index.max(),
                        "first_value": valid.index.min(), "last_value": valid.index.max(),
                        "hours": len(s), "measured": len(valid), "missing": int(s.isna().sum()),
                        "min": valid.min(), "max": valid.max(), "gaps": len(these_gaps),
                        "longest_gap_h": max((g["hours"] for g in these_gaps), default=0),
                        "flagged_intervals": sum(f["series"] == name for f in findings)})
    for filename, rows in [("summary", summary), ("findings", findings), ("gaps", gaps), ("yearly", yearly)]:
        pd.DataFrame(rows).to_csv(directory / f"{filename}.csv", index=False)
    lines = ["# Covariate archive audit", "", f"Generated {pd.Timestamp.now(tz='UTC')}. All times UTC.", "",
             "## Coverage", "", "| Series | First measurement | Last measurement | "
             "Measured / grid hours | Gaps | Longest h |",
             "|---|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['series']} | {row['first_value']} | {row['last_value']} | "
                     f"{row['measured']} / {row['hours']} | {row['gaps']} | {row['longest_gap_h']} |")
    lines += ["", "Full per-series ranges and counts: "
              "[summary.csv](../../data/archive/covariates/audit/summary.csv).",
              "Every suspicious interval, inclusive endpoints, extrema and adjacent values: "
              "[findings.csv](../../data/archive/covariates/audit/findings.csv).",
              "Every gap, including leading/trailing non-reporting intervals: "
              "[gaps.csv](../../data/archive/covariates/audit/gaps.csv).", "",
              "These CSV annexes are an integral part of this report; "
              "no finding is sampled or discarded.", ""]
    gap_frame = pd.DataFrame(gaps)
    gap_frame["length_bin"] = pd.cut(gap_frame.hours, bins=[0, 1, 6, 24, 168, float("inf")],
                                     labels=["1h", "2-6h", "7-24h", "25-168h", ">168h"])
    histogram = gap_frame.groupby(["series", "length_bin"], observed=False).size().unstack(fill_value=0)
    histogram.to_csv(directory / "gap_histogram.csv")
    lines += ["## Gap frequencies", "", "| Series | 1h | 2–6h | 7–24h | 25–168h | >168h |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in histogram.iterrows():
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Policy (applies to backfill and live reads)", "",
              "| Series | Source | Unit | Review range | Jump / h | Hampel floor | Fill internal gap ≤ h |",
              "|---|---|---|---|---:|---:|---:|"]
    for name, spec in SPECS.items():
        lines.append(f"| {name} | {spec.station} | {spec.unit} | {spec.bounds} | "
                     f"{spec.jump} | {spec.floor} | {spec.fill_hours} |")
    lines += ["", (REPO / "experiments/timesfm/archive_method.md").read_text()]
    (REPO / "experiments/timesfm/REPORT_archive.md").write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        report()
    else:
        build()
