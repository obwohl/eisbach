"""Rewrite the hourly covariate partitions in the slim schema, proving nothing is lost.

`_encode_partition` changed what a partition stores, not what it means: series-constant
columns are gone, cells equal to their default are blank, and timestamps are written as
`2013-07-01T00Z`. Reading is backward compatible, so this migration is an optimisation,
not a correctness fix — it exists because 124 MB of the archive is the hourly store and
roughly half of it was constants repeated 1.9 million times.

Every partition is decoded before and after and the two frames are compared cell by cell.
A single mismatch aborts the whole run before anything is replaced, so a partition is
either provably identical in meaning or untouched.

    python3 scripts/slim_hourly_partitions.py [--apply]

Without `--apply` it only reports. `data/archive/covariates/raw/` is not touched: those
payloads are the irreplaceable copy, and `hourly/` is derived from them.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

from eisbach.covariates import (
    SPECS,
    STORE,
    _encode_partition,
    _read_hourly_partition,
    _write_partition,
)

logger = logging.getLogger("slim")


def equivalent(before: pd.DataFrame, after: pd.DataFrame) -> str | None:
    """Return the first reason the two decoded frames differ, or None."""
    if list(before.columns) != list(after.columns):
        return f"columns {list(before.columns)} != {list(after.columns)}"
    if len(before) != len(after):
        return f"{len(before)} rows became {len(after)}"
    for col in before.columns:
        left, right = before[col], after[col]
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            left, right = pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")
            differs = ~((left == right) | (left.isna() & right.isna()))
        else:
            left, right = left.astype("string"), right.astype("string")
            differs = ~((left == right) | (left.isna() & right.isna()))
        if differs.any():
            row = int(differs.to_numpy().argmax())
            return f"{col} row {row}: {before[col].iloc[row]!r} -> {after[col].iloc[row]!r}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="replace the partitions")
    parser.add_argument("--root", type=Path, default=STORE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    paths = [(name, path) for name in sorted(SPECS)
             for path in sorted((args.root / "hourly" / name).glob("*.csv"))]
    before_bytes = after_bytes = 0
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "hourly"
        # Verify every partition first. Applying nothing until all 2560 have passed is
        # what makes this safe to run against the real store.
        for name, path in paths:
            before = _read_hourly_partition(path, name)
            if before.empty:
                continue
            target = probe / name / path.name
            _write_partition(target, _encode_partition(before))
            # Read the rewritten file back through the ordinary reader, so the check is
            # what a caller would actually get, not what the encoder intended.
            after = _read_hourly_partition(target, name)
            reason = equivalent(before.reset_index(drop=True), after.reset_index(drop=True))
            if reason is not None:
                raise SystemExit(f"{path}: refusing to rewrite — {reason}")
            before_bytes += path.stat().st_size
            after_bytes += target.stat().st_size

    logger.info("%d partitions, %.1f MB -> %.1f MB (%.0f %% smaller)", len(paths),
                before_bytes / 1e6, after_bytes / 1e6,
                100 * (1 - after_bytes / before_bytes) if before_bytes else 0)
    if not args.apply:
        logger.info("dry run; pass --apply to replace them")
        return 0
    for name, path in paths:
        frame = _read_hourly_partition(path, name)
        if not frame.empty:
            _write_partition(path, _encode_partition(frame))
    logger.info("rewrote %d partitions", len(paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
