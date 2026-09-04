# Notes for agents working in this repository

## What this is

A thrice-daily probabilistic forecast of the Eisbach's water temperature. `main.py`
fetches, forecasts, validates and plots; everything else supports that.

```
main.py               orchestration only — keep logic out of it
eisbach/data.py       GKD scraping, Bright Sky fetch, feature assembly
eisbach/inference.py  forecast + the three-track backtest resolution
eisbach/archive.py    provenance-tracked storage
eisbach/validate.py   post-run plausibility gate
eisbach/plotting.py   matplotlib output
eisbach/model/        vendored inference code — see below
tests/                the whole suite; must not touch the network
data/archive/         every forecast ever made — tracked, irreplaceable, append-only
data/model/           checkpoint cache, fetched on demand and gitignored
research/             unmaintained research code, never imported or run
docs/PRD.md           what this is for, what it guarantees, what is still open
docs/model.md         how the model works, and what is inert in the vendored tree
docs/page/            template for the published site
.github/workflows/    the thrice-daily forecast run and the CI check
```

## Things that will bite you

**`eisbach/model/duet/` is vendored verbatim** from `obwohl/ts_proba_cuda` at commit
`a8de694`. Keep it byte-identical to upstream so it stays diffable — it is excluded from
ruff for that reason. Upstream HEAD is **not** compatible with this checkpoint. Read
`eisbach/model/PROVENANCE.md` before touching anything in there.

**The channel order is load-bearing.** `("wassertemp", "airtemp_96", "pressure_96")` is
the order the model was trained on, and the projection heads are indexed positionally.
Reordering it produces plausible-looking nonsense rather than an error.

**The `_96` suffix is not a typo.** Weather covariates are shifted 96 hours backwards, so
that at any timestamp the model sees the weather four days *ahead*. That shift is what
lets a 96-hour horizon use a weather forecast at all. `COVARIATE_SHIFT_HOURS` and the
model's `horizon` must stay equal.

**Backtests are not all equally honest.** `live` is a forecast we really made; `replay`
is rebuilt from the DWD forecast as it was issued; `oracle` uses the weather that
actually occurred and therefore flatters the model. The precedence `live > replay >
oracle` is enforced on write so a regenerable row can never overwrite a genuine one.
Never present an oracle backtest as evidence of real-world skill, and never widen the
weather-snapshot lookup to include snapshots issued *after* the reference time — that is
a leak, just a smaller one.

**`data/archive/` is irreplaceable.** It is the only copy of every forecast ever made and
of the DWD forecasts as they were issued. GKD serves only a rolling window, so anything
not captured is eventually unrecoverable. Treat it as append-only.

**Seven weather snapshots are already lost.** In these seven, the `timestamp` column is
completely empty — 840 of 1560 rows, values present but index gone, so they can never be
replayed and, historical DWD forecasts being a paid product, never refetched. By
`archive_timestamp`: `2026-06-06T10:02:52`, `2026-06-18T22:03:10`, `2026-06-23T22:03:14`,
`2026-07-08T22:03:22`, `2026-07-13T22:03:31`, `2026-07-18T22:02:52`, `2026-07-23T22:03:05`.
Do not spend time trying to recover them, and do not read a replay gap in that window as a
bug in the lookup.

**Generated outputs do not belong in git.** The PNGs and CSV are deployed to GitHub Pages
via `actions/upload-pages-artifact`, which bypasses git entirely, so republishing them
three times a day costs no history. Committing them is what grew `.git` to 175 MB.

## Conventions

- English throughout, except the German domain names above, which are the model's trained
  column names and must not be translated.
- `logging`, not `print`.
- Errors raise. The predecessor to this code caught everything, printed a message and
  exited 0, so failures were invisible to the caller — do not reintroduce that.
- Tests must not touch the network. The one exception is the model checkpoint, fetched
  once and cached.

## Before you commit

```bash
pytest -q
ruff check .
```
