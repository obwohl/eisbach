# Eisbach Forecast

A 96-hour probabilistic forecast of the water temperature of the Eisbach in Munich,
published as a plot and a quantile table twice a day.

![Latest forecast](https://raw.githubusercontent.com/obwohl/eisbach/outputs/Prediction.png)

*Regenerated twice a day. The plots live on the [`outputs`](https://github.com/obwohl/eisbach/tree/outputs)
branch, which is replaced wholesale each run — keeping 1 GB a year of regenerable PNGs in
`main`'s history was how this repository got to a 175 MB `.git`.*

## What it does

Every run scrapes the last 40 days of measured water temperature, pulls the DWD weather
forecast for the next 8 days, and predicts the next 96 hours as seven quantiles
(1 %, 5 %, 25 %, 50 %, 75 %, 95 %, 99 %).

| Input | Source |
| --- | --- |
| Water temperature, hourly | [GKD Bayern](https://www.gkd.bayern.de), gauge *München Himmelreichbrücke* |
| Air temperature, pressure, precipitation | [Bright Sky](https://brightsky.dev) (DWD station `03379`) |

The trick that makes a 96-hour horizon work is that the weather covariates are shifted
backwards by 96 hours before they reach the model, as `airtemp_96` and `pressure_96`. At
any timestamp the model therefore sees the weather *four days ahead* of that point, which
is exactly the known-future information a weather forecast provides. Precipitation is
fetched and archived but is **not** currently a model input.

## The model

The forecast does not come from a foundation model with a wrapper around it. It comes
from a **DUET-Prob** network I built and trained myself, specifically for this river.

The interesting part is the expert mixture. Water temperature in a shallow concrete
channel is chaotic and strongly weather-driven, and a generic sequence model handles that
badly — it hedges. This one routes between two linear experts and nine **Echo State
Network** experts, which are reservoir models: a fixed random recurrent state that the
network learns to read out rather than to train through. They are unusually well suited
to chaotic dynamics, and using them as mixture components on weather-coupled data is the
part of this project I would actually defend at a whiteboard.

The output head is a Student-t distribution rather than a point estimate, so the forecast
is a full predictive distribution and the quantile bands mean something.

| | |
| --- | --- |
| Parameters | 2,677,319 |
| Context → horizon | 384 h (16 days) → 96 h (4 days) |
| Channels | `wassertemp`, `airtemp_96`, `pressure_96` |
| Output | Student-t distribution, sampled at 7 quantiles |

The training code lives in [`obwohl/ts_proba_cuda`](https://github.com/obwohl/ts_proba_cuda),
my fork of a time-series benchmarking project, developed well past its origin. This
repository vendors only the ~150 KB of it that inference actually reaches — see
`eisbach/model/PROVENANCE.md` for the exact commit and what was stripped. The 10 MB
checkpoint is downloaded on first use and verified against a pinned SHA256.

The checkpoint in production is deliberately not the newest one. Later training runs
explored further but forecast worse on this river, so the pin sits on the configuration
that actually won its backtests.

**Against a foundation model.** Amazon's Chronos-2 was evaluated head to head, over 10
backtest windows, both models given identical history and the same perfect weather:

| | MAE | CRPS |
| --- | --- | --- |
| This model | **0.450** | **0.311** |
| Chronos-2, multivariate | 0.624 | 0.420 |

The gap widens on the volatile windows, where Chronos-2 responds to uncertainty by
widening its intervals until they stop saying anything (CRPS 1.220 against 0.277 on the
worst case). A model that has learned one river's thermodynamics can commit to a narrow
band; a zero-shot model cannot. The comparison code and full results are in `archive/`,
along with an AutoGluon / Chronos-Bolt arm that is **parked, not abandoned**.

## Backtests, and how honest they are

The plots show the current forecast alongside backtests at −96 h, −192 h and −288 h.
Those backtests are not all of equal quality, so the archive records which kind each one
is (see `eisbach/archive.py`):

| Kind | Future covariates come from | Honest? | Reproducible? |
| --- | --- | --- | --- |
| `live` | the DWD forecast available at the time | yes | no — must be stored |
| `replay` | an archived DWD forecast snapshot | yes | only where a snapshot exists |
| `oracle` | the weather that actually occurred | **no** | always |

`live` is preferred because it is both free (already computed) and honest. `oracle` is
the last resort: it hands the model a perfect weather forecast, so it flatters the
result and must not be read as evidence of real-world skill. Historical DWD *forecasts*
are a paid product we do not have, which is why the oracle fallback exists at all.

Every archived row carries its `kind`, `covariate_source`, `model_id` and `code_version`,
so a stored forecast can always be attributed to the code and checkpoint that produced it.

## Outputs

Regenerable outputs go to the `outputs` branch; the archive is the only copy of its data
and stays in `main`.

| File | Where | Contents |
| --- | --- | --- |
| `Prediction.png` | `outputs` | The forecast, with history and air temperature |
| `Prediction_Backtest.png` | `outputs` | The same, plus the three backtests |
| `Prediction.csv` | `outputs` | The forecast as quantiles, in local time |
| `data/archive/forecasts/YYYY-MM.csv` | `main` | Every forecast ever made, with provenance |
| `data/archive/weather/YYYY-MM.csv` | `main` | DWD forecasts as issued, enabling `replay` |
| `data/archive/observations/YYYY-MM.csv` | `main` | Measured values, so verification needs no re-scrape |

## Running it

```bash
pip install -e ".[dev]"
python main.py
```

The checkpoint downloads automatically on first run (~10 MB, cached in `data/model/`).
Inference is CPU-only and the whole pipeline takes well under a minute.

```bash
pytest          # offline, apart from that one checkpoint fetch
ruff check .
```

## Layout

```
main.py                 pipeline entrypoint
eisbach/data.py         scraping, weather fetch, feature construction
eisbach/inference.py    forecast and backtest orchestration
eisbach/archive.py      provenance-tracked storage
eisbach/plotting.py     matplotlib output
eisbach/model/          vendored DUET-Prob inference code
tests/                  unit tests
archive/                unmaintained research code, kept for the record only
docs/                   operational notes
TODO.md                 audit findings and the cleanup plan
```
