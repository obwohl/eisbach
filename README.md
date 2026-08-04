# Eisbach Forecast

A 96-hour probabilistic forecast of the water temperature of the Eisbach in Munich,
regenerated twice a day.

**→ [obwohl.github.io/eisbach](https://obwohl.github.io/eisbach/)**

![Latest forecast](https://obwohl.github.io/eisbach/Prediction.png)

## The model

A **DUET-Prob** network I built and trained for this river. Not a foundation model with a
wrapper around it.

It routes between linear experts and **Echo State Network** experts — reservoir models
with a fixed random recurrent state that the network learns to read out. They handle
chaotic dynamics well, which is what a shallow concrete channel in changing weather
produces. The output is a Student-t distribution, so the quantile bands mean something.

| | |
| --- | --- |
| Parameters | 2.7 M |
| Context → horizon | 384 h → 96 h |
| Inputs | water temperature, air temperature, air pressure |
| Output | 7 quantiles, 1 % to 99 % |

### Against a foundation model

Ten backtest windows, identical history, same perfect weather for both:

| | MAE ↓ | CRPS ↓ |
| --- | --- | --- |
| **This model** | **0.450** | **0.311** |
| Chronos-2 (Amazon), multivariate | 0.624 | 0.420 |

The gap widens on volatile windows, where Chronos-2 answers uncertainty by widening its
intervals until they stop saying anything. A model that has learned one river's
thermodynamics can commit to a narrow band; a zero-shot model cannot.

## Backtests you can trust

Every plot shows backtests at −96 h, −192 h and −288 h. They are not equally honest, and
the archive records which is which:

| Kind | Future weather came from | Honest? |
| --- | --- | --- |
| `live` | the forecast available at the time | yes |
| `replay` | an archived forecast, as issued | yes |
| `oracle` | the weather that actually occurred | **no** — flatters the model |

`live` is preferred and free: it is a forecast we really made, retrieved rather than
recomputed. `oracle` is the last resort and is drawn dashed and labelled, so a
too-good-looking backtest is never mistaken for real-world skill.

## Data

| Input | Source |
| --- | --- |
| Water temperature | [GKD Bayern](https://www.gkd.bayern.de), gauge *München Himmelreichbrücke* |
| Weather | [Bright Sky](https://brightsky.dev) / DWD, station `03379` |

Weather covariates are shifted 96 hours backwards, so at any timestamp the model sees the
weather four days ahead — the known-future information a forecast provides.

## Running it

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU-only; skip if you want CUDA
pip install -e ".[dev]"
python main.py
```

The 10 MB checkpoint downloads on first run and is verified against a pinned SHA256.
Inference is CPU-only and the pipeline takes under a minute.

## Layout

```
main.py            entrypoint
eisbach/data.py    data sources and feature assembly
eisbach/model/     the forecasting model
eisbach/archive.py forecast storage, with provenance
eisbach/validate.py plausibility gate — a bad forecast fails the run
eisbach/plotting.py output
archive/           earlier research, unmaintained
docs/              operational notes
```

`docs/page/` is the template for the published site; `data/archive/` holds every forecast
ever made and the DWD forecasts as they were issued — it is the only copy of both.

## Licence

MIT. The model code descends from the DUET and Autoformer time-series work; see
`eisbach/model/PROVENANCE.md` for what is inherited and what is mine.
