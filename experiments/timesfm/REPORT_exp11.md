# exp11: 15-minute versus hourly forecasting

Run on Apple Silicon, 2026-09-14. Fix commit: `61e9fab00fe45de59c1e83270fa6b5393aa083bf` on `local/exp11-fixes`.
Base experiment commit: `f4b880dce0cb5616f2c618c5f78355d2959103a1`.
Backend: TimesFM 3.0.2 / MLX 0.32.2; checkpoint google/timesfm-3.0-pytorch.

## Interpretation

The 15-minute grid is not worth it in this experiment. Against the hourly model with
matched 160-day context, the paired MAE improvement is 0.57% and CRPS improvement 0.36%;
both confidence intervals include zero, including every individual lead bucket.
Against hourly with one year of context, fine-grid MAE is 18.27% worse and CRPS 19.32%
worse. At leads up to 24 hours the differences are not distinguishable from zero;
for 24–48, 48–72 and 72–96 hours both metrics are worse with unadjusted 95% paired
bootstrap intervals excluding zero. The secondary quarter-hour analysis also favours
hourly (interpolated), but remains indicative only.

The successful run took 61s / 26s / 109s for hourly-year / hourly-matched / fine.
There are 120 common windows and 11,518 observed full-hour targets per variant (two
of the nominal 11,520 measurements are missing). Scores are finite, rows unique and
scored counts identical across variants. Pooled tables weight observed hours; paired
comparisons average per-window scores, explaining the slight numerical differences.

Weather is hourly and interpolated to the fine grid. Future weather is observed, so
this is an oracle experiment, not proof of live forecast skill. The 384-step secondary
comparison interpolates hourly medians; its inference relies on that approximation.
All bootstrap intervals are paired and unadjusted, with n_tested=2 per reference and
metric; the extra lead comparisons are exploratory, not multiplicity-adjusted.

## Design preserved

- The hourly target is subsampled from the native 15-minute measurements at the full
  hour, not averaged. Both resolutions predict the same observations at the same instants.
- The headline comparison scores only the same 96 full-hour marks per window. The fine
  model's intermediate predictions are excluded from this comparison.
- Context is measured in steps: 15,360 quarter-hour steps = 160 days; the matched hourly
  model uses 3,840 steps, and the annual hourly model uses 8,760 steps.
- Past-only covariates are water temperature at Lenggries and Bad Tölz, and discharge
  at Lenggries and Bad Tölz KW. Future covariates are Munich air temperature and the
  mean catchment temperature at Bad Tölz, Lenggries, Kochel and Garmisch.
- Hourly weather is interpolated to the fine grid. It adds no finer weather observations.
- All comparisons use the same 120 anchors; bootstrap intervals use 8,000 paired draws
  and seed 0. `lead_lo` denotes buckets (0,6], (6,12], (12,18], (18,24], (24,48],
  (48,72], and (72,96] hours. Negative paired deltas favour the compared variant.

## Decision implications

Keep hourly resolution with one year of context as the reference for further research.
This run supplies no evidence that quarter-hour resolution buys accuracy at any tested
lead. Its observed inference cost was about 4.2 times the matched hourly model and
1.8 times the annual hourly model; these are run timings, not a controlled hardware
benchmark. Similar scores at matched context are not proof of strict equivalence.
The poorer comparison against annual context reflects the operational context limit
as well as the resolution choice; it does not isolate resolution alone.

Do not infer production readiness from this experiment. Besides the oracle-weather
qualification above, the README records the checkpoint's non-production/non-commercial
license. Backend agreement was checked on one window at each resolution, not across
all 120 windows. Do not treat the previously claimed 2e-6 agreement as verified here.

## Execution fix and validation

The original run produced NaN forecasts and then failed in bench.paired(). MLX 3.0.2
omits PyTorch's missing-input preparation. The exp11 fix trims leading missing targets
and interpolates input gaps exactly as PyTorch does, without changing the scoring
truth, anchors, frequencies, contexts or weather interpolation. Non-finite forecasts
now raise immediately. Original failure: exp11_failed_nan.log / exp11_failed_nan.csv.

Input preparation matched PyTorch exactly in the first selected window at both
resolutions. Maximum quantile differences after preparation: 0.00001764 C hourly,
0.00461984 C quarter-hourly. This is a spot check, not a full equivalence guarantee.

Validation: 189 passed, 2 skipped in the root test suite; 3 explicit experiment
regression tests passed; ruff check . and git diff --check passed. No production
files changed. The fix and this report are delivered on `local/exp11-fixes`; the
concurrently used `claude/sleepy-brown-950428` branch is not modified.

## Original experiment output (verbatim)

```
2026-09-14 13:55:15,812 120 anchors, 2023-01-19 17:00:00+00:00 .. 2026-09-09 23:00:00+00:00
2026-09-14 13:55:15,814 per year: {2023: 45, 2024: 21, 2025: 21, 2026: 33}
2026-09-14 13:55:15,889 TimesFM on the MLX backend
2026-09-14 13:56:20,182 hourly, 1 year of context        61s
2026-09-14 13:56:45,776 hourly, 160 days (matched)       26s
2026-09-14 13:58:34,391 15-minute, 160 days (its max)    109s

=== scored at the same 96 full-hour marks, 120 windows ===
                        label     n  runs      mae     crps   cov_80  width_80
    hourly, 1 year of context 11518   120 0.357749 0.232935 0.813857  1.161403
15-minute, 160 days (its max) 11518   120 0.423076 0.277911 0.800833  1.355486
   hourly, 160 days (matched) 11518   120 0.425483 0.278910 0.766018  1.267068

--- MAE against 'hourly, 160 days (matched)', paired ---
                        label      mae        pct     ci_lo     ci_hi  better_in verdict
    hourly, 1 year of context 0.357789 -15.930528 -0.098952 -0.036056   0.641667  better
15-minute, 160 days (its max) 0.423165  -0.569048 -0.025167  0.020320   0.516667       —

--- MAE against 'hourly, 1 year of context', paired ---
                        label      mae       pct    ci_lo    ci_hi  better_in verdict
15-minute, 160 days (its max) 0.423165 18.272364 0.029483 0.100465   0.350000   worse
   hourly, 160 days (matched) 0.425587 18.949242 0.036056 0.098952   0.358333   worse

--- CRPS against 'hourly, 160 days (matched)', paired ---
                        label     crps        pct     ci_lo     ci_hi  better_in verdict
    hourly, 1 year of context 0.232962 -16.495623 -0.065805 -0.026233   0.658333  better
15-minute, 160 days (its max) 0.277970  -0.362776 -0.015228  0.013326   0.475000       —

--- CRPS against 'hourly, 1 year of context', paired ---
                        label     crps       pct    ci_lo    ci_hi  better_in verdict
15-minute, 160 days (its max) 0.277970 19.319763 0.022943 0.066938   0.358333   worse
   hourly, 160 days (matched) 0.278982 19.754202 0.026233 0.065805   0.341667   worse

--- MAE by lead bucket ---
lead_lo                           0      6      12     18     24     48     72
label                                                                         
15-minute, 160 days (its max)  0.147  0.246  0.293  0.303  0.417  0.475  0.553
hourly, 1 year of context      0.144  0.246  0.280  0.293  0.370  0.394  0.426
hourly, 160 days (matched)     0.146  0.251  0.293  0.308  0.423  0.472  0.557

--- all 384 quarter-hour marks, 120 windows (indicative) ---
  15-minute forecast      MAE 0.4210
  hourly, interpolated    MAE 0.3565
  difference +0.0645 (+18.1 %), 95 % CI [+0.0284, +0.0993]
```

## Supplemental tables (from the same result CSV)

```
Verified: finite scores, unique rows, identical scored counts across all variants.

--- CRPS by lead bucket ---
lead_lo                           0      6      12     18     24     48     72
label                                                                         
15-minute, 160 days (its max)  0.096  0.162  0.199  0.207  0.275  0.313  0.357
hourly, 1 year of context      0.095  0.159  0.188  0.195  0.240  0.255  0.277
hourly, 160 days (matched)     0.095  0.164  0.199  0.207  0.275  0.309  0.365

--- Full paired MAE against 'hourly, 160 days (matched)' ---
                        label  n_runs      mae     delta        pct     ci_lo     ci_hi  better_in verdict  n_tested
    hourly, 1 year of context     120 0.357789 -0.067798 -15.930528 -0.098952 -0.036056   0.641667  better         2
15-minute, 160 days (its max)     120 0.423165 -0.002422  -0.569048 -0.025167  0.020320   0.516667       —         2

--- Paired MAE by lead, fine minus reference (unadjusted CIs) ---
 lead_lo  n_runs      mae     delta       pct     ci_lo    ci_hi  better_in verdict  n_tested
       0     120 0.146820  0.000952  0.652353 -0.007989 0.010123   0.500000       —         2
       6     120 0.245695 -0.005402 -2.151349 -0.023534 0.011782   0.466667       —         2
      12     120 0.293056  0.000540  0.184641 -0.020286 0.021419   0.508333       —         2
      18     120 0.303025 -0.005398 -1.750037 -0.030662 0.017909   0.450000       —         2
      24     120 0.417356 -0.005856 -1.383684 -0.032589 0.020734   0.508333       —         2
      48     120 0.475232  0.002983  0.631752 -0.032932 0.039900   0.483333       —         2
      72     120 0.552591 -0.004501 -0.807883 -0.042410 0.033833   0.541667       —         2

--- Full paired MAE against 'hourly, 1 year of context' ---
                        label  n_runs      mae    delta       pct    ci_lo    ci_hi  better_in verdict  n_tested
15-minute, 160 days (its max)     120 0.423165 0.065376 18.272364 0.029483 0.100465   0.350000   worse         2
   hourly, 160 days (matched)     120 0.425587 0.067798 18.949242 0.036056 0.098952   0.358333   worse         2

--- Paired MAE by lead, fine minus reference (unadjusted CIs) ---
 lead_lo  n_runs      mae    delta       pct     ci_lo    ci_hi  better_in verdict  n_tested
       0     120 0.146820 0.002417  1.673522 -0.008997 0.013548   0.425000       —         2
       6     120 0.245695 0.000173  0.070616 -0.023327 0.023166   0.516667       —         2
      12     120 0.293056 0.013350  4.772754 -0.012792 0.039460   0.441667       —         2
      18     120 0.303025 0.010444  3.569693 -0.019505 0.040111   0.483333       —         2
      24     120 0.417356 0.047132 12.730545  0.007699 0.085899   0.433333   worse         2
      48     120 0.475232 0.081499 20.699025  0.024515 0.138047   0.391667   worse         2
      72     120 0.552591 0.126159 29.584792  0.063133 0.190516   0.375000   worse         2

--- Full paired CRPS against 'hourly, 160 days (matched)' ---
                        label  n_runs     crps     delta        pct     ci_lo     ci_hi  better_in verdict  n_tested
    hourly, 1 year of context     120 0.232962 -0.046020 -16.495623 -0.065805 -0.026233   0.658333  better         2
15-minute, 160 days (its max)     120 0.277970 -0.001012  -0.362776 -0.015228  0.013326   0.475000       —         2

--- Paired CRPS by lead, fine minus reference (unadjusted CIs) ---
 lead_lo  n_runs     crps     delta       pct     ci_lo    ci_hi  better_in verdict  n_tested
       0     120 0.096096  0.000920  0.966907 -0.004520 0.006629   0.533333       —         2
       6     120 0.161686 -0.001905 -1.164789 -0.012441 0.008606   0.458333       —         2
      12     120 0.198977  0.000361  0.181580 -0.012217 0.012978   0.458333       —         2
      18     120 0.206554 -0.000538 -0.259661 -0.014976 0.013414   0.366667       —         2
      24     120 0.274897 -0.000444 -0.161124 -0.016153 0.015330   0.466667       —         2
      48     120 0.313404  0.003975  1.284482 -0.017931 0.026786   0.500000       —         2
      72     120 0.357512 -0.007292 -1.998915 -0.032294 0.017537   0.541667       —         2

--- Full paired CRPS against 'hourly, 1 year of context' ---
                        label  n_runs     crps    delta       pct    ci_lo    ci_hi  better_in verdict  n_tested
15-minute, 160 days (its max)     120 0.277970 0.045008 19.319763 0.022943 0.066938   0.358333   worse         2
   hourly, 160 days (matched)     120 0.278982 0.046020 19.754202 0.026233 0.065805   0.341667   worse         2

--- Paired CRPS by lead, fine minus reference (unadjusted CIs) ---
 lead_lo  n_runs     crps    delta       pct     ci_lo    ci_hi  better_in verdict  n_tested
       0     120 0.096096 0.001026  1.079094 -0.005721 0.007701   0.450000       —         2
       6     120 0.161686 0.002431  1.526721 -0.011249 0.015818   0.500000       —         2
      12     120 0.198977 0.011150  5.936213 -0.004280 0.026566   0.425000       —         2
      18     120 0.206554 0.011644  5.973778 -0.004316 0.027947   0.416667       —         2
      24     120 0.274897 0.034563 14.381298  0.011127 0.057976   0.408333   worse         2
      48     120 0.313404 0.058130 22.771579  0.024104 0.092491   0.375000   worse         2
      72     120 0.357512 0.080689 29.148392  0.042658 0.120033   0.366667   worse         2
```

## Reproduction and local artifacts

Follow `HANDOFF_exp11.md` for dataset construction, using the fix on this branch.
The builders produced 28 hourly gauge columns (the handoff says 29), 18 hourly weather
columns across six sites, and five quarter-hour gauge columns. exp11 directly reads
only `stations_15min.csv` and `weather_south.csv`.

The run used Python 3.12.13, TimesFM 3.0.2, MLX 0.32.2, NumPy 2.5.3 and pandas 3.0.5.
PyTorch 2.14.0 was used for the backend spot check. From the repository root:

```bash
export TIMESFM_BACKEND=mlx
export HF_HOME="$PWD/data/experiments/hf"
python experiments/timesfm/exp11_resolution.py
pytest -q
pytest -q experiments/timesfm/test_exp11_resolution.py
ruff check .
```

For the supplemental CRPS table, load `data/experiments/exp11_resolution.csv` and use
`bench.pool(scores, by=["label", "lead_lo"])`, pivoting `crps` by label and lead.
For paired lead intervals, group the same scores by `lead_lo` and call
`bench.paired(group, reference, metric="mae")` or `metric="crps"` with each hourly
reference. No new forecasts or alternative anchors are used for these supplements.

Datasets, result CSVs, complete package freeze (`exp11_requirements.txt`), logs and the
supplemental analysis script remain in the local, gitignored `data/experiments/` cache.
They are not included in the branch. All decision tables are embedded above, so reading
this report does not require access to that cache. Fresh source downloads may change;
the fingerprints below identify the actual run inputs and scores.

## Input fingerprints (SHA-256)

- `stations_hourly.csv`: `731d84985990ad581b27f2b512ec0e33364267a198273f9c2348ea4df8327bff`
- `weather_south.csv`: `4095944d49a97dec6fdd908edd07d4692374d645bb9f1ac5fac7ba48b068e0ae`
- `stations_15min.csv`: `1faa670334f6894267c8d73c75d9223cc6c91d568f6cf39cff29e05316dc6c50`
- `exp11_resolution.csv`: `4a2431781f7d7f2e6c234d88b9d1614550f282eb71ad4e7480b1045835bb87a6`
