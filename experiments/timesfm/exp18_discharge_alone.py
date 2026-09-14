"""Experiment 18 — the discharge question, asked without the weather in the room.

exp12 walked six measuring points and found the discharge worthless: on its own
indistinguishable from zero, and next to its own water temperature worth −0.4 % to
−0.7 % with every interval crossing zero and "better in" at 51–54 %, a coin flip.

But exp12 always asked with `airtemp` and `t_catchment` already handed over. That is the
question "does the discharge add anything **to a model that already knows the weather**",
and it is not the same question as "does the discharge carry information about the
Eisbach". Air temperature drives both the Isar's temperature and, through snowmelt and
rain, a good deal of its volume, so it can stand in for a covariate that is genuinely
informative and leave it looking redundant.

The intuition being tested is a physical one and a reasonable one: 50 m³/s at 11 °C and
200 m³/s at 11 °C are very different amounts of heat arriving downstream, and it is hard
to believe the quantity is irrelevant. Nothing so far has actually cornered it.

So this strips the model back to the river:

* `nur Eisbach` — univariate, nothing but its own history
* `+ T` — the Bad Tölz water temperature
* `+ Q` — the Bad Tölz discharge, without the temperature
* `+ T + Q` — both

and then runs the identical four **with** the settled weather, so the two readings sit
side by side. If the discharge helps bare and stops helping once the weather is present,
that is an answer — redundancy, not irrelevance — and a different one from exp12's.

Names, because they matter here: `isar_toelz` is the **water temperature** at the Bad Tölz
measuring point; `q_toelz_kw` is the **discharge**, from the Bad Tölz power-station gauge.

The heat flux `Q × T` is in here too, as the one exception to the no-invented-features
rule. It is not an invented weighting: it is a physical quantity with a unit, the heat
arriving per second relative to 0 °C, and it is exactly what the intuition above is
about. If the model cannot combine two raw series into it but can use it when handed it
ready-made, that is worth knowing. The deleted `t_mix` was a different animal — a
discharge-weighted *mixture* with coefficients nobody could justify.

Scored twice: pooled, and split by how much the river actually moved. A covariate can be
worth nothing on a plateau and everything across a cold snap, and the pooled mean, which
is mostly plateaus, would hide it. The swing classification is exp5's.

**Escalation, not grinding.** Every variant starts on 100 windows at a short context,
which costs under a minute, and only what is still *undecided* goes on to 250 windows and
then to a year of context. Undecided has a definition here rather than a feeling: an
interval that excludes zero has answered, and so has an interval that lies entirely
within ±1 %, because that is a tight null — the effect is measured and it is negligible.
Escalation is for intervals that are wide, meaning the sample is too small to tell, not
for intervals that are narrow and centred on nothing. Grinding a null down to another
decimal place buys nothing.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402
from checkpoint import bind, resume, save  # noqa: E402
from exp5_regimes import classify  # noqa: E402
from exp6_upstream import TARGET, pick_anchors  # noqa: E402
from exp8_catchment_weather import load_all  # noqa: E402
from exp12_pairs import CONFIRM_CONTEXT, SCREEN_CONTEXT, evaluate  # noqa: E402

logger = logging.getLogger(__name__)

WEATHER = ["airtemp", "t_catchment"]
T = "isar_toelz"
Q = "q_toelz_kw"
FLUX = "flux_toelz"

#: (windows, context) in the order they are tried. Cheap and wide first; a year of
#: context only for what the cheap stages could not settle.
LADDER = [(100, SCREEN_CONTEXT), (250, SCREEN_CONTEXT), (250, CONFIRM_CONTEXT)]

#: An interval lying entirely inside ±this is a settled null, not an open question.
#: A covariate whose true effect is under a percent does not deserve a variate slot,
#: so measuring it more precisely changes no decision.
NEGLIGIBLE_PCT = 1.0


def verdict(subset: pd.DataFrame, candidate: str, reference: str,
            metric: str) -> tuple[str, float, float, float]:
    """Decided how, and by how much — in percent of the reference's own score."""
    p = bench.paired(subset[subset.label.isin([candidate, reference])], reference,
                     metric=metric)
    row = p[p.label == candidate]
    if row.empty:
        return "fehlt", float("nan"), float("nan"), float("nan")
    r = row.iloc[0]
    base = bench.pool(subset[subset.label == reference])[metric].iloc[0]
    lo, hi = 100 * r["ci_lo"] / base, 100 * r["ci_hi"] / base
    if hi < 0:
        return "hilft", r["pct"], lo, hi
    if lo > 0:
        return "schadet", r["pct"], lo, hi
    if lo > -NEGLIGIBLE_PCT and hi < NEGLIGIBLE_PCT:
        return "sicher belanglos", r["pct"], lo, hi
    return "unentschieden", r["pct"], lo, hi


def sweep(fc, df, truth, anchors, variants, path, *, context: int) -> list[dict]:
    bind(path, df, anchors, context, [[lab, list(f), list(p)] for lab, f, p in variants])
    rows = resume(path, [label for label, _, _ in variants], anchors)
    done = {r["label"] for r in rows}
    for label, future, past_only in variants:
        if label in done:
            continue
        t0 = time.time()
        rows += evaluate(fc, df, anchors, truth, label=label, past_only=past_only,
                         context=context, future=future)
        save(path, rows)
        logger.info("ctx=%d %-28s %.0fs", context, label, time.time() - t0)
    return rows


def report(scores: pd.DataFrame, truth: pd.Series, title: str, pairs: list[tuple[str, str]]) -> None:
    print(f"\n=== {title} ===")
    print(bench.pool(scores)[["label", "n", "runs", "mae", "crps"]].to_string(index=False))

    print("\n--- was der Abfluss neben seiner eigenen Temperatur bringt ---")
    for metric in ("mae", "crps"):
        for candidate, reference in pairs:
            subset = scores[scores.label.isin([candidate, reference])]
            p = bench.paired(subset, reference, metric=metric)
            row = p[p.label == candidate]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"   {metric.upper():4s} {candidate:26s} vs {reference:22s} "
                  f"{r['pct']:+6.2f} %  CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                  f"besser in {r['better_in']*100:.0f} %  {r['verdict']}")

    classified = classify(scores, truth)
    counts = (classified[classified.label == classified.label.iloc[0]]
              .groupby("regime", observed=False).reference_time.nunique())
    print(f"\n--- nach Bewegung des Flusses ({counts.to_dict()}) ---")
    print(bench.pool(classified.dropna(subset=["regime"]), by=["label", "regime"]).pivot(
        index="label", columns="regime", values="mae").round(4).to_string())

    cut = classified.abs_swing.quantile(0.75)
    movers = classified[classified.abs_swing >= cut]
    print(f"\n--- nur das bewegteste Viertel ({movers.reference_time.nunique()} Fenster) ---")
    for candidate, reference in pairs:
        subset = movers[movers.label.isin([candidate, reference])]
        p = bench.paired(subset, reference, metric="mae")
        row = p[p.label == candidate]
        if not row.empty:
            r = row.iloc[0]
            print(f"   MAE  {candidate:26s} vs {reference:22s} {r['pct']:+6.2f} %  "
                  f"CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  {r['verdict']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = load_all()
    # Heat flux relative to 0 °C: m³/s x °C. A physical quantity, not a fitted weight.
    df[FLUX] = df[Q] * df[T]
    truth = df[TARGET]
    anchors = pick_anchors(df, [*WEATHER, T, Q, TARGET], n=250)
    if not anchors:
        raise RuntimeError("no anchors where Bad Tölz is complete enough")
    logger.info("%d Anker, %s .. %s", len(anchors), anchors[0], anchors[-1])
    logger.info("pro Jahr: %s",
                pd.Series(anchors).dt.year.value_counts().sort_index().to_dict())

    sub = df[[T, Q, TARGET]].dropna()
    print("\n=== wie hängen Temperatur und Abfluss in Bad Tölz zusammen? ===")
    print(f"   r(T, Q)          = {sub[T].corr(sub[Q]):+.4f}")
    print(f"   r(T, Eisbach)    = {sub[T].corr(sub[TARGET]):+.4f}")
    print(f"   r(Q, Eisbach)    = {sub[Q].corr(sub[TARGET]):+.4f}")
    print(f"   Abfluss: {sub[Q].min():.1f} .. {sub[Q].max():.1f} m³/s, "
          f"Median {sub[Q].median():.1f}")

    fc = bench.load_forecaster()

    bare = [
        ("nur Eisbach", [], []),
        ("T", [], [T]),
        ("Q", [], [Q]),
        ("T+Q", [], [T, Q]),
        ("T×Q (Wärmestrom)", [], [FLUX]),
        ("T+Q+T×Q", [], [T, Q, FLUX]),
    ]
    with_weather = [
        ("Wetter", WEATHER, []),
        ("Wetter + T", WEATHER, [T]),
        ("Wetter + Q", WEATHER, [Q]),
        ("Wetter + T+Q", WEATHER, [T, Q]),
        ("Wetter + T×Q", WEATHER, [FLUX]),
    ]
    variants = [*bare, *with_weather]

    #: (candidate, reference) — the comparisons that actually decide something.
    pairs = [
        ("T+Q", "T"),
        ("Q", "nur Eisbach"),
        ("T×Q (Wärmestrom)", "T"),
        ("T+Q+T×Q", "T+Q"),
        ("Wetter + T+Q", "Wetter + T"),
        ("Wetter + Q", "Wetter"),
        ("Wetter + T×Q", "Wetter + T"),
    ]

    open_pairs = list(pairs)
    for stage, (n_windows, context) in enumerate(LADDER, start=1):
        if not open_pairs:
            logger.info("alles entschieden — Stufe %d entfällt", stage)
            break
        # A nested subset, so a stage never changes which windows an earlier one used.
        step = max(1, len(anchors) // n_windows)
        subset_anchors = anchors[::step][:n_windows]
        needed = {lab for pair in open_pairs for lab in pair}
        stage_variants = [v for v in variants if v[0] in needed]
        logger.info("Stufe %d: %d Fenster, %d h Kontext, %d Varianten, %d offene Fragen",
                    stage, len(subset_anchors), context, len(stage_variants),
                    len(open_pairs))

        rows = sweep(fc, df, truth, subset_anchors, stage_variants,
                     bench.CACHE / f"exp18_stage{stage}.csv", context=context)
        scores = pd.DataFrame(rows)
        report(scores, truth,
               f"Stufe {stage}: {len(subset_anchors)} Fenster, {context} h Kontext",
               open_pairs)

        still_open = []
        print(f"\n--- Stufe {stage}: was ist entschieden? ---")
        for candidate, reference in open_pairs:
            calls = {m: verdict(scores, candidate, reference, m) for m in ("mae", "crps")}
            line = "  ".join(f"{m.upper()} {v[0]} ({v[1]:+.2f} % "
                             f"[{v[2]:+.2f},{v[3]:+.2f}])" for m, v in calls.items())
            print(f"   {candidate:20s} vs {reference:16s} {line}")
            if any(v[0] == "unentschieden" for v in calls.values()):
                still_open.append((candidate, reference))
        open_pairs = still_open

    if open_pairs:
        print("\nNach der letzten Stufe noch unentschieden: "
              + ", ".join(f"{c} vs {r}" for c, r in open_pairs))
        print("Das heißt: der Effekt ist zu klein, um ihn mit diesem Aufwand von null zu "
              "trennen, und zu groß, um ihn belanglos zu nennen. Mehr Fenster würden das "
              "ändern; ob die Antwort eine Entscheidung ändert, ist eine andere Frage.")
    else:
        print("\nAlles entschieden.")


if __name__ == "__main__":
    main()
