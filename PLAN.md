# Wie es weitergeht

Stand: TimesFM läuft live mit, DUET bleibt die Produktion. Das Archiv ist verschlankt.
Was hier steht, ist der Rest.

## Erledigt

**Das Archiv.** 174 MB → 110 MB. Die stündlichen Partitionen speichern nur noch, was sich
ändert: `station` und `unit` standen ohnehin in `SPECS` und wurden beim Lesen überschrieben,
eine leere Zelle heißt jetzt „Standardwert", und ein Zeitstempel ist `2013-07-01T00Z` statt
`2013-07-01 00:00:00+00:00`. Alle 2560 Partitionen umgeschrieben, alle 1.856.658 Zeilen
danach gegen die committeten Blobs geprüft: null Unterschiede. Ging nur, weil `hourly/`
aus `raw/` ableitbar ist — die vier anderen Stores sind es nicht.

**TimesFM in der Produktion.** `eisbach/timesfm.py`: Ziel `eisbach`, past-only
`isar_toelz` und `loisach_beuerberg`, known-future die sieben Wetterreihen, 8760 h
Kontext, 96 h Horizont. Dezil-Plots mit Median + 20–80 + 10–90. Fehlertolerant an drei
Stellen — optionale Abhängigkeit, `continue-on-error` im CI, `try` um den Lauf —, damit
ein kaputter Kandidat die dreimal tägliche Vorhersage nie mitnimmt.

**Der Schattenlauf** (der alte Punkt 4) fällt damit ab: der Kandidat schreibt seine
eigenen Vorhersagen nach `data/archive/timesfm/` und das Wetter, das er dafür bekam, nach
`data/archive/covariate_forecasts/`. In ein paar Monaten steht damit der ehrliche
Vergleich zur Verfügung, den keine Orakel-Messung ersetzen kann.

## 1. Den Schattenlauf auswerten — in ein paar Monaten

Nicht früher. Es braucht genug geschlossene 96-Stunden-Fenster, in denen beide Modelle
live liefen. Dann: dieselben Fenster, dasselbe Dezil-Gitter, gepaarter Block-Bootstrap.
`eisbach/verification.py` kennt den Kandidaten-Store noch nicht — das ist die Arbeit.

Erst diese Zahl entscheidet, ob TimesFM die Produktion wird. Die 32 % aus exp20 sind eine
Orakel-gegen-Orakel-Messung auf überlappenden Sommerfenstern und beweisen das nicht.

## 2. `rain_lenggries` nachprüfen

Stationsgebunden fehlen die 575 Stunden, die die Koordinatenabfrage aus Kreuth-Glashütte
nachgeliefert hat. Ehrlichere Lücken, aber **eine andere Reihe** als die, auf der exp15
die Regen-Auswahl gemessen hat. Einmal nachmessen.

## 3. Die Live-Payloads beschneiden

Der eigentliche Wachstumstreiber, und größer als alles, was Punkt „Archiv" eingespart hat:
`raw/<station>/live/` legt bei jedem Lauf für jede der neun Stationen die volle
72-Stunden-Antwort ab. **~50 KB pro Lauf, ~55 MB und ~10.000 Dateien pro Jahr**, bei etwa
90 % Überlappung — jede Stunde wird rund neunmal abgelegt.

Diese Payloads sind aber die **unersetzliche** Hälfte: `hourly/` wird aus ihnen
rekonstruiert. Sie zu beschneiden heißt zu entscheiden, wann eine Live-Antwort durch einen
späteren Jahres-Backfill abgelöst ist. Das ist Design-Arbeit, keine Aufräumarbeit.

## Nicht jetzt

Konforme Kalibrierung (ganz zuletzt), Ensemble über Kontextlängen, Chronos-2.

## Der Kovariablensatz, zum Nachschlagen

| | Reihe | Station |
|---|---|---|
| Ziel | `eisbach` | GKD München-Himmelreichbrücke |
| past-only | `isar_toelz` | GKD Bad Tölz B472 |
| past-only | `loisach_beuerberg` | GKD Beuerberg |
| known-future | `airtemp` | DWD 03379 München-Stadt |
| known-future | `t_catchment` | DWD 01550 Garmisch-Partenkirchen |
| known-future | `rain_toelz` | DWD 05262 Waakirchen-Demmelberg |
| known-future | `rain_lenggries` | DWD 06257 Jachenau-Tannern |
| known-future | `rain_kochel` | DWD 06342 Schlehdorf |
| known-future | `rain_garmisch` | DWD 01550 Garmisch-Partenkirchen |
| known-future | `solar_hohenpeissenberg` | DWD 02290 Hohenpeißenberg |

Jeder Abfluss ist draußen: er trägt Information (ohne Wetter −3,0 % MAE, bei starken
Bewegungen −6,7 %), aber die Lufttemperatur weiß sie schon. Belege in
`experiments/timesfm/STATUS.md`.
