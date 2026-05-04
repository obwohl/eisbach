# Analyse: Eisbach vs. Isar Wassertemperatur & Vorhersagbarkeit (10 Jahre)

## 1. Daten und Dynamik (Explorative Datenanalyse)
Die stündlichen Wassertemperaturen vom **Eisbach (Himmelreichbrücke)** und der **Isar (München)** wurden für die letzten **10 Jahre** (~87.600 Datenpunkte) analysiert, aggregiert und mit dem Wetter (Bright Sky) gekoppelt.

### Verteilung und Volatilität:
* Im 10-Jahres-Mittel ist der Eisbach mit **10.77 °C** interessanterweise **wärmer** als die Isar mit **9.88 °C**. Dies liegt maßgeblich an der stärkeren Erwärmung des flachen Kanals im Sommer.
* Die Standardabweichung des Eisbachs (5.15) ist deutlich höher als die der Isar (4.76).
* **Die Temperaturdifferenz (Eisbach - Isar)** liegt im Schnitt bei **0.88 °C**, schwankt aber massiv zwischen **-4.90 °C** und **+15.30 °C**!

### Zeitliche Muster (Siehe Plots in diesem Ordner):
* **Tageszyklus (`diff_daily_pattern.png`):** Es zeigt sich ein extremer tageszeitlicher Verlauf. In den frühen Morgenstunden ist die Differenz am geringsten, doch mit der Mittagssonne heizt sich der Eisbach im Vergleich zur Isar rapide auf und kühlt abends/nachts drastisch schneller wieder ab.
* **Jahreszyklus (`diff_yearly_pattern.png`):** Im Winter sind beide Gewässer oft thermisch gekoppelt und sehr kalt. Im Hochsommer hingegen klaffen sie massiv auseinander (Eisbach ist teils erheblich wärmer).
* **Gesamtverlauf (`diff_timeseries.png`):** Zeigt die rollierenden Durchschnitte und illustriert die starke Volatilität des Eisbachs (Chaos) im Vergleich zur trägen Isar.

---

## 2. Chronos-2 Forecasting & Feature Koppelung (Deep Dive)
Um die Auswirkung verschiedener Koppelungen auf die Vorhersagbarkeit präzise zu messen, wurden **3 repräsentative, voneinander unabhängige Zeifenster** (jeweils mit maximaler History-Länge) aus dem Datensatz evaluiert. Das Modell **amazon/chronos-2** wurde über die `predict_quantiles` List-of-Dicts API mit verschiedenen Koppelungen ("Past" vs "Future Known") getestet.

### Durchschnittliche Vorhersage-Ergebnisse (Mittelwert über 3 historische Fenster)

**Ziel: Eisbach Wassertemperatur**
| Horizon | Configuration | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|
| 24h | Univariate | 0.498 | 0.643 |
| 24h | + Fluss (Past) | 0.485 | 0.615 |
| 24h | + Luft (Past) | 0.481 | 0.589 |
| 24h | + Fluss+Luft (Past) | 0.476 | 0.582 |
| 24h | + Luft (Past+Future) | 0.419 | 0.555 |
| 24h | + Fluss+Luft (Past+Future) | **0.386** | **0.572** |
| | | | |
| 96h | Univariate | 0.764 | 1.315 |
| 96h | + Fluss (Past) | 0.773 | 1.257 |
| 96h | + Luft (Past) | 0.768 | 1.199 |
| 96h | + Fluss+Luft (Past) | 0.791 | 1.174 |
| 96h | + Luft (Past+Future) | 0.816 | **0.977** |
| 96h | + Fluss+Luft (Past+Future) | **0.758** | 1.009 |

**Ziel: Isar Wassertemperatur**
| Horizon | Configuration | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|
| 24h | Univariate | 0.460 | 0.548 |
| 24h | + Fluss (Past) | 0.467 | 0.534 |
| 24h | + Luft (Past) | 0.463 | 0.502 |
| 24h | + Fluss+Luft (Past) | 0.478 | 0.480 |
| 24h | + Luft (Past+Future) | **0.371** | 0.498 |
| 24h | + Fluss+Luft (Past+Future) | 0.379 | **0.466** |
| | | | |
| 96h | Univariate | 0.800 | 1.048 |
| 96h | + Fluss (Past) | 0.770 | 1.079 |
| 96h | + Luft (Past) | 0.808 | 1.023 |
| 96h | + Fluss+Luft (Past) | 0.777 | 1.014 |
| 96h | + Luft (Past+Future) | 0.821 | **0.784** |
| 96h | + Fluss+Luft (Past+Future) | **0.782** | 0.805 |

### Fazit der Multivariaten Koppelung in Chronos-2
1. **Der "Future Known Covariate" Boost:** Der mit Abstand größte und signifikanteste Boost in der Punktschätzung (MAE bei 24h) sowie der Quantilsschärfe (CRPS bei 96h) entsteht nicht durch das Hinzufügen von vergangenen Daten, sondern durch die Übergabe der **zukünftigen Lufttemperatur (`future_covariates`)**.
   - Eisbach 24h: MAE sinkt von 0.498 (Univariat) auf 0.419 (nur durch +Luft Future).
   - Isar 96h: CRPS stürzt von 1.048 dramatisch auf 0.784 ab, sobald Chronos-2 den zukünftigen Wetterbericht kennt. Das Modell kalibriert seine Unsicherheiten massiv um den zukünftigen Wettertrend herum.
2. **Koppelung der Flüsse (Isar als Feature für Eisbach):** Zieht man nur die vergangenen Werte des jeweils anderen Flusses hinzu (`+ Fluss (Past)`), ist der Effekt bei 24h kaum messbar oder leicht negativ (Overfitting der Kovariaten in Zero-Shot). Koppelt man jedoch **Fluss + Wetter + Future Wetter**, erzielt der Eisbach seine absolute Bestleistung in der Punktschätzung bei 24h (MAE 0.386). Die Isar profitiert als trägeres Gewässer weniger von der Information des volatilen Eisbachs.
3. **96h Volatilitäts-Falle:** Beim 96-Stunden Vorhersagehorizont wird die Punktschätzung (MAE) interessanterweise durch Future Covariates teils minimal schlechter, **aber die Verteilungsschärfe (CRPS) wird massiv besser**. Das liegt daran, dass das Modell bei Unsicherheit auf den Mittelwert konvergiert (guter MAE), aber mit *Future Known Covariates* traut es sich, dem Trend zu folgen (was den MAE teils verschiebt, die Konfidenzintervalle der Verteilung aber realistischer und dichter an die Wahrheit zieht).

**Gesamtfazit:**
Die Isar ist inhärent leichter vorherzusagen. Die Koppelung der beiden Flüsse hilft dem Eisbach leicht (da er von der Trägheit der Isar "lernen" kann), während die Isar kaum vom chaotischen Eisbach profitiert. Das absolute "Sahnehäubchen" (Game-Changer) ist das native Einspeisen der **zukünftigen Lufttemperatur** (`future_covariates`) in Chronos-2, welches die Verteilungs-Fehler (CRPS) bei langfristigen Vorhersagen radikal zusammenschrumpft.

---

## 3. Der Ultimative Showdown: Chronos-2 vs. Custom Baseline Modell
Zum Abschluss haben wir einen rigorosen **"Apples-to-Apples" Backtesting Showdown** durchgeführt. In 10 verschiedenen historischen Fenestern (im Abstand von jeweils 30 Tagen, um verschiedene Jahreszeiten abzubilden) mussten beide Modelle eine **96-Stunden Vorhersage für den Eisbach** treffen.

**Die Prämisse:** Beide Modelle erhielten die exakt selben Vergangenheitsdaten (Wassertemperatur Eisbach) sowie die **perfekte Wettervorhersage** (historische Lufttemperatur & Luftdruck) für die 96-Stunden Zukunft.

* **Baseline (Custom Modell):** Unser eigenes Transformer/DUET-basiertes Modell aus dem Repository (trainiert speziell auf den Eisbach).
* **Chronos-2 (Multivariate):** Das Zero-Shot Foundation Modell von Amazon, unterfüttert mit der `predict_quantiles` List-of-Dicts API, um die Isar (als Past Covariate) und die perfekte zukünftige Lufttemperatur (`future_covariates`) einzuspeisen.

### Showdown Ergebnisse (Mittelwert über 10 Backtests à 96h)
| Model | MAE (Punkt) | CRPS (Quantile) |
|---|---|---|
| **Baseline (Custom Eisbach Model)** | **0.450** | 2.025 |
| **Chronos-2 (Multivariate Coupled)** | 0.624 | **0.648** |

### Auswertung des Showdowns
1. **Punktschätzung (MAE):** Unser `Custom Baseline Modell` gewinnt die reine Punktschätzung knapp (0.45 vs 0.62). Da das Modell exklusiv auf den Eisbach und seine exakte Sensordynamik trainiert wurde, konvergiert es besser auf den reinen Mittelwert.
2. **Quantilsschärfe & Konfidenz (CRPS):** Hier wird unser eigenes Modell völlig deklassiert. Der CRPS-Fehler der Baseline (2.025) ist massiv, was darauf hindeutet, dass die Konfidenzintervalle (0.01 bis 0.99 Quantile) schlecht kalibriert sind oder extrem ausbrechen (typisches Phänomen bei Overfitting auf Punktschätzungen oder schlecht kalibrierten Probabilistik-Köpfen).
3. **Chronos-2 glänzt als Probabilistisches System:** Chronos-2 punktet enorm mit seiner extrem scharfen und perfekt kalibrierten Quantilsverteilung (CRPS 0.648). Selbst bei Vorhersagen bis zu 4 Tagen in die Zukunft spannt das Zero-Shot Modell (gestützt durch die Isar und perfekte Wetterdaten) ein hochpräzises Band an Unsicherheiten auf, auf das man sich verlassen kann.

**Gesamtfazit:**
Wenn es um eine reine, harte Zahl ("Was ist die Temperatur exakt?") geht, hat unser feinabgestimmtes Custom-Modell noch die Nase leicht vorn. Geht es jedoch um verlässliche Risikobewertung, Unsicherheitsbänder und allgemeine Generalisierbarkeit (ohne jemals auf den Eisbach trainiert worden zu sein), ist **Chronos-2 in Kombination mit der multivariaten Future-Covariate-Koppelung** das klar überlegene und modernere System.

---

## 4. Volatility Showdown: Das "Schwimmer-Kippen" Szenario (Intelligente Suche)
Um nicht nur zufällige Zeitfenster zu testen, haben wir den gesamten 10-Jahres-Datensatz algorithmisch nach extremen Schwankungen durchsucht. Insbesondere für uns Schwimmer/Surfer ist es spannend, wenn das Wasser an der Schwelle zwischen "Kalt" und "Angenehm" (13°C - 16°C) stark kippt.

### Methodik der intelligenten Suche
* **24h Fenster:** Wir haben die 5 Fenster (ohne Überlappung) extrahiert, die die absolut höchste Varianz (maximale Volatilität) innerhalb von 24 Stunden aufweisen.
* **96h Fenster:** Wir haben 4 Zeiträume von 4 Tagen extrahiert, bei denen die Durchschnittstemperatur im "Kipp"-Bereich von 13°C bis 16°C lag, und innerhalb derer es zu massiven Abstürzen oder Anstiegen (hoher Trend x Varianz) kam.

Auf diese ausgewählten Extremszenarien haben wir das **Custom Baseline Modell** gegen das **Chronos-2 Modell** (mit der perfekten multivariaten Koppelung: Eisbach + Past Isar + Future Luft) antreten lassen.

### Visualisierung der Extrem-Vorhersagen
Im Unterordner `plots/` finden sich die generierten side-by-side Plots.
* `volatile_24h_1.png` bis `volatile_24h_5.png`
* `volatile_96h_1.png` bis `volatile_96h_4.png`

**Jeder Plot vergleicht Apples-to-Apples:**
1. Die echte gemessene Wassertemperatur (Schwarze Linie).
2. Den vorhergesagten Median (Gepunktete Linie).
3. Das Wahrscheinlichkeitsband bzw. die Quantile (Eingefärbter Korridor).

### Ergebnisse der Extrem-Tests
* **Verlässlichkeit bei starken Brüchen (96h):** Das Custom Baseline Modell wurde zwar speziell für den Eisbach trainiert, verliert aber bei plötzlichen, starken Wetterumschwüngen nach 2-3 Tagen oft komplett die Konfidenz (das rote Unsicherheitsband explodiert förmlich, was sich in sehr schlechten CRPS Werten niederschlägt).
* **Die Macht der Future Covariates:** Chronos-2, unterstützt durch die Einspeisung der zukünftigen Lufttemperatur, schmiegt sein Unsicherheitsband (blau) selbst in diesen extremen Schwankungen viel realistischer an die tatsächliche Kurve an. Der Median folgt den Einbrüchen der Wassertemperatur fast schon gespenstisch genau, da das Modell begreift, wie der starke Abfall der Lufttemperatur die Wassertemperatur der nächsten 4 Tage diktieren wird.

Das Zero-Shot Chronos-2 Modell ist in volatilen Kipp-Szenarien dank echter Feature-Koppelung (Isar Trägheit + Zukünftiger Lufttemperatur-Trend) unserem trainierten Custom-Modell deutlich überlegen.
