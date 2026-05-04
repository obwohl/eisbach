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
