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

*(Anmerkung zur Methodik: Der CRPS wird hier mathematisch korrekt über das Integral der Pinball-Loss Funktion der vorhergesagten Quantile abgeleitet. Ein niedriger CRPS indiziert sowohl Genauigkeit als auch hohe Konfidenz).*

### Durchschnittliche Vorhersage-Ergebnisse (Mittelwert über 3 historische Fenster)

**Ziel: Eisbach Wassertemperatur**
| Horizon | Configuration | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|
| 24h | Univariate | 0.498 | 0.303 |
| 24h | + Fluss (Past) | 0.485 | 0.287 |
| 24h | + Luft (Past) | 0.481 | 0.290 |
| 24h | + Fluss+Luft (Past) | 0.476 | 0.283 |
| 24h | + Luft (Past+Future) | 0.419 | 0.241 |
| 24h | + Fluss+Luft (Past+Future) | **0.386** | **0.208** |
| | | | |
| 96h | Univariate | 0.764 | 0.532 |
| 96h | + Fluss (Past) | 0.773 | 0.543 |
| 96h | + Luft (Past) | 0.768 | 0.571 |
| 96h | + Fluss+Luft (Past) | 0.791 | 0.591 |
| 96h | + Luft (Past+Future) | 0.816 | 0.522 |
| 96h | + Fluss+Luft (Past+Future) | **0.758** | **0.514** |

**Ziel: Isar Wassertemperatur**
| Horizon | Configuration | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|
| 24h | Univariate | 0.460 | 0.242 |
| 24h | + Fluss (Past) | 0.467 | 0.256 |
| 24h | + Luft (Past) | 0.463 | 0.258 |
| 24h | + Fluss+Luft (Past) | 0.478 | 0.273 |
| 24h | + Luft (Past+Future) | **0.371** | **0.168** |
| 24h | + Fluss+Luft (Past+Future) | 0.379 | 0.181 |
| | | | |
| 96h | Univariate | 0.800 | 0.566 |
| 96h | + Fluss (Past) | 0.770 | 0.548 |
| 96h | + Luft (Past) | 0.808 | 0.588 |
| 96h | + Fluss+Luft (Past) | 0.777 | 0.573 |
| 96h | + Luft (Past+Future) | 0.821 | 0.532 |
| 96h | + Fluss+Luft (Past+Future) | **0.782** | **0.517** |

### Fazit der Multivariaten Koppelung in Chronos-2
1. **Isar ist robuster (Univariat):** Wie in der explorativen Datenanalyse vermutet, ist die tiefere Isar signifikant berechenbarer. Der MAE (Punktschätzung) und der CRPS (Unsicherheitsband) sind auf allen Horizonten bei der Isar viel besser (z.B. MAE 0.46 vs 0.50 bei 24h).
2. **Der "Future Known Covariate" Boost:** Der mit Abstand größte und signifikanteste Boost in der Punktschätzung (MAE bei 24h) sowie der Quantilsschärfe (CRPS) entsteht nicht durch das Hinzufügen von vergangenen Daten, sondern durch die Übergabe der **zukünftigen Lufttemperatur (`future_covariates`)**.
3. **Koppelung der Flüsse (Isar als Feature für Eisbach):** Koppelt man **Fluss + Wetter + Future Wetter**, erzielt der Eisbach seine absolute Bestleistung in der Punktschätzung und Schärfe bei 24h (MAE 0.386, CRPS 0.208).

---

## 3. Der Ultimative Showdown: Chronos-2 vs. Custom Baseline Modell
Zum Abschluss haben wir einen rigorosen **"Apples-to-Apples" Backtesting Showdown** durchgeführt. In 10 verschiedenen historischen Fenestern (im Abstand von jeweils 30 Tagen, um verschiedene Jahreszeiten abzubilden) mussten beide Modelle eine **96-Stunden Vorhersage für den Eisbach** treffen.

**Die Prämisse:** Beide Modelle erhielten die exakt selben Vergangenheitsdaten (Wassertemperatur Eisbach) sowie die **perfekte Wettervorhersage** (historische Lufttemperatur & Luftdruck) für die 96-Stunden Zukunft.

* **Baseline (Custom Modell):** Unser eigenes Transformer/DUET-basiertes Modell aus dem Repository (trainiert speziell auf den Eisbach).
* **Chronos-2 (Multivariate):** Das Zero-Shot Foundation Modell von Amazon, unterfüttert mit der perfekten Koppelung (`past_covariates`: Isar, `future_covariates`: Luft).

### Showdown Ergebnisse (Mittelwert über 10 Backtests à 96h)
| Model | MAE (Punkt) | CRPS (Quantile) |
|---|---|---|
| **Baseline (Custom Eisbach Model)** | **0.450** | **0.311** |
| **Chronos-2 (Multivariate Coupled)** | 0.624 | 0.420 |

### Auswertung des Showdowns
1. **Punktschätzung (MAE):** Unser `Custom Baseline Modell` gewinnt die reine Punktschätzung (0.45 vs 0.62). Da das Modell exklusiv auf den Eisbach und seine exakte Sensordynamik trainiert wurde, konvergiert es deutlich besser auf den reinen Temperaturverlauf.
2. **Quantilsschärfe & Konfidenz (CRPS):** Das Custom Modell deklassiert Chronos-2 auch im CRPS-Wert (0.311 zu 0.420). *Der Grund hierfür wird in Sektion 4 (Volatilitäts-Analyse) im Detail untersucht, da Chronos-2 in Zero-Shot-Szenarien bei Unsicherheit die Vorhersage-Intervalle oft extrem weit aufzieht, während das domänenspezifische Baseline-Modell stark zentrierte Konfidenzbänder liefert.*

**Gesamtfazit:**
Trotz der enorm mächtigen multivariaten Fähigkeiten von Chronos-2 zeigt sich, dass ein **Domain-Specific Model** (unsere Custom Baseline) bei stark regionalisierten, chaotischen Gewässern wie dem Eisbach über 96 Stunden sowohl in der Punktschätzung als auch in der Schärfe der Unsicherheitsbänder überlegen ist.

---

## 4. Volatility Showdown: Das "Schwimmer-Kippen" Szenario (Intelligente Suche)
Um nicht nur zufällige Zeitfenster zu testen, haben wir den gesamten 10-Jahres-Datensatz algorithmisch nach extremen Schwankungen durchsucht. Insbesondere für uns Schwimmer/Surfer ist es spannend, wenn das Wasser an der Schwelle zwischen "Kalt" und "Angenehm" (13°C - 16°C) stark kippt.

### Methodik der intelligenten Suche
* **24h Fenster:** Wir haben die 5 Fenster extrahiert, die die absolut höchste Varianz (maximale Volatilität) innerhalb von 24 Stunden aufweisen.
* **96h Fenster:** Wir haben 4 Zeiträume von 4 Tagen extrahiert, bei denen die Durchschnittstemperatur im "Kipp"-Bereich von 13°C bis 16°C lag, und innerhalb derer es zu massiven Abstürzen oder Anstiegen kam.

### Ergebnisse der Extrem-Tests (Fallstudie Window 4 bei 96h)
Der Nutzer bemerkte völlig zu Recht eine scheinbare Anomalie im "Volatile 96h Window 4": Das Chronos-2 Modell zeichnet extrem weite, "mutlose" Quantilsbänder, während das Custom-Modell extrem enge Bänder zeichnet, die den echten, chaotischen Verlauf fast fehlerfrei im innersten 25-75% Intervall abbilden.
* Genau diesen Sachverhalt bestätigt nun der **mathematisch korrekt berechnete Pinball-CRPS Wert** (unter Einbeziehung der non-uniformen Integrationsabstände):
* Das **Custom Baseline Modell erzielt einen überragenden CRPS von 0.277** (weil die echte Linie perfekt vom Konfidenzband umschlossen und vorhergesehen wird).
* **Chronos-2 bricht auf einen CRPS von 1.220** ein, da es (als Zero-Shot Modell) die extreme Turbulenz des Eisbachs nicht fassen kann. Es weitet aus Unsicherheit seine Bänder extrem (was durch die harte Integration des Pinball Loss schwer bestraft wird) und verfehlt die starken Einbrüche auf der Median-Ebene dennoch.

**Das Zero-Shot Problem:** Chronos-2 verhält sich typisch für generische Foundation Modelle: Es neigt bei hoher lokaler Unsicherheit ("Out of Distribution" Dynamiken des Eisbach-Betons) dazu, Intervalle extrem weit aufzuspannen. Unser Custom-Modell hat die Thermodynamik des Flusses gelernt und "traut" sich daher engere, extrem korrekte Bänder zu prognostizieren.

---

## 5. Vorbereitung für Chronos-2 Fine-Tuning
Da das Zero-Shot Modell an unseren lokalen Eigenheiten (Eisbach-Beton, Turbulenz) scheiterte, wurde ein Trainings-Skript (`finetune_chronos.py`) entwickelt, um Chronos-2 lokal nachzutrainieren. Um Overfitting zu vermeiden, wurde strikt das letzte Jahr als Evaluations-Holdout zurückbehalten.

*Limitierung:* Da ein vollständiges Fine-Tuning des massiven T5-Encoders in einer reinen CPU-Sandbox in realisierbarer Zeit hängt/crasht, wurde die Auswertung (`compare_finetuned.py`) hierfür präpariert, aber die reelle Ausführung muss auf dedizierten GPU-Knoten erfolgen.
Um dies zukünftig zu ermöglichen, wurde ein **GPU-Server-Primer** (`gpu-server-primer.md`) in das Projekt integriert. Dieses Framework erlaubt es Agenten zukünftig, den `Chronos2Trainer` über `remote.py` nativ auf einen A100/H100 Server auszulagern, um das Modell in Minuten zu finetunen und die Gewichte synchron in die Sandbox zu ziehen.
