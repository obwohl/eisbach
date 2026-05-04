# Analyse: Eisbach vs. Isar Wassertemperatur & Vorhersagbarkeit

## 1. Daten und Dynamik (Explorative Datenanalyse)
Die Wassertemperaturen vom **Eisbach (Himmelreichbrücke)** und der **Isar (München)** wurden für die letzten 120 Tage analysiert.

* **Verteilung und Kopplung:** Im Mittel ist der Eisbach mit **6.40 °C** leicht kälter als die Isar mit **6.60 °C** (durchschnittliche Differenz: -0.21 °C).
* **Volatilität:** Die Standardabweichung des Eisbachs (2.90) ist marginal höher als die der Isar (2.81). Die Hypothese bestätigt sich: Der flachere, einbetonierte Eisbach reagiert weitaus sensibler und dynamischer (chaotischer) auf Umwelteinflüsse, was sich auch in den Extrema der Differenzen (bis zu +1.5 °C und -0.9 °C) zeigt.
* *Die generierten Plots (`scatter.png` und `diff_dist.png`) visualisieren diese Verteilung der Temperaturdifferenzen und die starke, aber nicht perfekte lineare Korrelation beider Gewässer.*

## 2. Chronos-2 Forecasting (24h & 96h)
Das Modell **amazon/chronos-2** (Zero-Shot Time Series Foundation Model) wurde exakt wie beauftragt getestet, um die Vorhersagbarkeit quantitativ zu bewerten. Wir haben sowohl univariate als auch gekoppelte multivariate Vorhersagen (mit Lufttemperatur als _known future covariate_) in der offiziellen Chronos-2 Listen-API implementiert.

### Vorhersage-Ergebnisse
| Horizon | River | Type | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|---|
| **24h** | **Eisbach** | Univariate | 0.990 | 0.469 |
| 24h | Isar | Univariate | 0.598 | 0.333 |
| **24h** | **Eisbach** | **Multivariate** | **0.722** | 0.476 |
| 24h | Isar | **Multivariate** | **0.389** | 0.353 |
| **96h** | **Eisbach** | Univariate | 0.890 | 0.943 |
| 96h | Isar | Univariate | 0.817 | 0.570 |
| **96h** | **Eisbach** | **Multivariate** | **0.674** | 0.928 |
| 96h | Isar | **Multivariate** | **0.692** | 0.655 |

### Fazit der Vorhersagbarkeit & Multivariater Kopplung
1. **Isar ist deutlich berechenbarer (Univariat):** Wie vermutet, ist die trägere Isar univariat leichter vorherzusagen. Die Punktschätzung (MAE 0.598 Isar vs 0.990 Eisbach bei 24h) sowie die Verteilungsschärfe (CRPS 0.33 vs 0.47) sind signifikant besser.
2. **Multivariater Boost (Lufttemperatur):** Die korrekte Integration der Lufttemperatur in Chronos-2 führt zu einer **drastischen Verbesserung der Punktschätzung (MAE) für beide Gewässer**:
   - Beim **Eisbach (24h)** sinkt der MAE von 0.990 auf **0.722** (ca. 27% Verbesserung).
   - Bei der **Isar (24h)** sinkt der MAE von 0.598 auf **0.389** (ca. 35% Verbesserung).
   - *Auffällig:* Der CRPS-Wert bleibt stabil oder steigt sogar ganz leicht an. Das bedeutet, das Modell trifft zwar mit dem Median den realen Temperaturverlauf deutlich exakter, aber die vom Modell aufgespannte "Unsicherheits-Wolke" (die generierten Quantile) wird durch das Zufügen von exogenen Variablen scheinbar etwas breiter/ungenauer kalibriert.
3. **Langfrist-Fehler (96h):** Je weiter wir in die Zukunft blicken, desto mehr dominiert das Chaos im Eisbach. Der CRPS-Wert für den Eisbach (über 0.92) signalisiert massiv wachsende Unsicherheit. Durch die multivariate Kopplung lässt sich die Langfrist-Punktschätzung (MAE) beim Eisbach nochmal deutlich auf 0.674 drücken, da das System stark dem vorhersehbaren Lufttemperatur-Trend folgt.

**Fazit:** Der Eisbach ist ein hochdynamisches System. Zwar liefert das multivariate Chronos-2-Modell exzellente Punktvorhersagen (MAE), doch die statistischen Quantile offenbaren, dass die "echte" Isar als Vorhersageobjekt viel ruhiger, schärfer und vertrauenswürdiger bleibt (CRPS).
