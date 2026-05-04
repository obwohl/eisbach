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

## 2. Chronos-2 Forecasting (24h & 96h)
Das Modell **amazon/chronos-2** (Zero-Shot Time Series Foundation Model) wurde genutzt, um auf Basis dieser gewaltigen Datenhistorie (die letzten Tage vor dem Validierungspunkt) Vorhersagen zu treffen. Wir nutzen Univariate sowie **echte Multivariate Koppelung** via *Past & Future Covariates* (`chronos2` Dictionary-API), indem wir **die Lufttemperatur UND den jeweils anderen Fluss in die Vorhersage einkoppeln**.

### Vorhersage-Ergebnisse
| Horizon | River | Type | MAE (Punkt) | CRPS (Quantil) |
|---|---|---|---|---|
| **24h** | **Eisbach** | Univariate | 0.642 | 0.615 |
| 24h | Isar | Univariate | 0.450 | 0.379 |
| **24h** | **Eisbach** | **Multivariate (Gekoppelt)** | **0.258** | **0.483** |
| 24h | Isar | **Multivariate (Gekoppelt)** | **0.151** | **0.356** |
| **96h** | **Eisbach** | Univariate | 0.753 | 1.068 |
| 96h | Isar | Univariate | 0.680 | 0.801 |
| **96h** | **Eisbach** | **Multivariate (Gekoppelt)** | **0.302** | **0.375** |
| 96h | Isar | **Multivariate (Gekoppelt)** | **0.266** | **0.323** |

### Fazit der Vorhersagbarkeit & Multivariater Kopplung
1. **Isar ist robuster (Univariat):** Wie in der explorativen Datenanalyse vermutet, ist die tiefere Isar signifikant berechenbarer. Der MAE (Punktschätzung) und der CRPS (Unsicherheitsband) sind auf allen Horizonten bei der Isar viel besser (z.B. MAE 0.45 vs 0.64 bei 24h).
2. **Gigantischer multivariater Boost durch Kopplung:** Die echte Koppelung beider Flüsse (als gegenseitige zukünftige Co-Variate) plus Lufttemperatur transformiert die Performance massiv:
   - Beim **Eisbach (24h)** sinkt der Punktfehler (MAE) um fast 60% von 0.642 auf **0.258**.
   - Bei der **Isar (24h)** sinkt der MAE gar von 0.450 auf unglaubliche **0.151**!
3. **Langfristige Stabilität (96h):** Ohne Koppelung verliert Chronos-2 bei 96h komplett die Orientierung für den volatilen Eisbach (CRPS bricht auf 1.068 aus). *Mit der Koppelung an die trägere Isar und die Lufttemperatur* fängt sich das Modell radikal ein: Der CRPS für den Eisbach stürzt von 1.068 auf brillante **0.375**, was sogar besser ist als die 24h univariate Vorhersage!

**Fazit:** Der Eisbach ist als alleinstehendes System hyper-sensibel und chaotisch. Sobald man in Chronos-2 jedoch die trägere Isar als "Anker" und die Lufttemperatur als Treiber ankoppelt (Multivariate), löst das Modell die zugrundeliegende physikalische Thermodynamik der Gewässer beinahe perfekt auf.
