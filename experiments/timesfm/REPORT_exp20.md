# DUET-Ausreißer und Vergleich mit dem alten Modell

14. September 2026. Die riesigen Bänder auf der Deployment-Seite sind reproduzierbar.
Die Ursache ist eine historische Wasserbeobachtung von **154,4 °C am 09.09.2026
um 08:00 UTC**. Der TimesFM-Vorteil aus Exp19 bleibt beim Vergleich mit dem
alten DUET vor diesem Messfehler bestehen.

## Kausaler Kontrolllauf

Produktionsarchiv: `722f467f196c2162fb5403b9a9e157ceeb5bf8d4`.
Prognosestart: 14.09.2026, 17:00 UTC; 384 Stunden Kontext, 96 Stunden Horizont.
Gewichte, Wetter und Prognosestart sind in beiden Läufen identisch. Nur die
eine Wasserbeobachtung wird im zweiten Lauf entfernt und zwischen ihren
Nachbarn zu 19,1 °C interpoliert. Keine Archivdatei wurde geändert.

| Eingabe | Median nach 96 h | Mittlere Breite q01–q99 |
|---|---:|---:|
| Mit 154,4-°C-Ausreißer | 8,2813 °C | 24,0676 °C |
| Nur diese Stunde repariert | 17,0390 °C | 3,1756 °C |

Der erste Lauf reproduziert alle archivierten Produktionsquantile bis auf
maximal **0,00000763 °C**. Das erklärt die gezeigte Fehlprognose direkt.
Die RMS-Abweichung der Wasserhistorie vom Median steigt durch den Ausreißer
von 1,148 auf 6,999 °C und verfälscht die Eingabeskalierung.

Der Modellcode unter `eisbach/model/` und `main.py` ist zwischen dem alten
Stand `7024d7ff5ee2` und dem untersuchten Produktionsstand unverändert.
Auch die Gewichte sind dieselben (SHA256
`1c7a531768d883af0c70aea1d7fe62fe59638000bf70097d61fb90f2bc4309b0`).
Der vorhandene Schutz gegen unplausible Pegelmessungen liegt im Forschungszweig
(Commit `7513df8ac011d36c1be6289b979bf5a26ed34ac8`), fehlt aber in diesem
Produktionsstand. Ein zeitlicher Zusammenhang mit der Umstellung belegt daher
keine Verschlechterung des Modells selbst.

## Tatsächlicher Lauf des alten DUET

`exp20_old_duet_comparison.py` lädt Datenadapter und Modelladapter direkt aus
`7024d7ff5ee2` und führt DUET erneut aus. Verwendet werden ausschließlich die
**23 Starts vor dem Ausreißer**, 17.08.–08.09.2026, jeweils 96 Stunden.
Alle 2208 Prognose-Ziel-Paare werden auf denselben Zielwerten bewertet.
Die Horizonte überlappen; dies sind keine 23 unabhängigen Stichproben.

Beide Hauptmodelle erhalten gemessenes zukünftiges Wetter entsprechend ihren
festen Eingabekanälen. TimesFM bleibt bei 8760 Stunden und dem finalen Featuresatz
aus Exp19. Seine unveränderten gespeicherten Quantile werden wiederverwendet.
Der alte DUET-Lauf reproduziert sämtliche DUET-Orakelquantile des bisherigen
Vergleichs **exakt: maximale Differenz 0,0 °C**.

| Metrik | DUET vor Umstellung, Orakel | TimesFM final, Orakel |
|---|---:|---:|
| MAE | 0,4410 °C | **0,3000 °C** |
| RMSE | 0,5536 °C | **0,3813 °C** |
| CRPS, gemeinsame Dezilnäherung | 0,2919 °C | **0,1963 °C** |
| Abdeckung 80-%-Band | 94,4 % | 84,2 % |
| Mittlere Breite 80-%-Band | 2,0751 °C | 1,0633 °C |

TimesFM erreicht damit **32,0 % weniger MAE, 31,1 % weniger RMSE und 32,7 %
weniger CRPS** gegen das alte, von diesem Ausreißer unbeeinflusste DUET.
Die damals tatsächlich live archivierten DUET-Läufe erreichen zusätzlich
MAE 0,4213 °C und CRPS 0,2824 °C. Diese Referenz bleibt in der Galerie sichtbar.

Die explodierten Produktionsläufe ab 11.09. waren schon in Exp19 nicht enthalten:
Sie hatten noch keinen vollständig beobachteten 96-h-Horizont. Diese Kontrolle
entfernt zusätzlich den letzten Start am 09.09. Der Vorteil ist somit kein
Artefakt der aktuell riesigen Bänder. Er gilt für diesen Zeitraum und den
Orakelvergleich; ein universell bestes Modell oder ein entsprechender Vorteil
mit echten Wetterprognosen ist damit nicht bewiesen.

## Artefakte und Reproduktion

- `data/experiments/duet_regression/diagnosis.png`: direkter Ursachentest,
  identische Achsen, native 98-/90-/50-%-Bänder.
- `data/experiments/old_duet_vs_timesfm/index.html`: alle 23 Vergleichsfenster,
  Übersicht und Scores; gemeinsame 80-/50-%-Bänder wie in Exp19.
- JSON-Manifeste, Quantile und Scores liegen daneben. Generierte Dateien werden
  nicht committed. Die vorbereiteten Eingabecaches aus Exp17/19 sind erforderlich.

```sh
MPLCONFIGDIR=/private/tmp/eisbach-mpl .venv-exp11/bin/python experiments/timesfm/exp20_duet_regression.py
MPLCONFIGDIR=/private/tmp/eisbach-mpl .venv-exp11/bin/python experiments/timesfm/exp20_old_duet_comparison.py
```

Dieser Audit ändert weder Produktionscode noch Archive und deployt nichts.
