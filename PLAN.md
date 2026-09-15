# Wie es weitergeht

Stand nach PR #42. Die Produktion ist repariert — der erste Lauf nach dem Merge
(14.09. 22:00) liefert wieder Median 17,30 °C bei 3,84 °C Bandbreite, gegen 14,19 °C
bei 24,07 °C im Lauf davor. Das Gate wirkt.

## 1. TimesFM in die Produktion — die große Sache

Alles andere hängt daran. Der Umschalter auf der Seite wartet, das Archiv liefert die
Historie, der Workflow kopiert die Dateien schon. Es fehlt:

- ein Inferenzmodul für TimesFM 3.0 mit dem gesetzten Satz: Ziel `eisbach`, past-only
  `isar_toelz` und `loisach_beuerberg`, known-future `airtemp`, `t_catchment`, die vier
  Regenreihen und `solar_hohenpeissenberg`; 8760 h Kontext, 96 h Horizont;
- Plots und CSV mit **Median + 20–80 + 10–90** — TimesFM liefert nur Dezile, 25/75 und
  5/95 gibt es nicht;
- die Abhängigkeit im CI: 1,3 GB Checkpoint in einem Job, der heute mit 10 MB auskommt.

**Fehlertolerant einhängen.** Schlägt der TimesFM-Teil fehl, läuft DUET wie bisher durch
und die Seite zeigt ein Modell. Die dreimal tägliche Vorhersage darf an einem Kandidaten
nicht sterben.

## 2. Das Archiv verschlanken — bevor es wächst

151 MB in 2560 Dateien, und `git add data/archive` committet den laufenden Monat bei
jedem Lauf neu. Je Stunde elf Spalten, davon sind `samples`, `duplicate_count`,
`conflict`, `station` und `unit` über die ganze Datei konstant. 20 KB wo 10 KB reichen.

Das ist eine Stunde Arbeit jetzt und ein Ärgernis in einem Jahr.

## 3. `rain_lenggries` nachprüfen

Stationsgebunden fehlen die 575 Stunden, die die Koordinatenabfrage aus
Kreuth-Glashütte nachgeliefert hat. Ehrlichere Lücken, aber **eine andere Reihe** als
die, auf der exp15 die Regen-Auswahl gemessen hat. Einmal nachmessen.

## 4. Schattenlauf statt Orakel

Jede Kovariablen-Zahl im Forschungsbaum benutzt das Wetter, das eintrat. Gemessen kostet
das ~4,8 % des Vorteils für die Münchner Luft; für Süd-Luft, Regen und Strahlung gibt es
gar keine archivierte Prognose, und für die Vergangenheit wird es nie eine geben.

Der billige Weg ist der, den du selbst vorgeschlagen hast: TimesFM live mitlaufen lassen
und **seine eigenen Vorhersagen archivieren**. Nach ein paar Monaten ist die Bewertung
ehrlich, ohne dass irgendeine Wetterprognose aufgehoben werden muss. Das fällt als
Nebenprodukt von Punkt 1 ab.

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
