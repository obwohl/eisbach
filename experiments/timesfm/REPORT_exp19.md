# Finaler TimesFM-Satz gegen das Produktionsmodell DUET

14. September 2026. **Hauptvergleich: Orakel gegen Orakel**, wie gewünscht.
Keine erneute Feature-Selektion, kein Training, keine Parameteroptimierung.

Nachprüfung der auffälligen Produktionsbänder: [Exp20](REPORT_exp20.md) weist
einen 154,4-°C-Sensorausreißer als Ursache nach. Der erneut ausgeführte alte
DUET-Code liefert exakt dieselben Vergleichsquantile; auf 23 Starts vor dem
Ausreißer bleiben 32,7 % CRPS-Vorteil für TimesFM bestehen.

Der finale TimesFM-Satz gewinnt diesen kleinen Vier-Wochen-Vergleich:
**33,2 % weniger MAE, 32,3 % weniger RMSE und 33,8 % weniger CRPS** als DUET
mit ebenfalls gemessenem zukünftigem Wetter.

| Modell | MAE °C | RMSE °C | Bias °C | CRPS °C | Abdeckung 80-%-Band | Breite 80-%-Band °C |
|---|---:|---:|---:|---:|---:|---:|
| **DUET, Orakel** | 0,4534 | 0,5666 | +0,0638 | 0,2987 | 93,9 % | 2,0828 |
| **TimesFM final, Orakel** | **0,3028** | **0,3834** | −0,0532 | **0,1979** | **83,9 %** | **1,0689** |
| DUET, tatsächlich live archiviert | 0,4233 | 0,5349 | +0,0347 | 0,2837 | 94,7 % | 2,0657 |
| TimesFM nur Luft, Wetter-Replay | 0,3066 | 0,3952 | +0,0163 | 0,2014 | 82,9 % | 1,0525 |

Die beiden unteren Zeilen sind zusätzliche Referenzen, nicht der Hauptvergleich.
DUETs Orakel ist in diesem Zeitraum etwas schlechter als sein Live-Lauf:
perfektes Wetter garantiert bei einem festen trainierten Modell keinen besseren
Score; außerdem sind die historische Messreihe und deren Aufbereitung nicht
bitgenau das damalige Live-Eingabefenster. Die Live-Zeile bleibt deshalb sichtbar.
Der Vorteil von TimesFM hängt hier nicht daran, DUET absichtlich nur fehlerhafte
Wetterprognosen zu geben.

## Feste Modelle und Eingaben

- **TimesFM 3.0 / MLX:** 8760 Stunden Kontext, 96 Stunden Horizont, stündlich.
  Ziel `eisbach`; past-only `isar_toelz`, `loisach_beuerberg`;
  known-future `airtemp`, `t_catchment`, `rain_toelz`, `rain_lenggries`,
  `rain_kochel`, `rain_garmisch`, `solar_hohenpeissenberg`.
- **DUET Orakel:** unveränderte Produktionsgewichte, 384 Stunden Kontext,
  96 Stunden Horizont. Originalkanäle in Originalreihenfolge:
  `wassertemp`, `airtemp_96`, `pressure_96`. Münchner Lufttemperatur und Druck
  werden tatsächlich gemessen und wie in Produktion um 96 Stunden verschoben.
  Der aktuelle Produktionsadapter `assemble_long_frame` / `long_to_wide` /
  `forecast` wird unverändert verwendet, CPU-Inferenz. Kein Nachtrainieren.
- Beide Orakel verwenden dieselbe bereinigte historische Wasserreihe und
  identische ungefüllte Zielbeobachtungen. Zukunftswasser geht in kein Modell.
  DUET bekommt seine eigenen trainierten Kanäle; zusätzliche Fluss- und
  Wetterkanäle lassen sich ihm nicht ohne erneutes Training unterschieben.
- Der ergänzende TimesFM-Replay verwendet nur Münchner Luft als Wetterkovariate,
  mit einem vollständig archivierten Snapshot **vor** dem Prognosestart,
  höchstens 24 Stunden alt. Keine später geschriebenen Snapshots werden benutzt.
  Er ist nicht als Replay des vollständigen finalen Satzes beschriftet.

Das ist ein fairer Vergleich der beiden festen **Systemkonfigurationen bei
beobachtetem Wetter**, keine isolierte Untersuchung der Modellarchitektur:
Kontextlänge und Features unterscheiden sich bewusst entsprechend dem jeweiligen
festen Modell. Es beweist nicht, dass in jedem Zeitraum oder mit jedem echten
Wetterforecast dasselbe Ergebnis herauskommt.

## Umfang und Scoring

Angefragter Zeitraum: 17.08.–14.09.2026. Ein Lauf je Kalendertag, aus den
zulässigen archivierten Produktionsstarts derjenige am nächsten an 09:00 UTC;
Auswahl ohne Kenntnis der Prognosefehler. **24 vollständige 96-h-Fenster**,
Starts 17.08. 05:00 UTC bis 09.09. 08:00 UTC; Zielstunden bis 13.09. 08:00 UTC.
Die jüngsten Prognosen sind noch nicht über volle 96 Stunden beobachtet.

Von 70 archivierten Live-Starts im gewählten Datumsintervall sind 59 zulässig,
10 noch nicht vollständig beobachtet und einer wegen fehlender Ziel- oder
Orakelwetterwerte ausgeschlossen. Daraus entstehen 24 tägliche Starts.
Keine Zielwerte wurden zum Scoring interpoliert. Alle Modelle werden auf
denselben **2304 Prognose-Ziel-Paaren** bewertet; wegen überlappender Horizonte
sind dies **651 verschiedene Zielstunden**, keine 2304 unabhängigen Beobachtungen.
Es werden keine Signifikanztests oder unabhängigen Stichproben behauptet.

Punktschätzung = Median. MAE, RMSE und Bias über sämtliche 96 Stunden.
CRPS = bestehende gemeinsame Integration der Pinball-Loss über die Dezile
q0,1 bis q0,9. DUETs sieben nativen Quantile werden dafür linear auf die
TimesFM-Dezile interpoliert. Das ist derselbe **auf Dezile begrenzte CRPS**
wie in den bisherigen Experimenten, keine Integration über unbeobachtete Tails.
Die Grafiken zeigen Median, q0,1–q0,9 und interpoliertes q0,25–q0,75.
Auf beiden Modellpanels gelten dieselben Achsenskalen.

Die 80-%-Abdeckung liegt bei TimesFM näher am Nominalwert; seine Bänder sind
zugleich ungefähr halb so breit. Das ist hier eine erfreuliche Kombination
von Schärfe und Abdeckung, keine allgemeine Kalibrierungsgarantie für alle Jahreszeiten.

## Grafiken

Die lokal erzeugte Galerie liegt unter
`data/experiments/final_oracle_vs_duet/index.html`.

- `overview.png`: Tagesverlauf, je Start die ersten 24 Vorhersagestunden, und
  MAE-/CRPS-Kurven über alle Vorlaufbereiche. Etwaige Lücken bleiben sichtbar.
- `window_00.png` bis `window_23.png`: alle 24 Vier-Tage-Vorhersagen als
  Produktions-artige Quantilfächer, jeweils mit 48 Stunden Vorgeschichte,
  gemessenem Verlauf, MAE und CRPS. Im Menü frei auswählbar.
- Startansicht ist das **chronologisch letzte** vollständige Fenster,
  nicht das nachträglich beste Ergebnis.

## Kurze Robustheitskontrolle

Am 09.09. trat ein bekannter Sensorausreißer auf. Beide Orakel hier arbeiten
mit derselben bereinigten historischen Reihe; die ursprüngliche DUET-Live-Reihe
kann davon abweichen. Ohne den letzten, betroffenen Start bleiben 23 Fenster:
DUET Orakel MAE **0,4410**, CRPS **0,2919**; TimesFM final MAE **0,3000**,
CRPS **0,1963**. Der klare Unterschied bleibt bestehen. Mehr Backtesting wurde
für diese Anschauung nicht angeschlossen.

## Was ist entschieden, was bleibt offen?

**Die experimentelle TimesFM-Konfiguration steht.** Dieser Lauf bestätigt
zusätzlich den deutlichen Gesamtvorteil in dem betrachteten Zeitraum. Es gibt
keine weitere Feature- oder Kontextfrage, die vor dieser Demonstration entschieden
werden müsste.

Ein Produktionswechsel ist ein eigener Umsetzungsschritt: echte Wetterprognosen
für den finalen Satz beschaffen und archivieren, Datenlücken behandeln und das
Modell in den vorhandenen Produktionsablauf einbauen. Für den vollständigen Satz
existiert bisher kein historischer echter Wetter-Replay. Diese offenen
Betriebsfragen werden durch einen Orakelvergleich nicht erledigt; sie sind hier
auch nicht stillschweigend als erledigt ausgegeben. Produktion wurde nicht verändert.

## Provenienz und Reproduktion

Produktionsstand auf GitHub geprüft und read-only aus Git gelesen:
`722f467f196c2162fb5403b9a9e157ceeb5bf8d4` (`origin/main`). Die benutzten
Modelladapter und Gewichte stimmen mit diesem Stand überein.
DUET-Checkpoint-ID: `1c7a531768d8`, identisch in allen Live-Läufen und dem neu
berechneten Orakel. TimesFM-Revision:
`43046b85ec22d584a13f8098c2ed39c889e129c2`.

Runner: `exp19_final_vs_duet.py` erzeugt die 24 TimesFM-Orakel und 24 schlanken
Wetter-Replays samt Live-Referenz; `exp19_duet_oracle.py` ergänzt die
24 DUET-Orakel. Keine TimesFM-Inferenz wurde wegen des Wechsels auf
Orakel-gegen-Orakel wiederholt. Präsentationskorrekturen wurden aus gespeicherten
Quantilen gerendert. Rohquantile, Eingabefingerprints, Snapshot-Zeitstempel,
Ausschlussliste und gelesene Produktionsarchive liegen im ignorierten
`data/experiments/final_vs_duet/`; der Hauptvergleich liegt in
`data/experiments/final_oracle_vs_duet/`. Die ursprünglichen Manifest-Quellhashes
halten den Zustand zur Inferenz fest; spätere Darstellungsänderungen können
deshalb einen bewussten Provenienzfehler beim erneuten Inferenzstart auslösen,
statt alte Ergebnisse still zu verwenden.

```bash
TIMESFM_BACKEND=mlx HF_HUB_OFFLINE=1 .venv-exp11/bin/python experiments/timesfm/exp19_final_vs_duet.py
.venv-exp11/bin/python experiments/timesfm/exp19_duet_oracle.py
```

`eisbach/`, `main.py`, `tests/` und `data/archive/` wurden nicht verändert.
Keine generierten PNG/CSV werden committet. Die Ergebnisse sind lokale Artefakte;
die Produktion auf GitHub wurde nicht umgestellt.

Validierung: `pytest -q` **192 bestanden, 2 übersprungen**; `ruff check .`
**ohne Befund**. Übersicht und Vier-Tage-Grafik visuell geprüft; 24 Bilddateien,
Galerie-Verweise, Startauswahl und Auswahlhandler geprüft. Die automatische
Browsersteuerung war nicht verfügbar; die lokale Vorschau läuft auf
`http://127.0.0.1:8769/` und wurde zum Öffnen an die App übergeben.
