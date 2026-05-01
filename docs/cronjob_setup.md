# Einrichtung von Cron-Jobs zur Vorhersage-Generierung

GitHub Actions Cron-Trigger (Schedule) sind oft unzuverlässig und können zu verzögerten oder übersprungenen Ausführungen führen. Eine verlässlichere Methode ist die Nutzung des kostenlosen Dienstes **[cron-job.org](https://cron-job.org/de/)**, der externe Webrequests als Trigger einsetzt.

Um unsere Skripte (`main.py` für die tägliche Eisbach-Vorhersage und `src/archive_forecast.py` für das 5-Tage-Archiv) zuverlässig zu triggern, gibt es zwei hervorragende Ansätze:

## Methode A: Webhook-Trigger für GitHub Actions (Empfohlen, Serverless)
Anstatt den unzuverlässigen `schedule` Event von GitHub Actions zu nutzen, bauen wir einen Workflow, der auf das `repository_dispatch` Event hört. Dann lassen wir cron-job.org diesen Trigger auslösen.

**1. GitHub Action Workflow (`.github/workflows/trigger.yml`) erstellen:**
```yaml
name: Tägliche Vorhersage & Archivierung
on:
  repository_dispatch:
    types: [daily_cron]
jobs:
  run-scripts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Führe Eisbach Vorhersage aus (Hauptskript)
        run: PYTHONPATH=. python main.py

      - name: Führe Archivierung aus (läuft intern nur alle 5 Tage)
        run: PYTHONPATH=. python src/archive_forecast.py

      # Optional: Push der Änderungen (z.B. Archiv-CSV)
```

**2. Personal Access Token (PAT) bei GitHub erstellen:**
Gehe in deine GitHub Settings -> Developer settings -> Personal access tokens. Erstelle einen Token mit `repo` Rechten.

**3. Bei cron-job.org konfigurieren:**
* **URL:** `https://api.github.com/repos/DEIN_USERNAME/DEIN_REPO/dispatches`
* **HTTP-Methode:** `POST`
* **Headers:**
  * `Accept: application/vnd.github.v3+json`
  * `Authorization: Bearer DEIN_PAT_TOKEN`
* **Body:** `{"event_type": "daily_cron"}`
* **Ausführungsplan:** Einmal täglich (z.B. jeden Tag um 06:00 Uhr).


## Methode B: Eigenes Server-Backend (z.B. auf Render, Railway oder VPS)
Wenn du das Projekt auf einem Server oder einer PaaS (Platform as a Service) laufen hast, kannst du eine winzige Webanwendung (`app.py`) bereitstellen, die als Empfänger für cron-job.org dient.

**1. `app.py` im Projektverzeichnis erstellen:**
(Dieses Skript liegt bereits im Projektverzeichnis bei).
Es nutzt Flask, um einen Endpoint `/trigger-forecast` bereitzustellen. Ein Secret-Token sorgt dafür, dass nicht jeder das Skript starten kann.

**2. Server starten:**
Setze die Umgebungsvariable:
```bash
export CRON_SECRET="ein_sehr_geheimes_passwort"
python app.py
```
*(Im Produktivbetrieb gunicorn o.ä. verwenden)*

**3. Bei cron-job.org konfigurieren:**
* **URL:** `https://dein-server.de/trigger-forecast`
* **HTTP-Methode:** `POST` oder `GET`
* **Header (Authentifizierung):**
  * `Authorization: Bearer ein_sehr_geheimes_passwort`
* **Ausführungsplan:** Einmal täglich (z.B. jeden Tag um 06:00 Uhr).

*Hinweis zum Archivierungs-Skript:* Da `src/archive_forecast.py` intelligent programmiert ist und beim Start prüft, ob die letzten Daten bereits 5 Tage alt sind, kann dieses Skript gefahrlos **jeden Tag** zusammen mit dem Hauptskript getriggert werden. Es wird einfach an den Tagen 2 bis 5 stillschweigend abbrechen ("Ignoriere Trigger...").
