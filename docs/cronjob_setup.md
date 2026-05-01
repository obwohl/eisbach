# Einrichtung von Cron-Jobs zur Vorhersage-Generierung

GitHub Actions Cron-Trigger (Schedule) sind oft unzuverlässig. Eine verlässlichere Methode ist die Nutzung des kostenlosen Dienstes **[cron-job.org](https://cron-job.org/de/)**, der per Webhook unsere GitHub Action exakt nach Zeitplan auslöst.

Wir haben einen Workflow (`.github/workflows/daily_cron.yml`) angelegt, der auf ein `repository_dispatch` Event hört. Dieser Workflow führt sowohl das Hauptskript (`main.py`) als auch das Archivierungs-Skript (`src/archive_forecast.py`) aus und pusht die neuen Daten (z.B. die Archiv-CSV) automatisch ins Repository zurück.

Damit das funktioniert, richte cron-job.org wie folgt ein:

**1. Personal Access Token (PAT) bei GitHub erstellen**
* Gehe in deine GitHub Settings -> Developer settings -> Personal access tokens (Fine-grained oder Classic).
* Erstelle einen Token mit Schreibrechten auf dieses Repository (`repo` Scope).
* Kopiere den Token (er wird nur einmal angezeigt).

**2. Bei cron-job.org konfigurieren**
* **URL:** `https://api.github.com/repos/DEIN_GITHUB_USERNAME/DEIN_REPOSITORY_NAME/dispatches` (Passe Username und Repo an!)
* **HTTP-Methode:** `POST`
* **Ausführungsplan:** Einmal täglich (z.B. jeden Tag um 06:00 Uhr).
* **Erweiterte Einstellungen / Header:**
  * `Accept: application/vnd.github.v3+json`
  * `Authorization: Bearer DEIN_PAT_TOKEN` (Deinen vorhin generierten Token hier einfügen)
* **Erweiterte Einstellungen / Body:**
  * Wähle als Typ "Raw / Custom" (oder ähnlich) und füge folgendes JSON ein:
  ```json
  {"event_type": "trigger_forecast"}
  ```

Sobald cron-job.org diesen Ping sendet, startet GitHub die Action.
*Hinweis zum Archivierungs-Skript:* Da `src/archive_forecast.py` intelligent programmiert ist, bricht es seine Ausführung ab, wenn die letzte 5-Tage-Archivierung noch keine 5 Tage her ist. Es kann (und sollte) also gefahrlos jeden Tag mitgetriggert werden.
