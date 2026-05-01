import os
import subprocess
import threading
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hole das Secret Token aus den Umgebungsvariablen.
# Ohne dieses Token sollte niemand den Trigger auslösen können.
CRON_SECRET = os.environ.get("CRON_SECRET", "default_dev_secret")

def run_scripts():
    """Führt die Kernskripte asynchron aus."""
    logging.info("Starte Skript-Ausführung via Cron-Trigger...")

    # 1. Hauptskript (Eisbach Vorhersage)
    try:
        logging.info("Starte main.py...")
        subprocess.run(["python", "main.py"], check=True)
        logging.info("main.py erfolgreich abgeschlossen.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Fehler bei main.py: {e}")

    # 2. Archivierungs-Skript (prüft intern, ob 5 Tage vergangen sind)
    try:
        logging.info("Starte src/archive_forecast.py...")
        # PYTHONPATH=. stellen wir im subprocess call sicher
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        subprocess.run(["python", "src/archive_forecast.py"], env=env, check=True)
        logging.info("src/archive_forecast.py abgeschlossen.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Fehler bei archive_forecast.py: {e}")

@app.route('/trigger-forecast', methods=['GET', 'POST'])
def trigger():
    """
    Endpoint, der von cron-job.org aufgerufen wird.
    Erwartet den Header: Authorization: Bearer <CRON_SECRET>
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized. Missing or invalid Authorization header"}), 401

    token = auth_header.split(" ")[1]
    if token != CRON_SECRET:
        return jsonify({"error": "Forbidden. Invalid token"}), 403

    # Skripte asynchron im Hintergrund starten, damit cron-job.org sofort einen 200 OK bekommt
    # und nicht in einen Timeout läuft (Vorhersagen können dauern).
    thread = threading.Thread(target=run_scripts)
    thread.start()

    return jsonify({"message": "Trigger accepted. Scripts are running in the background."}), 200

if __name__ == '__main__':
    # Startet den Server auf Port 8080.
    # Im Produktiveinsatz sollte ein WSGI Server wie Gunicorn verwendet werden.
    app.run(host='0.0.0.0', port=8080)
