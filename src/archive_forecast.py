import os
import logging
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.data import fetch_brightsky_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def archive_current_forecast():
    """
    Opportunistisch historische 5-Tage-Wettervorhersagen von Bright Sky abrufen
    und in einer zentralen CSV Datei (append) archivieren.
    Schreibt nur, wenn die letzte Vorhersage >= 5 Tage her ist.
    """
    station_id = "03379"
    now = datetime.now(timezone.utc)

    # Zielordner und Dateiname
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "forecast_archive")
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, "forecast_5d_archive.csv")

    # Überprüfen, wann der letzte Eintrag gemacht wurde
    if os.path.exists(filepath):
        try:
            # Lade nur die archive_timestamp Spalte, um Speicher zu sparen
            existing_data = pd.read_csv(filepath, usecols=['archive_timestamp'])
            if not existing_data.empty:
                last_timestamp_str = existing_data['archive_timestamp'].iloc[-1]
                last_timestamp = datetime.strptime(last_timestamp_str, "%Y-%m-%dT%H:%M:%S%z")

                # Verwende einen Schwellenwert von 4 Tagen und 23 Stunden anstelle von genau 5 Tagen.
                # GitHub Actions Runner starten nicht immer auf die Sekunde genau.
                # Wenn der Job am Tag 6 minimal früher startet als am Tag 1, würde er sonst versehentlich abgelehnt werden.
                delta = now - last_timestamp
                if delta < timedelta(days=4, hours=23):
                    logging.info(f"Ignoriere Trigger. Die letzte Speicherung ist erst {delta.days} Tage und {delta.seconds//3600} Stunden her. (Erfordert ~5 Tage).")
                    return
        except Exception as e:
            logging.error(f"Fehler beim Lesen der bestehenden Archiv-Datei: {e}")
            # Bei Fehlern brechen wir ab, um die Datei nicht zu korrumpieren
            return

    # Wenn wir hier sind, sind entweder 5 Tage vergangen oder die Datei existiert noch nicht.
    # Lade exakt die 5-Tage Vorhersage
    end_date = now + timedelta(days=5)
    logging.info(f"Hole 5-Tage-Forecast-Daten für Station {station_id} ab {now.isoformat()} bis {end_date.isoformat()}")

    df = fetch_brightsky_data(now, end_date, station_id)

    if df is None or df.empty:
        logging.warning("Keine Wettervorhersagen erhalten. Abbruch.")
        return

    # Füge hinzu, WANN diese Vorhersage abgefragt wurde
    archive_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    df['archive_timestamp'] = archive_timestamp

    # Anhängen an die Datei (erstellt sie, wenn sie nicht existiert)
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    df.to_csv(filepath, mode='a', header=write_header, index=False)
    logging.info(f"Erfolgreich {len(df)} Zeilen an Vorhersagedaten an {filepath} angehängt.")

if __name__ == "__main__":
    archive_current_forecast()
