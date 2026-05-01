import os
import logging
from datetime import datetime, timedelta, timezone
from src.data import fetch_brightsky_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def archive_current_forecast():
    """
    Opportunistisch historische Wettervorhersagen (für die nächsten 10 Tage)
    von Bright Sky abrufen und lokal als CSV archivieren.
    """
    station_id = "03379"
    now = datetime.now(timezone.utc)
    # Bright Sky liefert für Vorhersagen oft Daten von "jetzt" bis in 10 Tage in die Zukunft.
    end_date = now + timedelta(days=10)

    logging.info(f"Hole Forecast-Daten für Station {station_id} ab {now.isoformat()} bis {end_date.isoformat()}")
    df = fetch_brightsky_data(now, end_date, station_id)

    if df is None or df.empty:
        logging.warning("Keine Wettervorhersagen erhalten. Abbruch.")
        return

    # Füge hinzu, WANN diese Vorhersage abgefragt wurde
    archive_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    df['archive_timestamp'] = archive_timestamp

    # Zielordner anlegen
    target_dir = os.path.join(os.path.dirname(__file__), "..", "data", "forecast_archive")
    os.makedirs(target_dir, exist_ok=True)

    # Dateiname basierend auf dem Abfragezeitpunkt
    filename = f"forecast_{now.strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(target_dir, filename)

    df.to_csv(filepath, index=False)
    logging.info(f"Erfolgreich {len(df)} Zeilen an Vorhersagedaten nach {filepath} archiviert.")

if __name__ == "__main__":
    archive_current_forecast()
