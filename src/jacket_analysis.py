
import pandas as pd
import requests
import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_weather_data_chunked(days_back=1826):
    """Fetches hourly weather data for Munich for the last n days in chunks."""
    station_id = "03379" # Munich City (DWD)

    end_date_final = datetime.datetime.now(datetime.timezone.utc)
    start_date_final = end_date_final - datetime.timedelta(days=days_back)

    all_dfs = []

    # Chunk size in days (e.g., 360 to be safe within 366 limit)
    chunk_size = 360

    current_end = end_date_final

    while current_end > start_date_final:
        current_start = current_end - datetime.timedelta(days=chunk_size)
        if current_start < start_date_final:
            current_start = start_date_final

        start_str = current_start.isoformat()
        end_str = current_end.isoformat()

        url = "https://api.brightsky.dev/weather"
        params = {
            'dwd_station_id': station_id,
            'date': start_str,
            'last_date': end_str
        }

        logger.info(f"Fetching chunk from {start_str} to {end_str}...")
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json().get('weather', [])

            if data:
                chunk_df = pd.DataFrame(data)
                all_dfs.append(chunk_df)
                logger.info(f"Fetched {len(data)} records in this chunk.")
            else:
                logger.warning("No data in this chunk.")

        except Exception as e:
            logger.error(f"Error fetching chunk: {e}")
            # Continue trying other chunks? Or abort?
            # If we miss a chunk, the stats will be wrong. Best to warn.

        # Move back in time
        current_end = current_start - datetime.timedelta(seconds=1)

    if not all_dfs:
        return None

    # Concatenate all chunks
    df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by timestamp (overlaps might occur at boundaries)
    df.drop_duplicates(subset=['timestamp'], inplace=True)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Bright Sky returns UTC. Convert to Berlin time.
    df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Berlin")

    # Sort by time
    df.sort_values('timestamp', inplace=True)

    # Reindex to ensure full hourly coverage (fill gaps)
    full_idx = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h', tz="Europe/Berlin")
    df = df.set_index('timestamp').reindex(full_idx)
    df.index.name = 'timestamp'

    # Interpolate small gaps in numeric columns
    numeric_cols = ['temperature', 'relative_humidity', 'wind_speed', 'precipitation']
    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit=24) # Limit interpolation to 24h gaps

    # Clean data:
    # 1. Drop rows where Temperature or Humidity is missing (critical for calculations)
    df.dropna(subset=['temperature', 'relative_humidity'], inplace=True)

    # 2. Fill remaining gaps for Wind and Precip with conservative values
    df['wind_speed'] = df['wind_speed'].fillna(0.0)
    df['precipitation'] = df['precipitation'].fillna(0.0)

    df = df.reset_index()

    return df

def calculate_wearability(row):
    """
    Calculates T_max and checks if jacket is wearable using the 'Wet Shell Algorithm' (V3).

    Returns:
        is_wearable (bool)
        t_max (float)
        hum_malus (float) - renamed from malus_feuchte
        wind_bonus (float) - renamed from bonus_wind
        rain_bonus (float) - new concept (Wet Shell Bonus) replacing malus_regen
    """
    # 1. BASIS-WERT (Casual Walking)
    # Etwas niedriger als "ganz offen", da wir Puffer brauchen
    limit = 20.0

    # Extract values
    temp = row['temperature']
    rh = row['relative_humidity']
    wind = row['wind_speed'] # km/h
    regen = row['precipitation'] # mm

    if pd.isna(temp) or pd.isna(rh):
        return False, 0.0, 0.0, 0.0, 0.0

    # 2. FEUCHTIGKEITS-MALUS (Der Membran-Killer)
    # Hohe Feuchte blockiert die Atmungsaktivität von innen nach außen.
    # Ab 60% rF ziehen wir progressiv ab.
    hum_malus = 0.0
    if rh > 60:
        hum_malus = (rh - 60) * 0.10
        limit -= hum_malus

    # 3. REGEN-EFFEKT (Der "Wet Shell" Bonus)
    # Die nasse Wolle verdunstet Wasser nach außen.
    # Das kühlt die Jacke ab. Wir addieren Temperatur-Toleranz.
    rain_bonus = 0.0
    if regen > 0:
        # Basis-Kühlung durch kaltes Regenwasser und Verdunstung auf dem Stoff
        wet_shell_bonus = 2.5

        # ABER: Wenn die Luft 100% gesättigt ist, verdunstet außen nichts!
        # Der Bonus muss schrumpfen, je höher die Luftfeuchte ist.
        # Physik: Verdunstungseffizienz = (100 - RH) / 40.0 (Skalierungsfaktor)
        evaporation_efficiency = (100 - rh) / 40.0

        # Wir garantieren aber mind. 1.0°C Bonus, da Regenwasser meist
        # kälter ist als die Luft (konduktive Kühlung).
        # NOTE: evaporation_efficiency can be negative if RH > 100 (unlikely but possible in data artifacts)
        # or just very small. max(1.0, ...) handles this.
        real_rain_bonus = max(1.0, wet_shell_bonus * evaporation_efficiency)

        rain_bonus = real_rain_bonus
        limit += rain_bonus

    # 4. WIND-EFFEKT (Die Ventilation)
    # Logik: Wenn es regnet, machen Sie die Jacke halb zu.
    # Der Wind kühlt dann schlechter als bei offener Jacke.
    wind_factor = 0.0
    if wind > 5:
        wind_factor = (wind - 5) * 0.15

    wind_bonus = 0.0
    if regen > 0:
        # Bei Regen ist die Jacke halb zu -> Wind wirkt nur zu 50%
        wind_bonus = wind_factor * 0.5
        limit += wind_bonus
    else:
        # Trocken -> Jacke ganz offen -> voller Wind-Bonus (max 3.0°C)
        wind_bonus = min(3.0, wind_factor)
        limit += wind_bonus

    t_max = limit

    # 5. Decision
    is_wearable = temp <= t_max

    # Return structure adapted: rain_bonus is positive, hum_malus is positive (subtracted later in logic description but calculated as positive value here)
    return is_wearable, t_max, hum_malus, wind_bonus, rain_bonus

def get_day_night_status(timestamp, city_info):
    """
    Determines if it is Day or Night for a given timestamp in Munich.
    """
    try:
        # Calculate sun info for that date
        s = sun(city_info.observer, date=timestamp.date(), tzinfo=ZoneInfo("Europe/Berlin"))
        sunrise = s['sunrise']
        sunset = s['sunset']

        if sunrise <= timestamp <= sunset:
            return "Day"
        else:
            return "Night"
    except Exception as e:
        # Fallback if astral fails (e.g. out of range latitude?? Unlikely for Munich)
        return "Unknown"

def generate_plots(df):
    """
    Generates multiple plots:
    1. Heatmap (Original)
    2. Monthly Overview
    3. Day/Night Overview
    4. Period Analysis
    """
    logger.info("Generating plots...")

    # --- 1. Heatmap ---
    df['date_str'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    pivot_table = df.pivot_table(index='hour', columns='date_str', values='is_wearable', aggfunc='max')
    pivot_table = pivot_table.fillna(False).astype(int)

    color_no = '#FF6B6B'
    color_yes = '#4ECDC4'
    cmap = ListedColormap([color_no, color_yes])

    plt.figure(figsize=(24, 12))
    plt.imshow(pivot_table, cmap=cmap, aspect='auto', origin='lower', interpolation='nearest')

    plt.ylabel("Uhrzeit", fontsize=14, labelpad=10)
    plt.xlabel("Datum", fontsize=14, labelpad=10)
    plt.title("Tragbarkeit der Jacke in München (5 Jahre) - Heatmap", fontsize=18, pad=20)

    plt.yticks(range(0, 24), labels=[f"{h:02d}:00" for h in range(0, 24)], fontsize=10)
    dates = pivot_table.columns
    num_dates = len(dates)
    step = max(1, num_dates // 25)
    tick_locs = range(0, num_dates, step)
    tick_labels = dates[0::step]
    plt.xticks(tick_locs, tick_labels, rotation=45, ha='right', fontsize=10)

    legend_elements = [mpatches.Patch(facecolor=color_yes, label='Tragbar'), mpatches.Patch(facecolor=color_no, label='Nicht Tragbar')]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig('jacket_plot_heatmap.png', dpi=200, bbox_inches='tight')

    # --- 2. Monthly Overview ---
    monthly = df.groupby(df['timestamp'].dt.month)['is_wearable'].agg(['count', 'sum'])
    monthly['pct'] = (monthly['sum'] / monthly['count']) * 100
    monthly['not_wearable_pct'] = 100 - monthly['pct']

    plt.figure(figsize=(12, 6))
    month_indices = range(1, 13)
    p1 = plt.bar(month_indices, monthly['pct'], color=color_yes, label='Tragbar')
    p2 = plt.bar(month_indices, monthly['not_wearable_pct'], bottom=monthly['pct'], color=color_no, label='Nicht Tragbar')

    plt.xlabel("Monat")
    plt.ylabel("Anteil (%)")
    plt.title("Tragbarkeit nach Monaten")
    plt.xticks(month_indices, ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig('jacket_plot_monthly.png', dpi=150)

    # --- 3. Day/Night Overview ---
    dn_stats = df.groupby('day_night')['is_wearable'].agg(['count', 'sum'])
    dn_stats['pct'] = (dn_stats['sum'] / dn_stats['count']) * 100

    # Ensure 'Day' and 'Night' exist
    day_pct = dn_stats.loc['Day', 'pct'] if 'Day' in dn_stats.index else 0
    night_pct = dn_stats.loc['Night', 'pct'] if 'Night' in dn_stats.index else 0

    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Tag (Hell)', 'Nacht (Dunkel)'], [day_pct, night_pct], color=[color_yes, '#45B7AF'])
    plt.ylim(0, 100)
    plt.ylabel("Tragbarkeit (%)")
    plt.title("Tragbarkeit Tag vs. Nacht")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('jacket_plot_day_night.png', dpi=150)

    logger.info("Plots saved: jacket_plot_heatmap.png, jacket_plot_monthly.png, jacket_plot_day_night.png")

def main():
    # 1. Fetch Data (5 Years = ~1826 days)
    # Adding a few days buffer
    df = fetch_weather_data_chunked(days_back=365 * 5 + 5)

    if df is None or df.empty:
        logger.error("No data available.")
        return

    # 2. Apply Algorithm
    logger.info("Calculating wearability...")
    results = df.apply(calculate_wearability, axis=1, result_type='expand')
    df['is_wearable'] = results[0]
    df['t_max'] = results[1]
    df['hum_malus'] = results[2] # Renamed
    df['wind_bonus'] = results[3] # Renamed
    df['rain_bonus'] = results[4] # Renamed

    # 3. Stratify Day/Night
    logger.info("Calculating Day/Night...")
    city = LocationInfo("Munich", "Germany", "Europe/Berlin", 48.1351, 11.5820)

    # Optimize applying getting day/night - calculate once per unique date?
    # Actually, sunrise/sunset changes daily. We need it per timestamp.
    # To speed up, we can calculate sunrise/sunset for each unique date, then map.
    unique_dates = df['timestamp'].dt.date.unique()
    sun_cache = {}

    logger.info("Pre-calculating sun data...")
    for d in unique_dates:
        try:
            s = sun(city.observer, date=d, tzinfo=ZoneInfo("Europe/Berlin"))
            sun_cache[d] = (s['sunrise'], s['sunset'])
        except:
            sun_cache[d] = (None, None)

    def quick_day_night(ts):
        d = ts.date()
        sunrise, sunset = sun_cache.get(d, (None, None))
        if sunrise and sunset:
            if sunrise <= ts <= sunset:
                return "Day"
        return "Night"

    df['day_night'] = df['timestamp'].apply(quick_day_night)

    # 4. Statistics
    total_hours = len(df)
    wearable_hours = df['is_wearable'].sum()
    not_wearable_hours = total_hours - wearable_hours

    wearable_pct = (wearable_hours / total_hours) * 100 if total_hours > 0 else 0

    # Stratified
    day_df = df[df['day_night'] == 'Day']
    night_df = df[df['day_night'] == 'Night']

    wearable_day = day_df['is_wearable'].sum()
    total_day = len(day_df)
    wearable_day_pct = (wearable_day / total_day * 100) if total_day > 0 else 0

    wearable_night = night_df['is_wearable'].sum()
    total_night = len(night_df)
    wearable_night_pct = (wearable_night / total_night * 100) if total_night > 0 else 0

    # Top 10 Coldest Not Wearable
    not_wearable_df = df[df['is_wearable'] == False].copy()
    coldest_not_wearable = not_wearable_df.sort_values('temperature', ascending=True).head(10)

    coldest_reasons = []
    for _, row in coldest_not_wearable.iterrows():
        reason = (f"- **{row['timestamp']}**: Temp {row['temperature']:.1f}°C > T_max {row['t_max']:.1f}°C "
                  f"(Feuchte-Malus: {row['hum_malus']:.2f}, Wet-Shell-Bonus: {row['rain_bonus']:.2f}, Wind-Bonus: {row['wind_bonus']:.2f}) "
                  f"| RH: {row['relative_humidity']:.0f}%, Rain: {row['precipitation']:.1f}mm, Wind: {row['wind_speed']:.1f}km/h")
        coldest_reasons.append(reason)

    # Top 10 Warmest Wearable
    wearable_df = df[df['is_wearable'] == True].copy()
    warmest_wearable = wearable_df.sort_values('temperature', ascending=False).head(10)

    warmest_reasons = []
    for _, row in warmest_wearable.iterrows():
        reason = (f"- **{row['timestamp']}**: Temp {row['temperature']:.1f}°C <= T_max {row['t_max']:.1f}°C "
                  f"(Wet-Shell-Bonus: {row['rain_bonus']:.2f}, Wind-Bonus: {row['wind_bonus']:.2f}) | Rain: {row['precipitation']:.1f}mm, Wind: {row['wind_speed']:.1f}km/h")
        warmest_reasons.append(reason)

    # Monthly Stratification
    df['month'] = df['timestamp'].dt.month
    monthly_stats = []

    # German month names
    month_names = {
        1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
        7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
    }

    for m in range(1, 13):
        m_df = df[df['month'] == m]
        if m_df.empty:
            continue

        m_total = len(m_df)
        m_wearable = m_df['is_wearable'].sum()
        m_wearable_pct = (m_wearable / m_total * 100)

        # Day
        m_day = m_df[m_df['day_night'] == 'Day']
        m_day_total = len(m_day)
        m_day_wearable = m_day['is_wearable'].sum()
        m_day_wearable_pct = (m_day_wearable / m_day_total * 100) if m_day_total > 0 else 0

        # Night
        m_night = m_df[m_df['day_night'] == 'Night']
        m_night_total = len(m_night)
        m_night_wearable = m_night['is_wearable'].sum()
        m_night_wearable_pct = (m_night_wearable / m_night_total * 100) if m_night_total > 0 else 0

        monthly_stats.append(
            f"### {month_names[m]}\n"
            f"- **Gesamt:** {m_wearable} / {m_total} Stunden ({m_wearable_pct:.1f}% tragbar)\n"
            f"- **Tag:** {m_day_wearable_pct:.1f}% tragbar ({m_day_wearable}/{m_day_total})\n"
            f"- **Nacht:** {m_night_wearable_pct:.1f}% tragbar ({m_night_wearable}/{m_night_total})\n"
        )

    monthly_section = "\n".join(monthly_stats)

    # --- New Section: Period-Based Analysis (Max 10% Exception) ---
    logger.info("Calculating Period-Based Analysis...")

    # Helper to check if a period is "wearable" (<= 10% not wearable)
    def is_period_wearable(sub_df):
        if sub_df.empty:
            return None # No data for this period
        total_h = len(sub_df)
        not_wearable_h = len(sub_df[sub_df['is_wearable'] == False])
        pct_not_wearable = (not_wearable_h / total_h) * 100
        return pct_not_wearable <= 10.0

    # Group by date
    df['date_str'] = df['timestamp'].dt.date
    grouped = df.groupby('date_str')

    period_stats = {
        'whole_day': {'total': 0, 'wearable': 0},
        'light_day': {'total': 0, 'wearable': 0},
        'night': {'total': 0, 'wearable': 0}
    }

    monthly_period_stats = {m: {'whole_day': {'total': 0, 'wearable': 0},
                                'light_day': {'total': 0, 'wearable': 0},
                                'night': {'total': 0, 'wearable': 0}} for m in range(1, 13)}

    for date, group in grouped:
        month = pd.to_datetime(date).month

        # Whole Day
        res_whole = is_period_wearable(group)
        if res_whole is not None:
            period_stats['whole_day']['total'] += 1
            monthly_period_stats[month]['whole_day']['total'] += 1
            if res_whole:
                period_stats['whole_day']['wearable'] += 1
                monthly_period_stats[month]['whole_day']['wearable'] += 1

        # Light Day
        day_group = group[group['day_night'] == 'Day']
        res_day = is_period_wearable(day_group)
        if res_day is not None:
            period_stats['light_day']['total'] += 1
            monthly_period_stats[month]['light_day']['total'] += 1
            if res_day:
                period_stats['light_day']['wearable'] += 1
                monthly_period_stats[month]['light_day']['wearable'] += 1

        # Night
        night_group = group[group['day_night'] == 'Night']
        res_night = is_period_wearable(night_group)
        if res_night is not None:
            period_stats['night']['total'] += 1
            monthly_period_stats[month]['night']['total'] += 1
            if res_night:
                period_stats['night']['wearable'] += 1
                monthly_period_stats[month]['night']['wearable'] += 1

    # Format Monthly Period Stats
    monthly_period_section_lines = []
    for m in range(1, 13):
        stats = monthly_period_stats[m]

        # Calculate percentages
        wd_pct = (stats['whole_day']['wearable'] / stats['whole_day']['total'] * 100) if stats['whole_day']['total'] > 0 else 0
        ld_pct = (stats['light_day']['wearable'] / stats['light_day']['total'] * 100) if stats['light_day']['total'] > 0 else 0
        n_pct = (stats['night']['wearable'] / stats['night']['total'] * 100) if stats['night']['total'] > 0 else 0

        monthly_period_section_lines.append(
            f"### {month_names[m]}\n"
            f"- **Ganzer Tag:** {wd_pct:.1f}% ({stats['whole_day']['wearable']}/{stats['whole_day']['total']} Tage)\n"
            f"- **Lichttag:** {ld_pct:.1f}% ({stats['light_day']['wearable']}/{stats['light_day']['total']} Tage)\n"
            f"- **Nacht:** {n_pct:.1f}% ({stats['night']['wearable']}/{stats['night']['total']} Nächte)\n"
        )
    monthly_period_section = "\n".join(monthly_period_section_lines)

    stats_md = f"""# Statistik zur Tragbarkeit der Jacke (MooRER ISAC-LL) - 5 Jahre

## Zeitraum
{df['timestamp'].min()} bis {df['timestamp'].max()}

## Gesamt
- **Gesamtstunden:** {total_hours}
- **Tragbar:** {wearable_hours} Stunden ({wearable_pct:.2f}%)
- **Nicht Tragbar:** {not_wearable_hours} Stunden ({100 - wearable_pct:.2f}%)

## Nach Tageszeit (Tag vs. Nacht)
### Tag (zwischen Sonnenaufgang und Sonnenuntergang)
- **Stunden:** {total_day}
- **Tragbar:** {wearable_day} Stunden ({wearable_day_pct:.2f}%)
- **Nicht Tragbar:** {total_day - wearable_day} Stunden ({100 - wearable_day_pct:.2f}%)

### Nacht
- **Stunden:** {total_night}
- **Tragbar:** {wearable_night} Stunden ({wearable_night_pct:.2f}%)
- **Nicht Tragbar:** {total_night - wearable_night} Stunden ({100 - wearable_night_pct:.2f}%)

## Analyse nach Monaten (Jahresverlauf) - Stundenbasiert
{monthly_section}

## Analyse nach Perioden (Ganzer Tag / Lichttag / Nacht)
Definition: Eine Periode gilt als "Tragbar", wenn maximal 10% der Stunden darin "Nicht Tragbar" sind.

### Gesamtübersicht
- **Ganzer Tag:** {period_stats['whole_day']['wearable']} / {period_stats['whole_day']['total']} Tage ({(period_stats['whole_day']['wearable']/period_stats['whole_day']['total']*100):.1f}%)
- **Lichttag:** {period_stats['light_day']['wearable']} / {period_stats['light_day']['total']} Tage ({(period_stats['light_day']['wearable']/period_stats['light_day']['total']*100):.1f}%)
- **Nacht:** {period_stats['night']['wearable']} / {period_stats['night']['total']} Nächte ({(period_stats['night']['wearable']/period_stats['night']['total']*100):.1f}%)

### Nach Monaten Stratifiziert
{monthly_period_section}

## Plausibilitäts-Check der Formel ("Wet Shell" Update)

### Warum geht die Jacke manchmal bei gemäßigten Temperaturen NICHT?
Hier schlägt meist der **Feuchtigkeits-Malus** zu.
1.  **Hoher Luftfeuchtigkeit (>60%):** Die Membran "atmet" schlechter. Das Limit sinkt drastisch (0.1°C pro % über 60). Bei 90% Feuchte fehlen 3.0°C am Limit.
2.  **Schwüle ohne Regen:** Wenn es feucht ist, aber NICHT regnet, fehlt der kühlende "Wet Shell" Bonus.

{chr(10).join(coldest_reasons)}

### Warum geht die Jacke bei Regen oft doch noch gut? (Der Wet Shell Effekt)
Hier greift die neue Logik: **Regen kühlt die Oberfläche**.
1.  **Wet Shell Bonus:** Regen bringt bis zu +2.5°C Toleranz, solange die Luft nicht 100% gesättigt ist (Verdunstung).
2.  **Wind:** Wirkt bei geschlossener Jacke (Regenfall) nur zu 50%, aber hilft immer noch.

{chr(10).join(warmest_reasons)}
"""

    with open('jacket_stats.md', 'w') as f:
        f.write(stats_md)
    logger.info("Statistics saved to jacket_stats.md")

    # 5. Plot
    generate_plots(df)

if __name__ == "__main__":
    main()
