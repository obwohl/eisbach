
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
    Calculates T_max and checks if jacket is wearable using the 'Wet Shell Algorithm' (V4).

    Returns:
        is_wearable (bool)
        t_max (float)
        hum_malus (float) - V4 Exponential Membrane Choke
        wind_bonus (float) - V4 Evap Bonus (if wet) or V3 Wind Bonus (if dry)
        sorption_malus (float) - V4 Sorption Heat Malus (negative value)
    """
    # 1. BASIS-WERT (Transition / Shell Only Mode)
    # V4: Dynamic Base. We use 18.5°C (Shell Only) as the defining upper limit for "Wearability".
    limit = 18.5

    # Extract values
    temp = row['temperature']
    rh = row['relative_humidity']
    wind = row['wind_speed'] # km/h
    regen = row['precipitation'] # mm

    if pd.isna(temp) or pd.isna(rh):
        return False, 0.0, 0.0, 0.0, 0.0

    # 2. FEUCHTIGKEITS-MALUS (The Membrane Choke)
    # V4: Exponential decay.
    hum_malus = 0.0
    if rh > 60:
        hum_malus = ((rh - 60) / 40.0) ** 3 * 8.0
        limit -= hum_malus

    # 3. REGEN & WIND EFFEKT (Wet Shell vs Dry Shell)
    sorption_malus = 0.0
    wind_bonus = 0.0

    if regen > 0:
        # --- WET SCENARIO (V4) ---
        sorption_malus = -1.5
        limit += sorption_malus # Subtracts 1.5

        # B. Evaporative Bonus (Unlocked by Wind)
        wind_bonus = 0.2 * wind
        limit += wind_bonus

    else:
        # --- DRY SCENARIO (V3 Logic retained for Dry) ---
        # Wind Bonus (Ventilation)
        wind_factor = 0.0
        if wind > 5:
            wind_factor = (wind - 5) * 0.15

        wind_bonus = min(3.0, wind_factor)
        limit += wind_bonus

    t_max = limit

    # 4. Decision
    is_wearable = temp <= t_max

    return is_wearable, t_max, hum_malus, wind_bonus, sorption_malus

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

def calculate_monthly_period_stats(df):
    """
    Calculates the percentage of wearable periods (Whole Day, Light Day, Night) per month.
    Returns a DataFrame indexed by month (1-12) with columns:
    ['whole_day_pct', 'light_day_pct', 'night_pct', 'whole_day_count', 'whole_day_total', etc.]
    """
    logger.info("Calculating Monthly Period Stats...")

    def is_period_wearable(sub_df):
        if sub_df.empty:
            return None
        total_h = len(sub_df)
        not_wearable_h = len(sub_df[sub_df['is_wearable'] == False])
        pct_not_wearable = (not_wearable_h / total_h) * 100
        return pct_not_wearable <= 10.0

    df['date_str'] = df['timestamp'].dt.date
    grouped = df.groupby('date_str')

    # Structure to hold aggregated data
    # Month -> { type -> {total, wearable} }
    stats_data = {m: {'whole_day': {'total': 0, 'wearable': 0},
                      'light_day': {'total': 0, 'wearable': 0},
                      'night': {'total': 0, 'wearable': 0}} for m in range(1, 13)}

    for date, group in grouped:
        month = pd.to_datetime(date).month

        # Whole Day
        res_whole = is_period_wearable(group)
        if res_whole is not None:
            stats_data[month]['whole_day']['total'] += 1
            if res_whole:
                stats_data[month]['whole_day']['wearable'] += 1

        # Light Day
        day_group = group[group['day_night'] == 'Day']
        res_day = is_period_wearable(day_group)
        if res_day is not None:
            stats_data[month]['light_day']['total'] += 1
            if res_day:
                stats_data[month]['light_day']['wearable'] += 1

        # Night
        night_group = group[group['day_night'] == 'Night']
        res_night = is_period_wearable(night_group)
        if res_night is not None:
            stats_data[month]['night']['total'] += 1
            if res_night:
                stats_data[month]['night']['wearable'] += 1

    # Convert to DataFrame
    rows = []
    for m in range(1, 13):
        d = stats_data[m]

        wd_total = d['whole_day']['total']
        wd_wear = d['whole_day']['wearable']
        wd_pct = (wd_wear / wd_total * 100) if wd_total > 0 else 0

        ld_total = d['light_day']['total']
        ld_wear = d['light_day']['wearable']
        ld_pct = (ld_wear / ld_total * 100) if ld_total > 0 else 0

        n_total = d['night']['total']
        n_wear = d['night']['wearable']
        n_pct = (n_wear / n_total * 100) if n_total > 0 else 0

        rows.append({
            'month': m,
            'whole_day_pct': wd_pct, 'whole_day_wearable': wd_wear, 'whole_day_total': wd_total,
            'light_day_pct': ld_pct, 'light_day_wearable': ld_wear, 'light_day_total': ld_total,
            'night_pct': n_pct, 'night_wearable': n_wear, 'night_total': n_total
        })

    return pd.DataFrame(rows).set_index('month')

def generate_plots(df, period_stats_df):
    """
    Generates multiple plots:
    1. Heatmap (Original)
    2. Monthly Overview (Hourly)
    3. Day/Night Overview (Hourly)
    4. Monthly Period Overview (New V4 Plot)
    """
    logger.info("Generating plots...")

    color_no = '#FF6B6B'
    color_yes = '#4ECDC4'

    # --- 1. Heatmap ---
    df['date_str'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour

    pivot_table = df.pivot_table(index='hour', columns='date_str', values='is_wearable', aggfunc='max')
    pivot_table = pivot_table.fillna(False).astype(int)

    cmap = ListedColormap([color_no, color_yes])

    plt.figure(figsize=(24, 12))
    plt.imshow(pivot_table, cmap=cmap, aspect='auto', origin='lower', interpolation='nearest')

    plt.ylabel("Uhrzeit", fontsize=14, labelpad=10)
    plt.xlabel("Datum", fontsize=14, labelpad=10)
    plt.title("Tragbarkeit der Jacke in München (5 Jahre) - Heatmap (Modell V4)", fontsize=18, pad=20)

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

    # --- 2. Monthly Overview (Hourly) ---
    monthly = df.groupby(df['timestamp'].dt.month)['is_wearable'].agg(['count', 'sum'])
    monthly['pct'] = (monthly['sum'] / monthly['count']) * 100
    monthly['not_wearable_pct'] = 100 - monthly['pct']

    plt.figure(figsize=(12, 6))
    month_indices = range(1, 13)
    p1 = plt.bar(month_indices, monthly['pct'], color=color_yes, label='Tragbar')
    p2 = plt.bar(month_indices, monthly['not_wearable_pct'], bottom=monthly['pct'], color=color_no, label='Nicht Tragbar')

    plt.xlabel("Monat")
    plt.ylabel("Anteil (%)")
    plt.title("Tragbarkeit nach Monaten (Stundenbasiert) - V4")
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
    plt.title("Tragbarkeit Tag vs. Nacht - V4")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('jacket_plot_day_night.png', dpi=150)

    # --- 4. Period-Based Overview (NEW) ---
    # Grouped Bar Chart: Month on X, Percentage on Y.
    # Grouped by: Light Day, Night, Whole Day

    plt.figure(figsize=(14, 7))

    bar_width = 0.25
    x = np.arange(len(month_indices)) # 0 to 11

    # Data
    y_light = period_stats_df['light_day_pct']
    y_night = period_stats_df['night_pct']
    y_whole = period_stats_df['whole_day_pct']

    # Colors
    c_light = '#F4D03F' # Sunny Yellow/Gold
    c_night = '#2E4053' # Dark Blue/Grey
    c_whole = '#27AE60' # Green

    # Bars
    r1 = plt.bar(x - bar_width, y_light, width=bar_width, color=c_light, label='Lichttag')
    r2 = plt.bar(x, y_night, width=bar_width, color=c_night, label='Nacht')
    r3 = plt.bar(x + bar_width, y_whole, width=bar_width, color=c_whole, label='Ganzer Tag')

    # Formatting
    plt.xlabel('Monat', fontsize=12)
    plt.ylabel('Wahrscheinlichkeit (%)', fontsize=12)
    plt.title('Tragbarkeits-Wahrscheinlichkeit nach Perioden (Tage) - V4', fontsize=16)
    plt.xticks(x, ['Jan', 'Feb', 'Mär', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez'])
    plt.ylim(0, 115) # Space for labels

    # Add percentage labels on top
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.0f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    add_labels(r1)
    add_labels(r2)
    add_labels(r3)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('jacket_plot_periods.png', dpi=150)

    logger.info("Plots saved: jacket_plot_heatmap.png, jacket_plot_monthly.png, jacket_plot_day_night.png, jacket_plot_periods.png")

def main():
    # 1. Fetch Data (5 Years = ~1826 days)
    # Adding a few days buffer
    df = fetch_weather_data_chunked(days_back=365 * 5 + 5)

    if df is None or df.empty:
        logger.error("No data available.")
        return

    # 2. Apply Algorithm
    logger.info("Calculating wearability (V4)...")
    results = df.apply(calculate_wearability, axis=1, result_type='expand')
    df['is_wearable'] = results[0]
    df['t_max'] = results[1]
    df['hum_malus'] = results[2]
    df['wind_bonus'] = results[3]
    df['sorption_malus'] = results[4]

    # 3. Stratify Day/Night
    logger.info("Calculating Day/Night...")
    city = LocationInfo("Munich", "Germany", "Europe/Berlin", 48.1351, 11.5820)

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

    # 4. Statistics Calculation

    # A. Hourly Stats
    total_hours = len(df)
    wearable_hours = df['is_wearable'].sum()
    not_wearable_hours = total_hours - wearable_hours
    wearable_pct = (wearable_hours / total_hours) * 100 if total_hours > 0 else 0

    day_df = df[df['day_night'] == 'Day']
    night_df = df[df['day_night'] == 'Night']
    wearable_day = day_df['is_wearable'].sum()
    total_day = len(day_df)
    wearable_day_pct = (wearable_day / total_day * 100) if total_day > 0 else 0
    wearable_night = night_df['is_wearable'].sum()
    total_night = len(night_df)
    wearable_night_pct = (wearable_night / total_night * 100) if total_night > 0 else 0

    # B. Period Stats
    period_stats_df = calculate_monthly_period_stats(df)

    # Format Monthly Period Stats for Markdown
    month_names = {
        1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
        7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
    }

    monthly_period_section_lines = []
    for m in range(1, 13):
        row = period_stats_df.loc[m]
        monthly_period_section_lines.append(
            f"### {month_names[m]}\n"
            f"- **Ganzer Tag:** {row['whole_day_pct']:.1f}% ({int(row['whole_day_wearable'])}/{int(row['whole_day_total'])} Tage)\n"
            f"- **Lichttag:** {row['light_day_pct']:.1f}% ({int(row['light_day_wearable'])}/{int(row['light_day_total'])} Tage)\n"
            f"- **Nacht:** {row['night_pct']:.1f}% ({int(row['night_wearable'])}/{int(row['night_total'])} Nächte)\n"
        )
    monthly_period_section = "\n".join(monthly_period_section_lines)

    # Global Period Stats (Summing up)
    total_whole_days = period_stats_df['whole_day_total'].sum()
    wearable_whole_days = period_stats_df['whole_day_wearable'].sum()

    total_light_days = period_stats_df['light_day_total'].sum()
    wearable_light_days = period_stats_df['light_day_wearable'].sum()

    total_nights = period_stats_df['night_total'].sum()
    wearable_nights = period_stats_df['night_wearable'].sum()

    # Hourly Monthly Section
    df['month'] = df['timestamp'].dt.month
    monthly_stats = []
    for m in range(1, 13):
        m_df = df[df['month'] == m]
        if m_df.empty: continue
        m_total = len(m_df)
        m_wearable = m_df['is_wearable'].sum()
        m_wearable_pct = (m_wearable / m_total * 100)

        m_day = m_df[m_df['day_night'] == 'Day']
        m_day_total = len(m_day)
        m_day_wearable = m_day['is_wearable'].sum()
        m_day_wearable_pct = (m_day_wearable / m_day_total * 100) if m_day_total > 0 else 0

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


    # Top 10 Coldest/Warmest (Same as before)
    not_wearable_df = df[df['is_wearable'] == False].copy()
    coldest_not_wearable = not_wearable_df.sort_values('temperature', ascending=True).head(10)
    coldest_reasons = []
    for _, row in coldest_not_wearable.iterrows():
        reason = (f"- **{row['timestamp']}**: Temp {row['temperature']:.1f}°C > T_max {row['t_max']:.1f}°C "
                  f"(Base 18.5 - Malus_Feuchte {row['hum_malus']:.2f} + Malus_Sorption {row['sorption_malus']:.2f} + Bonus_Wind {row['wind_bonus']:.2f}) "
                  f"| RH: {row['relative_humidity']:.0f}%, Rain: {row['precipitation']:.1f}mm, Wind: {row['wind_speed']:.1f}km/h")
        coldest_reasons.append(reason)

    wearable_df = df[df['is_wearable'] == True].copy()
    warmest_wearable = wearable_df.sort_values('temperature', ascending=False).head(10)
    warmest_reasons = []
    for _, row in warmest_wearable.iterrows():
        reason = (f"- **{row['timestamp']}**: Temp {row['temperature']:.1f}°C <= T_max {row['t_max']:.1f}°C "
                  f"(Base 18.5 - Malus_Feuchte {row['hum_malus']:.2f} + Malus_Sorption {row['sorption_malus']:.2f} + Bonus_Wind {row['wind_bonus']:.2f}) "
                  f"| Rain: {row['precipitation']:.1f}mm, Wind: {row['wind_speed']:.1f}km/h")
        warmest_reasons.append(reason)


    stats_md = f"""# Statistik zur Tragbarkeit der Jacke (MooRER ISAC-LL) - 5 Jahre
## Modell V4 ("Biometeorological Validation")

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

## Analyse nach Perioden (Ganzer Tag / Lichttag / Nacht)
Definition: Eine Periode gilt als "Tragbar", wenn maximal 10% der Stunden darin "Nicht Tragbar" sind.

### Gesamtübersicht
- **Ganzer Tag:** {int(wearable_whole_days)} / {int(total_whole_days)} Tage ({(wearable_whole_days/total_whole_days*100):.1f}%)
- **Lichttag:** {int(wearable_light_days)} / {int(total_light_days)} Tage ({(wearable_light_days/total_light_days*100):.1f}%)
- **Nacht:** {int(wearable_nights)} / {int(total_nights)} Nächte ({(wearable_nights/total_nights*100):.1f}%)

### Nach Monaten Stratifiziert (Perioden)
{monthly_period_section}

## Analyse nach Monaten (Jahresverlauf) - Stundenbasiert
{monthly_section}

## Modell-Updates in V4
(Siehe vorherige Versionen oder Dokumentation für Details zu V4 Logik)

### Plausibilitäts-Check (Extremwerte)
#### Warum geht die Jacke bei diesen "kalten" Temperaturen NICHT?
{chr(10).join(coldest_reasons)}

#### Warum geht die Jacke hier DOCH noch?
{chr(10).join(warmest_reasons)}
"""

    with open('jacket_stats.md', 'w') as f:
        f.write(stats_md)
    logger.info("Statistics saved to jacket_stats.md")

    # 5. Plot
    generate_plots(df, period_stats_df)

if __name__ == "__main__":
    main()
