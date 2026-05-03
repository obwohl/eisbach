# Technical Primer: Munich Weather Data Integration & Forecasting

This guide outlines the optimal strategy for integrating external weather data into our existing Munich pools (Hallenbäder) time series and provides a conceptual foundation for future predictive modeling.

## 1. Weather API Integration

We will use the **Bright Sky API** to fetch both historical weather data and future forecasts. This API is reliable, elegant, and directly queries DWD (Deutscher Wetterdienst) data.

### API Details
* **Endpoint:** `https://api.brightsky.dev/weather`
* **Station ID:** `03379` (Munich)
* **Required Parameters:**
  * `date`: The start date/time (ISO 8601 string)
  * `last_date`: The end date/time (ISO 8601 string)
* **Relevant Response Fields:** Extract `temperature` (air temperature) and `precipitation` (rainfall) from the JSON response.

### Implementation Guidelines
Since we currently record pool occupancy hourly in a long-format CSV, you should add the weather data (temperature and precipitation) as additional columns for each hourly timestamp. You can retrieve extensive historical data in bulk, so there is no need to implement complex incremental saving right now; simply fetch the required historical timeframe in one go when we are ready.

## 2. Strict Timezone Handling (Crucial)

To prevent misaligned time series and data leakage:
* **Normalize Everything to UTC:** All internal processing, including the existing pool occupancy time series, must be strictly normalized to UTC.
* **Audit Existing Data:** Please verify and consolidate the current pool data timestamps. Ensure they are explicitly handled and stored as UTC, not local Munich time.
* **API Requests:** When making requests to the Bright Sky API, ensure your `date` and `last_date` parameters are formatted in UTC.

## 3. Conceptual Outlook: Multivariable Forecasting with Chronos-2

While we are currently in the data-gathering phase, our ultimate goal is to predict pool occupancy using advanced time series models.

Your task is to **research and conceptualize** the state-of-the-art approach for multivariable time series prediction using exactly the **`amazon/chronos-2`** model.

### Strict Constraints for Chronos-2 Research
* **Model:** You must use the official `amazon/chronos-2` model (via Hugging Face).
* **Exclusions:** Do **not** use AutoGluon, "Bolt", or any other abstraction/variant. Read the original documentation and source paper directly.
* **Covariates:** We will use `temperature` and `precipitation` as weather features. Since we can obtain future weather forecasts from the Bright Sky API, these act as **future known covariates**.
* **Objective:** Investigate how to ideally structure and feed future known covariates into `amazon/chronos-2`. Unlike some legacy models where we manually shifted the time series to account for delays, Chronos-2 is a modern, deterministic encoder-only transformer. You need to determine the idiomatic, state-of-the-art method to provide these auxiliary weather features to Chronos-2 alongside the primary occupancy target, without manual data shifting.

Please familiarize yourself with the exact Hugging Face API for this model so we are prepared when we transition from data collection to prediction.