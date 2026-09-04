from datetime import datetime

import pandas as pd

from eisbach.data import (
    ARCHIVED_WEATHER_FIELDS,
    assemble_long_frame,
    fetch_brightsky_data,
    fetch_data_from_url,
    get_prepared_weather_data,
)


def test_fetch_brightsky_data_empty(mocker):
    # Mock requests.get to return empty weather data
    mock_response = mocker.Mock()
    mock_response.json.return_value = {'weather': []}
    mock_response.raise_for_status = mocker.Mock()
    mocker.patch('requests.get', return_value=mock_response)

    start_date = datetime(2026, 4, 1)
    end_date = datetime(2026, 4, 2)
    df = fetch_brightsky_data(start_date, end_date, "03379")

    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_fetch_data_from_url_empty(mocker):
    # Mock a request with no table
    mock_response = mocker.Mock()
    mock_response.content = b"<html><body><p>No data</p></body></html>"
    mock_response.raise_for_status = mocker.Mock()
    mocker.patch('requests.get', return_value=mock_response)

    df = fetch_data_from_url("http://example.com", "wassertemp")

    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_fetch_data_from_url_valid(mocker):
    # Mock a request with a valid table
    mock_html = b"""
    <html>
    <body>
        <table class="tblsort">
            <thead>
                <tr><th>Datum/Uhrzeit</th><th>Wassertemperatur [C]</th></tr>
            </thead>
            <tbody>
                <tr><td>01.04.2026 12:00</td><td>10,5</td></tr>
            </tbody>
        </table>
    </body>
    </html>
    """
    mock_response = mocker.Mock()
    mock_response.content = mock_html
    mock_response.raise_for_status = mocker.Mock()
    mocker.patch('requests.get', return_value=mock_response)

    df = fetch_data_from_url("http://example.com", "wassertemp")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "wassertemp" in df.columns
    assert df["wassertemp"].iloc[0] == 10.5


def brightsky_payload(periods=6, **overrides):
    """A Bright Sky response shaped like the real one, hourly and in UTC."""
    stamps = pd.date_range("2026-09-01 00:00", periods=periods, freq="1h", tz="UTC")
    payload = {
        "timestamp": [s.isoformat() for s in stamps],
        "temperature": [18.0 + i for i in range(periods)],
        "precipitation": [0.0] * periods,
        "pressure_msl": [1013.0 + i for i in range(periods)],
        "sunshine": [float(i * 10) for i in range(periods)],
        "solar": [0.1 * i for i in range(periods)],
        "cloud_cover": [50] * periods,
        "relative_humidity": [None] * periods,
        "wind_speed": [11.1] * periods,
        "dew_point": [11.0] * periods,
        # Fields Bright Sky sends that we do not archive.
        "icon": ["clear-day"] * periods,
        "source_id": [1] * periods,
    }
    payload.update(overrides)
    return pd.DataFrame(payload)


def test_the_archival_fields_are_carried_through(mocker):
    """Covariates not archived are gone forever, so capture them before training needs
    them. Solar radiation in particular is the obvious driver of a shallow urban channel
    that air temperature alone cannot explain."""
    mocker.patch("eisbach.data.fetch_brightsky_data", return_value=brightsky_payload())

    weather = get_prepared_weather_data()

    assert set(ARCHIVED_WEATHER_FIELDS) <= set(weather.columns)
    assert weather["sunshine"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    assert weather["cloud_cover"].tolist() == [50.0] * 6


def test_fields_bright_sky_does_not_send_are_not_fatal(mocker):
    """The forecast must not fail over a covariate nothing reads yet."""
    payload = brightsky_payload().drop(columns=["solar", "sunshine"])
    mocker.patch("eisbach.data.fetch_brightsky_data", return_value=payload)

    weather = get_prepared_weather_data()

    assert "solar" not in weather.columns
    assert "cloud_cover" in weather.columns
    assert weather["lufttemperatur_c"].notna().all()


def test_an_unreported_hour_stays_missing_rather_than_becoming_zero(mocker):
    """"No sunshine value" is not "no sunshine", and this frame is archived as evidence.
    `relative_humidity` really does arrive all-null from the forecast endpoint."""
    payload = brightsky_payload()
    payload["sunshine"] = [None] * len(payload)
    mocker.patch("eisbach.data.fetch_brightsky_data", return_value=payload)

    weather = get_prepared_weather_data()

    assert weather["sunshine"].isna().all(), "a sum over nothing must not invent a zero"
    assert weather["relative_humidity"].isna().all()


def test_the_model_input_frame_is_untouched_by_the_extra_weather():
    """`CHANNELS` is the trained channel order and is load-bearing. Archiving more
    weather is archival only — the melt takes those three columns and nothing else, so
    the model's input frame must come out byte for byte identical."""
    water = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-01 00:00", periods=200, freq="1h", tz="UTC"),
        "wassertemp": [15.0 + i * 0.01 for i in range(200)],
    })
    index = pd.date_range("2026-09-01 00:00", periods=400, freq="1h", tz="UTC")
    lean = pd.DataFrame({
        "lufttemperatur_c": [18.0 + i * 0.01 for i in range(400)],
        "niederschlag_mm": 0.0,
        "pressure": [1013.0 + i * 0.01 for i in range(400)],
    }, index=index)

    rich = lean.copy()
    for i, field in enumerate(ARCHIVED_WEATHER_FIELDS):
        rich[field] = float(i)

    before = assemble_long_frame(water, lean)
    after = assemble_long_frame(water, rich)

    pd.testing.assert_frame_equal(before, after)
    assert before.to_csv().encode() == after.to_csv().encode()
