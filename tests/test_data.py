from datetime import datetime

import pandas as pd

from eisbach.data import fetch_brightsky_data, fetch_data_from_url


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
