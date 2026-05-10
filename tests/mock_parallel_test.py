import sys
from unittest.mock import MagicMock, patch
import datetime

# Create a more robust mock for pandas
class MockPandas:
    def __init__(self):
        self.DataFrame = MagicMock()
        self.to_datetime = MagicMock(side_effect=lambda x, **kwargs: x)
        self.read_csv = MagicMock()
        self.concat = MagicMock()
        self.Timedelta = MagicMock(side_effect=lambda **kwargs: datetime.timedelta(**kwargs))

mock_pd = MockPandas()
sys.modules['pandas'] = mock_pd

# Mock subprocess
mock_subprocess = MagicMock()
sys.modules['subprocess'] = mock_subprocess

# Now we can import src.inference
import src.inference

def test_parallel_logic():
    print("Directly testing _run_single_backtest_task logic...")

    # Setup
    offset = 96
    last_timestamp = datetime.datetime(2024, 5, 10, 12, 0)

    # We need a mock df_long that behaves like a dataframe for filtering
    mock_df_long = MagicMock()
    # Mocking the comparison df_long['date'] <= backtest_end_date
    mock_df_long.__getitem__.return_value = mock_df_long
    mock_df_long.__le__.return_value = True # mock the comparison itself

    mock_df_long.copy.return_value = mock_df_long

    with patch('src.inference.load_forecast_from_archive', return_value=None), \
         patch('src.inference.pd.read_csv') as mock_read_csv, \
         patch('src.inference.subprocess.run') as mock_run:

        mock_read_csv.return_value = MagicMock() # The result DF

        result_offset, result_df = src.inference._run_single_backtest_task(offset, last_timestamp, mock_df_long)

        assert result_offset == offset
        assert mock_run.called
        print(f"Success: _run_single_backtest_task for offset {offset} triggered subprocess.run")

    print("Testing ProcessPoolExecutor usage in run_inference...")

    with patch('src.inference.ProcessPoolExecutor') as MockExecutor, \
         patch('src.inference.pd.to_datetime', side_effect=lambda x: x), \
         patch('src.inference.save_forecast_to_archive'), \
         patch('src.inference.os.makedirs'), \
         patch('src.inference.subprocess.run'):

        mock_executor_instance = MockExecutor.return_value.__enter__.return_value

        # Setup mock futures
        mock_future = MagicMock()
        mock_future.result.return_value = (96, MagicMock())
        mock_executor_instance.submit.return_value = mock_future

        # Mocking the dataframe logic inside run_inference
        mock_df_long = MagicMock()
        mock_df_long.__getitem__.return_value = mock_df_long
        mock_df_long.__le__.return_value = True # for df_long['date'] <= last_timestamp
        mock_df_long.dropna.return_value = mock_df_long
        mock_df_long.max.return_value = last_timestamp
        mock_df_long.copy.return_value = mock_df_long

        # We also need to mock the pd.read_csv at the end and in the middle
        with patch('src.inference.pd.read_csv', return_value=MagicMock()):
             try:
                 src.inference.run_inference(mock_df_long)
             except Exception as e:
                 print(f"run_inference partially executed (expectedly), checking executor. error was: {e}")

        assert mock_executor_instance.submit.called
        # Check if max_workers was passed correctly
        MockExecutor.assert_called_with(max_workers=2)
        print("Success: ProcessPoolExecutor.submit was called and max_workers=2 was verified")

if __name__ == "__main__":
    try:
        test_parallel_logic()
        print("Mock parallel logic test passed!")
    except Exception as e:
        print(f"Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
