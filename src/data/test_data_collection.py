import unittest
from unittest.mock import MagicMock
import pandas as pd
from data_collection import get_data

class TestGetData(unittest.TestCase):

    def setUp(self):
        # Create a mock session
        self.session = MagicMock()
        
        # Create mock curve data
        self.curve_data = pd.Series(
            data=[1, 2, 3, 4],
            index=pd.date_range(start="2021-01-01", periods=4, freq='H')
        )
        
        # Mock the get_curve and get_data methods
        curve = MagicMock()
        curve.get_data.return_value.to_pandas.return_value = self.curve_data
        self.session.get_curve.return_value = curve

    def test_resample(self):
        curve_names = ["exc de>se4 com mw cet h a"]
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-01-01 03:00:00")
        
        # Get data
        data = get_data(curve_names, self.session, start_date, end_date)
        
        # Check the number of data points before and after resampling
        original_points = len(self.curve_data)
        resampled_points = len(data[curve_names[0]])
        
        # Assert that the resampled points are correct (4 hours * 4 intervals per hour = 16)
        self.assertEqual(resampled_points, (original_points - 1) * 4 + 1)

if __name__ == '__main__':
    unittest.main()