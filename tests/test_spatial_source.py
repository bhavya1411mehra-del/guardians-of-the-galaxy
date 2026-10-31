import unittest
from unittest.mock import patch
import pandas as pd
import data_sources

class TestSpatialFetch(unittest.TestCase):
    def test_deduplicates_grid_cells_and_selects_hour(self):
        def fake_api(url, params):
            dates=['2026-11-01T10:00','2026-11-01T11:00']
            return [{'latitude': 31.4+i//2*0.4,'longitude':74.6+i%2*0.4,
                     'hourly':{'time':dates,'pm2_5':[80,100+i], 'us_aqi':[110,155+i]}}
                    for i in range(9)]
        with patch.object(data_sources, '_request', side_effect=fake_api):
            frame = data_sources.get_spatial_forecast_grid(31.6,74.8,pd.Timestamp('2026-11-01T11:00'),side=3)
        self.assertEqual(len(frame),9)
        self.assertIn('us_aqi',frame)
        self.assertEqual(frame.iloc[0].us_aqi,155)

class TestMultiLocationRequest(unittest.TestCase):
    def test_multi_coordinate_payload_is_accepted(self):
        from unittest.mock import Mock
        response = Mock()
        response.json.return_value = [{'hourly': {'time':[]}}, {'hourly': {'time':[]}}]
        response.raise_for_status.return_value = None
        with patch.object(data_sources.requests,'get',return_value=response):
            result = data_sources._request('https://example.test',{})
        self.assertEqual(len(result),2)
