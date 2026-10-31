import sys
import unittest
from pathlib import Path
from io import StringIO
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_lab import validate_pollution_csv, full_training_csv, find_pm25_sensors
from health_evidence import guidance, pm25_24h_mean, WHO_GUIDELINE

class ImportTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.date_range('2025-10-01', periods=150, freq='h')
        self.data = pd.DataFrame({'timestamp': self.times.strftime('%Y-%m-%d %H:%M:%S'),
                                 'pm25': [90.0 + i/10 for i in range(150)]})

    def test_aliases_and_order(self):
        raw = validate_pollution_csv(StringIO(self.data.iloc[::-1].to_csv(index=False)))
        self.assertEqual(list(raw.columns), ['time', 'pm2_5'])
        self.assertEqual(len(raw), 150)
        self.assertTrue(raw['time'].is_monotonic_increasing)

    def test_missing_weather_is_clear(self):
        raw = validate_pollution_csv(StringIO(self.data.to_csv(index=False)))
        with self.assertRaisesRegex(ValueError, 'Missing weather'):
            full_training_csv(raw)

    def test_complete_weather(self):
        raw = validate_pollution_csv(StringIO(self.data.to_csv(index=False)))
        for name, value in [('temperature_2m', 30),('relative_humidity_2m', 60),
                            ('wind_speed_10m', 6), ('precipitation', 0)]:
            raw[name] = value
        clean = full_training_csv(raw)
        self.assertEqual(len(clean), 150)

    def test_no_aqi_conversion(self):
        wrong = self.data.rename(columns={'pm25':'aqi'})
        with self.assertRaisesRegex(ValueError, 'AQI alone'):
            validate_pollution_csv(StringIO(wrong.to_csv(index=False)))

    def test_health_sources(self):
        title, notes, links = guidance('Children & teenagers')
        self.assertGreaterEqual(len(notes), 2)
        self.assertTrue(all(url.startswith('https://') for url in links))
        self.assertIn('NBK574582', WHO_GUIDELINE)

    def test_24h_is_actual_24h(self):
        times = pd.date_range('2025-10-01', periods=24, freq='h')
        frame = pd.DataFrame({'time': times, 'pm2_5':[40]*24})
        self.assertEqual(pm25_24h_mean(frame),40)
        frame.loc[10, 'time'] += pd.Timedelta(hours=1)
        self.assertIsNone(pm25_24h_mean(frame))

if __name__ == '__main__':
    unittest.main()
