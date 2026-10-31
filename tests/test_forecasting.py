import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sample_data import demo_city, fake_future_weather
from forecasting import forecast, make_training_rows, train_model


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = demo_city("Amritsar")
        cls.future = fake_future_weather(cls.history)

    def test_training_no_gap_and_targets(self):
        rows = make_training_rows(self.history)
        self.assertGreater(len(rows), 450)
        self.assertTrue(rows["horizon"].isin(list(range(1, 7))).all())
        self.assertFalse(rows["target"].isna().any())

    def test_horizon_predictions(self):
        model, stats = train_model(self.history)
        output = forecast(self.history, self.future, model)
        self.assertEqual(list(output["horizon"]), list(range(1, 7)))
        self.assertTrue((output["pm2_5"] >= 0).all())
        self.assertGreater(stats["test_examples"], 80)


if __name__ == "__main__":
    unittest.main()
