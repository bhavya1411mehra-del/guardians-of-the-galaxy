import unittest
import pandas as pd
from heatmap import color_for_value, interpolate_surface, demo_grid

class TestHeatmap(unittest.TestCase):
    def test_higher_values_are_darker_red(self):
        low = color_for_value(90, 'us_aqi')
        high = color_for_value(340, 'us_aqi')
        self.assertLess(sum(high[:3]), sum(low[:3]))
        self.assertGreater(high[0], high[1])

    def test_dense_spatial_values_make_polygons(self):
        pts = demo_grid(31.6, 74.8, 110, 3)
        surface = interpolate_surface(pts, 'pm2_5', steps=18)
        self.assertGreater(len(surface), 100)
        self.assertTrue((surface['value'] >= 0).all())
        self.assertGreater(surface['value'].max(), surface['value'].min())
        self.assertEqual(len(surface.iloc[0]['polygon']), 4)

    def test_one_station_cannot_fabricate_heatmap(self):
        sample = pd.DataFrame([{'lat': 31.6, 'lon':74.8, 'pm2_5':150}])
        self.assertTrue(interpolate_surface(sample,'pm2_5').empty)

    def test_invalid_points_removed(self):
        sample = demo_grid(31.6,74.8,90,4)
        sample.loc[0,'pm2_5'] = float('nan')
        surface = interpolate_surface(sample,'pm2_5')
        self.assertFalse(surface.empty)
