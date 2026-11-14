"""Leakage invariants; synthetic data only, never accesses held-out outcomes."""
import unittest
import pandas as pd
from spatial_experiments import make_features, predict_bundle, BASELINE


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.refs = pd.DataFrame(dict(
            date=['2024-01-01'] * 5,
            station_id=['target', 'possible_alias', 'a', 'b', 'c'],
            lat=[43.24, 43.24, 43.241, 43.242, 43.243],
            lon=[76.9] * 5, pm25=[100., 101., 10., 20., 30.]))
        self.target = self.refs.iloc[:1].copy()

    def test_target_and_alias_values_cannot_change_features(self):
        before, _ = make_features(self.target, self.refs, exclude_colocated=True)
        self.target.loc[0, 'pm25'] = 999.
        self.refs.loc[:1, 'pm25'] = 888.
        after, _ = make_features(self.target, self.refs, exclude_colocated=True)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.near1_pm25.iloc[0], 10.)
        self.assertEqual(before.reference_count.iloc[0], 3)

    def test_other_dates_cannot_change_features(self):
        before, _ = make_features(self.target, self.refs, exclude_colocated=True)
        extra = self.refs.copy()
        extra.date = '2024-01-02'
        extra.pm25 = 900.
        after, _ = make_features(self.target, pd.concat([self.refs, extra]), exclude_colocated=True)
        pd.testing.assert_frame_equal(before, after)

    def test_idw_inference(self):
        x, _ = make_features(self.target, self.refs, exclude_colocated=True)
        actual = predict_bundle(dict(family='idw', idw_column=BASELINE), x)
        self.assertEqual(actual[0], x[BASELINE].iloc[0])
        self.assertTrue(10 <= actual[0] <= 30)


if __name__ == '__main__':
    unittest.main()
