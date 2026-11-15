"""Focused presentation regression checks; uses local artifacts, never trains."""
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import gradio as gr

import app
import pollution_time_machine as time_machine


class DashboardChecks(unittest.TestCase):
    def test_overview_calendar_and_gaps(self):
        _, chart, _, status = app.load_overview()
        self.assertIn('Historical observations', status)
        self.assertEqual(chart.layout.xaxis.type, 'date')
        dates = pd.to_datetime(list(chart.data[-1].x))
        self.assertEqual(len(dates), 30)
        self.assertTrue((dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all())
        self.assertFalse(chart.data[-1].connectgaps)
        wire = json.loads(gr.Plot().postprocess(chart).plot)
        self.assertEqual(wire['layout']['xaxis']['range'], [dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')])

    def test_explorer_modes_and_basemaps(self):
        for mode in ['Measured stations', 'Estimated heatmap']:
            views = []
            for basemap in ['Dark', 'Street']:
                summary, chart, table = app.load_explorer(mode, basemap, True)
                self.assertNotIn('unavailable', summary.lower())
                style = chart.layout.map.style
                self.assertEqual(style['sources']['basemap']['type'], 'raster')
                self.assertTrue(style['sources']['basemap']['tiles'][0].startswith('https://'))
                measured = [t for t in chart.data if t.name in ('Measured station averages', 'Measured reference stations')]
                self.assertEqual(len(measured), 1)
                self.assertEqual(measured[0].mode, 'markers+text')
                self.assertEqual(measured[0].selected.marker.size, 16)
                self.assertEqual(len(measured[0].customdata), len(measured[0].lat))
                if mode == 'Estimated heatmap':
                    self.assertEqual(style['sources']['pollution-0']['type'], 'image')
                    self.assertEqual(len(chart.layout.map.layers), 0)
                gr.Plot().postprocess(chart)
                views.append(chart)
            for first, second in zip(views[0].data, views[1].data):
                np.testing.assert_array_equal(first.lat, second.lat)
                np.testing.assert_array_equal(first.lon, second.lon)
                np.testing.assert_array_equal(first.marker.color, second.marker.color)

    def test_time_machine_dates_and_bounds(self):
        _, dates = time_machine.dataset()
        for index, expected in [(-1, dates[0]), (1, dates[1]), (10**6, dates[-1])]:
            summary, chart, table = time_machine.render_day(index)
            self.assertIn(expected, summary)
            self.assertGreater(len(table), 0)
            self.assertEqual(chart.layout.map.style['sources']['basemap']['type'], 'raster')
        tab_callbacks = [f for f in app.demo.fns.values() if f.fn is time_machine.render_day]
        self.assertTrue(tab_callbacks)
        self.assertTrue(all(app.time_slider in f.inputs for f in tab_callbacks))

    def test_forecast_and_validation(self):
        card, chart = app.forecast_dashboard(*app.latest_defaults())
        self.assertIn('forecast-ready', card)
        self.assertEqual(chart.layout.xaxis.type, 'date')
        card, _ = app.forecast_dashboard(-1, 10, 5, 2, 0, 50, '2026-11-13')
        self.assertIn('role="alert"', card)
        text, chart, table = app.grouped_validation_panel()
        self.assertEqual(len(table), 7)
        self.assertIn('ML did not outperform', text)
        source = pd.read_csv(app.Path(app.__file__).parent / 'experiments/grouped_validation/comparison.csv').sort_values('mae')
        np.testing.assert_allclose(chart.data[0].x, source.mae)
        np.testing.assert_allclose(chart.data[1].x, source.rmse)
        for metric in ['MAE', 'RMSE']:
            cards, fig, _, _ = app.evaluation_dashboard(metric)
            self.assertNotIn('unavailable', cards)
            gr.Plot().postprocess(fig)

    def test_map_error_placeholder(self):
        with patch.object(app, 'measured_day', side_effect=ValueError('No local readings')):
            summary, fig, table = app.safe_map()
        self.assertIn('unavailable', summary)
        self.assertIsNotNone(fig)
        self.assertTrue(table.empty)


if __name__ == '__main__':
    unittest.main()
