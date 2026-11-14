"""Gradio front-end. Enter real daily data to produce a next-day prediction."""
import json
from pathlib import Path
import pandas as pd
import gradio as gr
from airguard import DATA, predict_from_inputs, read_daily
from station_map import load_map
from spatial_ai import latest_map, results_summary

RESULTS = DATA/'results.json'


def forecasting(pm25, yesterday, temp, wind, rain, humidity, observation_date):
    try:
        d = pd.Timestamp(observation_date)
        nextday = (d + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        vals = [pm25, yesterday, temp, wind, rain, humidity]
        if any(v is None for v in vals) or pm25 < 0 or yesterday < 0 or wind < 0 or rain < 0 or not 0 <= humidity <= 100:
            return 'Enter valid readings (PM2.5/wind/rain >= 0 and humidity 0–100%).'
        value = predict_from_inputs(*vals, forecast_date=observation_date)
        msg = ('Above 35 µg/m³: consider reviewing outdoor activity plans.' if value > 35 else
               'At or below 35 µg/m³: still check local official air-quality guidance.')
        return (f'### Forecast for {nextday}: {value:.1f} µg/m³ PM2.5\n\n'
                f'{msg}\n\n**Prototype warning:** this is a statistical estimate, not live measured air quality, '
                'medical guidance, or a validated health alert. The 35 µg/m³ cutoff is a demonstration threshold.')
    except Exception as e:
        return f'Unable to predict: {e}'


def latest_defaults():
    try:
        d = read_daily().dropna(subset=['pm25','temperature','wind','rain','humidity'])
        a, b = d.iloc[-1], d.iloc[-2]
        return [round(float(a.pm25),1), round(float(b.pm25),1), round(float(a.temperature),1),
                round(float(a.wind),1), round(float(a.rain),1), round(float(a.humidity),1),
                a.date.strftime('%Y-%m-%d')]
    except Exception:
        return [None,None,None,None,None,None,'2026-11-13']


def results_text():
    if not RESULTS.exists():
        return 'Train the model first to see real held-out test results.'
    r = json.loads(RESULTS.read_text())
    return (f"**Test dates:** {r['test_start']} to {r['test_end']} ({r['test_count']} days)\n\n"
            f"**Persistence baseline MAE:** {r['scores']['persistence_baseline']:.2f} µg/m³\n\n"
            f"**Linear regression MAE:** {r['scores']['linear_regression']:.2f} µg/m³\n\n"
            f"**Random forest MAE:** {r['scores']['random_forest']:.2f} µg/m³\n\n"
            f"**Selected model:** {r['selected_model']}\n\n"
            f"**Beats persistence baseline:** {'Yes' if r['beats_baseline'] else 'No'}")


def preview():
    path = DATA/'test_predictions.csv'
    if path.exists():
        return pd.read_csv(path).tail(30)
    return pd.DataFrame()

with gr.Blocks(title='Almaty AirGuard') as demo:
    gr.Markdown('# 🌫️ Almaty AirGuard\nPredict **next-day PM2.5** from today’s pollution and weather. Built for the Veritas AI hackathon.\n\n**Data note:** training PM2.5 comes from modelled CAMS/Open-Meteo estimates, **not** a local measuring station.')
    with gr.Tab('Try the AI'):
        gr.Markdown('Numbers below are loaded from the latest available historical day, **not automatically from today**. Replace them with today’s observed values to forecast tomorrow.')
        defaults = latest_defaults()
        with gr.Row():
            today = gr.Number(label='Today PM2.5 (µg/m³)', value=defaults[0])
            prev = gr.Number(label='Yesterday PM2.5 (µg/m³)', value=defaults[1])
            temp = gr.Number(label='Today temperature (°C)', value=defaults[2])
        with gr.Row():
            wind = gr.Number(label='Today wind (km/h)', value=defaults[3])
            rain = gr.Number(label='Today precipitation (mm)', value=defaults[4])
            humid = gr.Number(label='Today humidity (%)', value=defaults[5])
        day = gr.Textbox(label='Date of today’s input readings (YYYY-MM-DD)', value=defaults[6])
        btn = gr.Button('Predict next day', variant='primary')
        out = gr.Markdown()
        btn.click(forecasting, [today,prev,temp,wind,rain,humid,day], out)
    with gr.Tab('Pollution Map — measured stations'):
        gr.Markdown('### Actual PM2.5 readings across Almaty\nData: [AirData.kz](https://github.com/qazybekb/AirDatakz-OpenData). These are monitoring-station averages, **not neighborhood-wide AI predictions**.\n\nDownload the data once using `python station_map.py download` before loading the map. An internet connection is needed for OpenStreetMap tiles.')
        load = gr.Button('Load latest station map', variant='primary')
        summary = gr.Markdown()
        plot = gr.Plot(label='Almaty station map')
        station_table = gr.Dataframe(label='Stations ordered by PM2.5', interactive=False)
        def safe_map():
            try:
                return load_map()
            except Exception as exc:
                return f'Unable to load map: {exc}', None, pd.DataFrame()
        load.click(safe_map, outputs=[summary,plot,station_table])
    with gr.Tab('AI Spatial Map — estimated areas'):
        gr.Markdown('### AI estimates between sensors — separate from measured locations\n'
                    'Train the spatial model first. The map reports **same-day spatial estimates**, '
                    '**not tomorrow\'s neighborhood forecasts**. It displays the AI correction only when '
                    'it beats the inverse-distance baseline on held-out stations.')
        spatial_btn = gr.Button('Load spatial estimates',variant='primary')
        spatial_description = gr.Markdown()
        spatial_plot = gr.Plot(label='Estimated PM2.5 map')
        def safe_spatial():
            try:
                return latest_map()
            except Exception as exc:
                return f'Unable to show spatial map: {exc}', None
        spatial_btn.click(safe_spatial,outputs=[spatial_description,spatial_plot])
        gr.Markdown('### Honest model evaluation')
        evaluation_btn = gr.Button('Show held-out station results')
        evaluation_text = gr.Markdown()
        evaluation_table = gr.Dataframe(label='Largest AI mistakes on unseen stations', interactive=False)
        evaluation_btn.click(results_summary, outputs=[evaluation_text,evaluation_table])
    with gr.Tab('Model evaluation'):
        gr.Markdown(results_text())
        gr.Markdown('Last 30 unseen test cases (compare predicted and actual values):')
        gr.Dataframe(value=preview(), interactive=False)
    gr.Markdown('Sources: [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) and [Historical Weather](https://open-meteo.com/en/docs/historical-weather-api). Prototype for research/education, not operational air-quality advice.')

if __name__ == '__main__':
    demo.launch()
