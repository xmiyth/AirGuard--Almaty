"""Ground-observation map for Almaty AirGuard.

This shows monitoring-station measurements, NOT citywide continuous AI predictions.
Source: https://github.com/qazybekb/AirDatakz-OpenData
"""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
STATION_FILE = DATA / 'almaty_stations_pm25.csv.gz'
URL = 'https://raw.githubusercontent.com/qazybekb/AirDatakz-OpenData/main/almaty/pm25.csv.gz'
REQUIRED = {'datetime_utc', 'station_id', 'station_name', 'lat', 'lon', 'value_ugm3'}


def download_station_data():
    DATA.mkdir(exist_ok=True, parents=True)
    response = requests.get(URL, timeout=120, stream=True)
    response.raise_for_status()
    tmp = STATION_FILE.with_name(STATION_FILE.name + '.part')
    try:
        with tmp.open('wb') as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(STATION_FILE)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Saved measurements to {STATION_FILE}')


def daily_stations():
    if not STATION_FILE.exists():
        raise FileNotFoundError('Station data missing. Run: .\\.venv\\Scripts\\python.exe station_map.py download')
    # Only the columns needed for the map, keeping memory use manageable.
    df = pd.read_csv(STATION_FILE, usecols=lambda c: c in REQUIRED)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f'Station data columns missing: {sorted(missing)}')
    # Data timestamps are UTC. Group by local Almaty date.
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True, errors='coerce')
    df['date'] = df['datetime_utc'].dt.tz_convert('Asia/Almaty').dt.strftime('%Y-%m-%d')
    for col in ('lat', 'lon', 'value_ugm3'):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['date', 'station_id', 'lat', 'lon', 'value_ugm3'])
    df = df[(df.lat.between(42.9, 43.5)) & (df.lon.between(76.5, 77.3)) & (df.value_ugm3 >= 0) & (df.value_ugm3 < 1000)]
    # Reject partial 1-2 hour reports; >=12 hours per station per local day.
    result = (df.groupby(['date', 'station_id'], as_index=False)
              .agg(station_name=('station_name','first'), lat=('lat','median'), lon=('lon','median'),
                   pm25=('value_ugm3','mean'), hours=('datetime_utc','nunique')))
    result = result[result.hours >= 12].copy()
    return result.sort_values(['date','station_id'])


def map_figure(df):
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title='No stations with sufficient readings for this date')
        return fig
    hover = [f"{name}<br>PM2.5: {pm:.1f} µg/m³<br>Hours: {int(n)}" for name,pm,n in zip(df.station_name,df.pm25,df.hours)]
    fig.add_trace(go.Scattermap(
        lat=df.lat, lon=df.lon, mode='markers', text=hover, hoverinfo='text',
        marker={'size': 13, 'color': df.pm25, 'colorscale': 'RdYlGn_r', 'cmin': 0,
                'cmax': max(50, float(df.pm25.quantile(.95))), 'showscale': True,
                'colorbar': {'title': 'PM2.5'}}, name='Measured stations'))
    fig.update_layout(
        map={'style': 'open-street-map', 'center': {'lat': 43.2389, 'lon': 76.8897}, 'zoom': 10},
        margin={'l': 0, 'r': 0, 'b': 0, 't': 10}, height=520, showlegend=False)
    return fig


def load_map(default_date=None):
    df = daily_stations()
    if df.empty:
        return 'No station-days with 12+ hourly observations.', go.Figure(), pd.DataFrame()
    dates = df.groupby('date').station_id.nunique()
    # Avoid showing an incomplete latest date by default. Prefer newest day with >=3 reporting stations.
    good = dates[dates >= 3]
    selected = str(default_date) if default_date in set(df.date) else (good.index[-1] if len(good) else dates.index[-1])
    shown = df[df.date == selected].sort_values('pm25', ascending=False).copy()
    summary = (f'### Station observations: {selected}\n'
               f'**{len(shown)} monitoring locations** with at least 12 hourly observations each. '
               f'Minimum: **{shown.pm25.min():.1f}**, maximum: **{shown.pm25.max():.1f}** µg/m³.\n\n'
               'These are *measured station averages*, not predictions for every street or neighborhood. '
               'Low-cost sensors may differ in accuracy. Colors are relative PM2.5 concentrations, not an official AQI.')
    table = shown[['station_name', 'pm25', 'hours', 'lat', 'lon']].head(30).round({'pm25':1,'lat':4,'lon':4})
    return summary, map_figure(shown), table


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['download','check'])
    args = parser.parse_args()
    if args.action == 'download':
        download_station_data()
    else:
        df = daily_stations()
        print(f'Loaded {len(df):,} valid station-days, {df.station_id.nunique()} stations, {df.date.min()} to {df.date.max()}')
        print(df.tail(3).to_string(index=False))
