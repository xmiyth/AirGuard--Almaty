"""Almaty AirGuard: reproducible one-day PM2.5 forecasting experiment.
Data are modelled Open-Meteo/CAMS values, not ground station observations.
"""
from __future__ import annotations
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
LAT, LON = 43.2389, 76.8897
TZ = 'Asia/Almaty'
FEATURES = ['pm25','pm25_yesterday','temperature','wind','rain','humidity','month_sin','month_cos']


def api_json(url, params):
    try:
        r = requests.get(url, params=params, timeout=90)
        r.raise_for_status()
        result = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(f'Data download failed: {exc}. Try again or use an existing data/daily.csv.') from exc
    if result.get('error'):
        raise RuntimeError('API error: ' + str(result.get('reason')))
    return result


def hourly_to_daily(obj, name):
    raw = obj.get('hourly', {})
    if 'time' not in raw or name not in raw:
        raise RuntimeError(f'Missing {name} hourly readings from API')
    x = pd.DataFrame({'time': pd.to_datetime(raw['time']), name: raw[name]})
    x[name] = pd.to_numeric(x[name], errors='coerce')
    x['date'] = x['time'].dt.normalize()
    return x.groupby('date', as_index=False)[name].mean()


def download(start='2023-01-01', end=None):
    # Avoid potentially incomplete recent daily data; user can supply newer values in the UI.
    end = end or (date.today() - timedelta(days=7)).isoformat()
    base = {'latitude': LAT, 'longitude': LON, 'start_date': start, 'end_date': end, 'timezone': TZ}
    air = api_json('https://air-quality-api.open-meteo.com/v1/air-quality',
                   {**base, 'hourly': 'pm2_5', 'domains': 'cams_global'})
    weather = api_json('https://archive-api.open-meteo.com/v1/archive',
                       {**base, 'daily': 'temperature_2m_mean,wind_speed_10m_mean,precipitation_sum,relative_humidity_2m_mean'})
    a = hourly_to_daily(air, 'pm2_5').rename(columns={'pm2_5': 'pm25'})
    d = weather.get('daily', {})
    w = pd.DataFrame({'date': pd.to_datetime(d['time']),
                      'temperature': d['temperature_2m_mean'],
                      'wind': d['wind_speed_10m_mean'],
                      'rain': d['precipitation_sum'],
                      'humidity': d['relative_humidity_2m_mean']})
    daily = pd.merge(a, w, on='date', how='inner').sort_values('date')
    if len(daily) < 300:
        raise RuntimeError(f'Only {len(daily)} daily rows received; need >=300. Try a larger date range.')
    DATA.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DATA/'daily.csv', index=False)
    print(f'Saved {len(daily)} rows ({daily.date.min().date()} to {daily.date.max().date()}) to data/daily.csv')
    print('IMPORTANT: PM2.5 values are CAMS model estimates, not ground measurements.')


def read_daily(path=None):
    path = Path(path) if path else DATA/'daily.csv'
    if not path.exists():
        raise FileNotFoundError('No data/daily.csv found. First run: python airguard.py download')
    df = pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    columns = ['pm25','temperature','wind','rain','humidity']
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.drop_duplicates(subset=['date'])
    return df


def feature_frame(df):
    df = df.copy().sort_values('date').reset_index(drop=True)
    df['pm25_yesterday'] = df['pm25'].shift(1)
    df['month_sin'] = np.sin(2*np.pi*df['date'].dt.dayofyear/365.25)
    df['month_cos'] = np.cos(2*np.pi*df['date'].dt.dayofyear/365.25)
    # Don't build targets across gaps; next calendar day only.
    df['target'] = df['pm25'].shift(-1)
    next_date = df['date'].shift(-1)
    df.loc[(next_date-df['date']).dt.days.ne(1), 'target'] = np.nan
    df = df.dropna(subset=['pm25','target','pm25_yesterday']).reset_index(drop=True)
    return df


def train(path=None):
    df = feature_frame(read_daily(path))
    if len(df) < 250:
        raise RuntimeError(f'Need 250+ complete dated examples; got {len(df)}')
    boundary = int(len(df)*0.8)
    tr, te = df.iloc[:boundary], df.iloc[boundary:]
    Xtr, Xte, ytr, yte = tr[FEATURES], te[FEATURES], tr['target'], te['target']
    baseline = mean_absolute_error(yte, te['pm25'])
    models = {
        'linear_regression': make_pipeline(SimpleImputer(strategy='median'), LinearRegression()),
        'random_forest': make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(n_estimators=160, max_depth=9, min_samples_leaf=4, random_state=42, n_jobs=-1)),
    }
    scores = {'persistence_baseline': float(baseline)}
    predictions = pd.DataFrame({'date': te.date.dt.strftime('%Y-%m-%d').to_numpy(),
                                'actual_pm25': yte.to_numpy(), 'baseline_pm25': te.pm25.to_numpy()})
    for label, model in models.items():
        model.fit(Xtr, ytr)
        predictions[label] = model.predict(Xte).clip(min=0)
        scores[label] = float(mean_absolute_error(yte, predictions[label]))
    winner = min(models, key=lambda name: scores[name])
    DATA.mkdir(exist_ok=True, parents=True)
    # Keep held-out evaluation honest. The packaged demo uses only a model trained on the training period.
    joblib.dump(models[winner], DATA/'model.joblib')
    predictions.to_csv(DATA/'test_predictions.csv', index=False)
    failures = predictions.assign(abs_error=(predictions[winner]-predictions.actual_pm25).abs()).nlargest(5,'abs_error')
    result = {
        'source': 'Open-Meteo CAMS global modelled PM2.5, Open-Meteo historical weather',
        'train_start': str(tr.date.min().date()), 'train_end': str(tr.date.max().date()),
        'test_start': str(te.date.min().date()), 'test_end': str(te.date.max().date()),
        'train_count': len(tr), 'test_count': len(te), 'metric': 'MAE (µg/m³)',
        'scores': scores, 'selected_model': winner,
        'beats_baseline': bool(scores[winner] < baseline),
        'worst_errors': failures[['date','actual_pm25',winner,'abs_error']].round(2).to_dict(orient='records'),
    }
    (DATA/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def predict_from_inputs(pm25, pm25_yesterday, temperature, wind, rain, humidity, forecast_date=None):
    model_path = DATA/'model.joblib'
    if not model_path.exists():
        raise FileNotFoundError('Train model first: python airguard.py train')
    model = joblib.load(model_path)
    d = pd.to_datetime(forecast_date or date.today())
    row = pd.DataFrame([{'pm25': pm25, 'pm25_yesterday': pm25_yesterday,
        'temperature': temperature, 'wind': wind, 'rain': rain, 'humidity': humidity,
        'month_sin': np.sin(2*np.pi*d.dayofyear/365.25),
        'month_cos': np.cos(2*np.pi*d.dayofyear/365.25)}])
    return float(max(0,model.predict(row[FEATURES])[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download','train'])
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end')
    parser.add_argument('--csv', help='Use your own daily.csv with date,pm25,temperature,wind,rain,humidity')
    args = parser.parse_args()
    if args.command == 'download':
        download(args.start,args.end)
    else:
        train(args.csv)
