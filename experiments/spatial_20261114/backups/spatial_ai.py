"""AirGuard spatial PM2.5: same-day interpolation, with held-out-station testing.

IMPORTANT: Results are spatial estimates conditioned on same-day stations, not
next-day forecasts and not ground truth at unmonitored locations.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import plotly.graph_objects as go
from station_map import daily_stations

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
MODEL_FILE = DATA / 'spatial_model.joblib'
RESULT_FILE = DATA / 'spatial_results.json'
TEST_FILE = DATA / 'spatial_test_predictions.csv'
EARTH_KM = 6371.0
COLS = ['lat','lon','month_sin','month_cos','idw','idw4','nearest_km','neighbors_3km','neighbors_5km','reference_mean','spread']


def haversine(lat, lon, refs):
    """Distances from one location to lat/lon of n reference stations (km)."""
    a1, o1 = np.radians(float(lat)), np.radians(float(lon))
    a2, o2 = np.radians(refs[:, 0]), np.radians(refs[:, 1])
    t = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
    return 2*EARTH_KM*np.arcsin(np.sqrt(np.clip(t, 0, 1)))


def feature_row(lat, lon, date, references, skip_id=None):
    """References have columns: station_id, lat, lon, pm25. Never includes target."""
    if skip_id is not None:
        references = references[references.station_id != skip_id]
    if len(references) < 3:
        return None
    distances = haversine(lat, lon, references[['lat','lon']].to_numpy(dtype=float))
    v = references.pm25.to_numpy(dtype=float)
    order = np.argsort(distances)
    distances, v = distances[order], v[order]
    k = min(8, len(v))
    d, nearby = distances[:k], v[:k]
    w = 1 / np.maximum(d, 0.15)**2
    idw = float(np.sum(w*nearby)/np.sum(w))
    d4, v4 = distances[:min(4,len(v))], v[:min(4,len(v))]
    w4 = 1 / np.maximum(d4, 0.15)**2
    dt = pd.Timestamp(date)
    month = 2*np.pi*dt.dayofyear / 365.25
    return dict(lat=float(lat), lon=float(lon), month_sin=float(np.sin(month)),
                month_cos=float(np.cos(month)), idw=idw,
                idw4=float(np.sum(w4*v4)/np.sum(w4)),
                nearest_km=float(distances[0]), neighbors_3km=int((distances<=3).sum()),
                neighbors_5km=int((distances<=5).sum()),
                reference_mean=float(np.mean(nearby)), spread=float(np.std(nearby)))


def feature_table(targets, reference_by_date, training):
    rows, actual, dates, sites = [], [], [], []
    for date, day in targets.groupby('date', sort=True):
        refs = reference_by_date.get(date)
        if refs is None or len(refs)<3:
            continue
        for r in day.itertuples(index=False):
            feat = feature_row(r.lat, r.lon, date, refs, skip_id=r.station_id if training else None)
            if feat is None or feat['nearest_km'] > 5:
                continue # No extrapolating into large unmonitored areas
            rows.append(feat); actual.append(float(r.pm25)); dates.append(date); sites.append(str(r.station_id))
    if not rows:
        raise ValueError('No eligible station observations after spatial coverage filtering.')
    x = pd.DataFrame(rows)[COLS]
    return x, np.asarray(actual), dates, sites


def train():
    df = daily_stations().copy()
    days = np.sort(df.date.unique())
    if len(days) < 35:
        raise ValueError('Need at least 35 dates of station data for a meaningful time split.')
    # NEVER train on test station IDs, including during earlier dates.
    site_counts = df.groupby('station_id').date.nunique()
    eligible = sorted(site_counts[site_counts >= 25].index.astype(str))
    if len(eligible) < 12:
        raise ValueError('Need at least 12 stations with 25+ dates of observations.')
    train_ids, heldout_ids = train_test_split(eligible, test_size=0.2, random_state=42)
    train_ids, heldout_ids = set(train_ids), set(heldout_ids)
    df['station_id'] = df.station_id.astype(str)
    # The last 20% of dates are unseen in the model fit.
    cutoff = days[int(len(days)*.8)]
    reference = df[df.station_id.isin(train_ids)]
    reference_by_date = {day: group for day, group in reference.groupby('date')}
    fit_targets = reference[reference.date < cutoff]
    test_targets = df[(df.station_id.isin(heldout_ids)) & (df.date >= cutoff)]
    if test_targets.empty:
        raise ValueError('No held-out sites report during the test dates; try a different split seed.')
    print(f'Train dates before {cutoff}; test dates from {cutoff}; {len(train_ids)} reference stations, {len(heldout_ids)} held-out stations', flush=True)
    x_train, y_train, _, _ = feature_table(fit_targets, reference_by_date, training=True)
    x_test, y_test, test_dates, test_sites = feature_table(test_targets, reference_by_date, training=False)
    if len(x_test) < 30:
        raise ValueError('Too few held-out station-days for a reliable evaluation.')
    # Residual ML: learn corrections to the explicitly stated spatial baseline.
    model = RandomForestRegressor(n_estimators=90, min_samples_leaf=8,
                                  max_depth=12, max_features=.85,
                                  random_state=42, n_jobs=-1)
    residual = y_train - x_train['idw'].to_numpy()
    model.fit(x_train, residual)
    baseline = x_test['idw'].to_numpy()
    ml = np.maximum(0, baseline + model.predict(x_test))
    baseline_mae = float(mean_absolute_error(y_test, baseline))
    ml_mae = float(mean_absolute_error(y_test, ml))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))
    ml_rmse = float(np.sqrt(mean_squared_error(y_test, ml)))
    use_ml = ml_mae < baseline_mae
    result = {
        'evaluation':'Held-out stations AND later dates; same-day spatial interpolation',
        'train_until_exclusive':str(cutoff),'test_from':str(cutoff),
        'training_station_count':len(train_ids),'heldout_station_count':len(heldout_ids),
        'training_examples':len(x_train),'test_examples':len(x_test),
        'test_unique_sites':len(set(test_sites)),
        'baseline_idw_mae':round(baseline_mae,3),'ml_corrected_mae':round(ml_mae,3),
        'baseline_idw_rmse':round(baseline_rmse,3),'ml_corrected_rmse':round(ml_rmse,3),
        'improvement_pct':round(100*(baseline_mae-ml_mae)/baseline_mae,2) if baseline_mae else 0,
        'recommended_visualization':'ML correction' if use_ml else 'IDW baseline (ML did not beat it)',
        'source':'AirData.kz; low-cost station PM2.5 readings; 12+ valid hourly samples per station-day',
        'limitations':'Same-day estimates rely on measurements from reference stations; not tomorrow forecasts. Stations have differing accuracy. Spatial errors can be highly correlated. Split uses seeded station holdout and temporal holdout.'
    }
    DATA.mkdir(exist_ok=True,parents=True)
    joblib.dump({'model':model,'reference_site_ids':sorted(train_ids),'heldout_site_ids':sorted(heldout_ids),
                 'use_ml':use_ml,'features':COLS},MODEL_FILE)
    RESULT_FILE.write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame({'date':test_dates,'station_id':test_sites,'actual':y_test,
                  'idw_baseline':baseline,'ai_prediction':ml,
                  'baseline_abs_error':np.abs(y_test-baseline),
                  'ai_abs_error':np.abs(y_test-ml)}).to_csv(TEST_FILE,index=False)
    print(json.dumps(result,indent=2),flush=True)
    print('Saved spatial model, scores and held-out predictions under data/',flush=True)
    return result


def latest_map():
    if not MODEL_FILE.exists():
        raise FileNotFoundError('Train the spatial model first: .\\.venv\\Scripts\\python.exe spatial_ai.py train')
    df = daily_stations()
    artifact = joblib.load(MODEL_FILE)
    daily_count = df.groupby('date').station_id.nunique()
    valid_dates = daily_count[daily_count >= 15]
    if valid_dates.empty:
        raise ValueError('No dates with >=15 stations.')
    day = valid_dates.index[-1]
    measured = df[df.date == day]
    refs = measured[measured.station_id.astype(str).isin(artifact['reference_site_ids'])].copy()
    if len(refs)<5:
        raise ValueError('Too few reporting reference stations on latest eligible date.')
    pts = refs[['lat','lon']].to_numpy(dtype=float)
    if len(pts)<3:
        raise ValueError('Need three stations to define interpolated area.')
    # Convex hull + 3 km radius restriction: avoid claiming predictions far from monitors.
    tri = Delaunay(pts)
    lat_vals = np.linspace(float(refs.lat.min()),float(refs.lat.max()),48)
    lon_vals = np.linspace(float(refs.lon.min()),float(refs.lon.max()),50)
    candidates = np.array([(la,lo) for la in lat_vals for lo in lon_vals])
    candidates = candidates[tri.find_simplex(candidates) >= 0]
    features, lats, lons = [], [], []
    for la,lo in candidates:
        feat = feature_row(la,lo,day,refs)
        if feat and feat['nearest_km'] <= 3:
            features.append(feat);lats.append(la);lons.append(lo)
    if not features:
        raise ValueError('No interpolated points inside the monitored region.')
    x = pd.DataFrame(features)[COLS]
    idw = x.idw.to_numpy()
    values = np.maximum(0,idw + artifact['model'].predict(x)) if artifact['use_ml'] else idw
    fig = go.Figure()
    color_max = max(50,float(np.percentile(np.r_[values, refs.pm25.to_numpy()],98)))
    fig.add_trace(go.Scattermap(lat=lats,lon=lons,mode='markers',
        marker={'size':18,'opacity':.65,'color':values,'colorscale':'RdYlGn_r',
                'cmin':0,'cmax':color_max,'showscale':True,'colorbar':{'title':'PM2.5'}},
        text=[f'Estimated PM2.5: {x:.1f} µg/m³' for x in values],hoverinfo='text',name='Estimated, not measured'))
    fig.add_trace(go.Scattermap(lat=refs.lat,lon=refs.lon,mode='markers',
        marker={'size':7,'color':'#14263A'},
        text=[f'Reference station: {name}<br>Measured PM2.5: {v:.1f} µg/m³' for name,v in zip(refs.station_name,refs.pm25)],
        hoverinfo='text',name='Ground-station measurements'))
    fig.update_layout(map={'style':'open-street-map','center':{'lat':43.2389,'lon':76.8897},'zoom':10},
                      height=550,margin={'l':0,'r':0,'b':0,'t':10},legend={'orientation':'h','y':0})
    method = 'AI-corrected spatial interpolation' if artifact['use_ml'] else 'Distance-weighted interpolation (AI was worse on held-out sites)'
    summary = (f'### Estimated PM2.5 across monitored Almaty areas — {day}\n'
               f'**Displayed method:** {method}. **{len(refs)}** reference stations. '
               f'Colored overlay represents **estimates**, black dots represent **measured reference stations**. '
               'Only estimates within the station convex hull and 3 km of a reference sensor are shown. '
               '**Not** a next-day forecast or a validated street-level map. Low-cost sensors have measurement uncertainty.')
    return summary,fig


def results_summary():
    if not RESULT_FILE.exists():
        return 'Train first: `.\\.venv\\Scripts\\python.exe spatial_ai.py train`',pd.DataFrame()
    r = json.loads(RESULT_FILE.read_text(encoding='utf-8'))
    summary=(f"### Evaluation on unseen stations and later dates\n"
             f"**{r['test_examples']} test station-days** from **{r['test_unique_sites']} held-out stations**, "
             f"starting {r['test_from']}. Training used {r['training_examples']} earlier examples.\n\n"
             f"**Inverse-distance baseline MAE:** {r['baseline_idw_mae']} µg/m³\n\n"
             f"**AI-corrected MAE:** {r['ml_corrected_mae']} µg/m³\n\n"
             f"**Improvement:** {r['improvement_pct']}% (negative = AI worse)\n\n"
             f"**Displayed method:** {r['recommended_visualization']}\n\n"
             'The reference observations come from other reporting stations on the same date, never from the hidden station. '
             'This is a spatial interpolation evaluation, not next-day prediction. '
             'Limitations: differences in low-cost sensor accuracy, uneven station placement, and time correlation.')
    p = pd.read_csv(TEST_FILE).sort_values('ai_abs_error',ascending=False).head(12) if TEST_FILE.exists() else pd.DataFrame()
    return summary,p


if __name__=='__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('action',choices=['train','check'])
    args = cli.parse_args()
    if args.action=='train':train()
    else:print(results_summary()[0])
