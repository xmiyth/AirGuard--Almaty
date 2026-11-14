"""Isolated, reproducible spatial model selection; never changes website artifacts.

Run with .venv/Scripts/python.exe -X utf8 spatial_experiments.py
Outputs and temporary files stay on D:. A completed run cannot be overwritten.
This estimates completed local-day PM2.5 using other stations on that day.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments' / 'spatial_20261114'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'tmp').mkdir(exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT / 'tmp')
os.environ['JOBLIB_TEMP_FOLDER'] = str(OUT / 'tmp')
os.environ['OMP_NUM_THREADS'] = '4'

import hashlib
import json
import platform
import sys
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from station_map import daily_stations
from spatial_ai import haversine

SEED = 20261114
HIGH = 35.0  # descriptive evaluation threshold, not an official AQI/health category
KS = [1, 3, 4, 8, 12, 20]
POWERS = [1.0, 2.0, 3.0]
FLOORS = [0.05, 0.15, 0.5]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def idw_name(k, p, floor):
    return f'idw_k{k}_p{p:g}_floor{floor:g}'


BASELINE = idw_name(8, 2, .15)


def make_features(targets, references):
    """Only other same-day reference station values; target PM2.5 is output only.

    Entire validation/test station pools are removed by callers. Training rows
    additionally leave out their own station ID, on every date.
    """
    groups = {date: g for date, g in references.groupby('date')}
    rows, labels = [], []
    for date, day in targets.groupby('date', sort=True):
        refs = groups.get(date)
        if refs is None:
            continue
        coords = refs[['lat', 'lon']].to_numpy(float)
        ids = refs.station_id.to_numpy(str)
        pm = refs.pm25.to_numpy(float)
        angle = 2 * np.pi * pd.Timestamp(date).dayofyear / 365.25
        for t in day.itertuples(index=False):
            mask = ids != str(t.station_id)
            if mask.sum() < 3:
                continue
            dist = haversine(t.lat, t.lon, coords[mask])
            order = np.argsort(dist)
            d, v = dist[order], pm[mask][order]
            if d[0] > 5:
                continue
            c = coords[mask][order]
            row = dict(lat=t.lat, lon=t.lon, month_sin=np.sin(angle),
                       month_cos=np.cos(angle), nearest_km=d[0],
                       neighbors_3km=int((d <= 3).sum()),
                       neighbors_5km=int((d <= 5).sum()), reference_count=len(v),
                       reference_mean=v[:8].mean(), spread=v[:8].std())
            for k in KS:
                for p in POWERS:
                    for floor in FLOORS:
                        w = np.maximum(d[:k], floor) ** -p
                        row[idw_name(k, p, floor)] = np.dot(w, v[:k]) / w.sum()
            for i in range(8):
                # Repeat the farthest available neighbor if fewer than eight;
                # reference_count makes the padding explicit, no learned imputation.
                j = min(i, len(v)-1)
                row.update({f'near{i+1}_pm25': v[j], f'near{i+1}_km': d[j],
                            f'near{i+1}_lat': c[j, 0], f'near{i+1}_lon': c[j, 1]})
            rows.append(row)
            labels.append(dict(date=date, station_id=str(t.station_id), actual=float(t.pm25)))
    if not rows:
        raise ValueError('Empty eligible split')
    return pd.DataFrame(rows), pd.DataFrame(labels)


def metrics(meta, pred):
    y = meta.actual.to_numpy()
    err = pred - y
    high = y >= HIGH
    # High days use mean target concentration solely for reporting after prediction.
    day_means = meta.groupby('date').actual.mean()
    high_days = meta.date.isin(day_means[day_means >= HIGH].index).to_numpy()
    return dict(n=len(y), stations=int(meta.station_id.nunique()),
                mae=float(np.abs(err).mean()), rmse=float(np.sqrt(np.mean(err**2))),
                bias=float(err.mean()), high_n=int(high.sum()),
                high_mae=float(np.abs(err[high]).mean()) if high.any() else None,
                high_rmse=float(np.sqrt(np.mean(err[high]**2))) if high.any() else None,
                high_day_n=int(high_days.sum()),
                high_day_mae=float(np.abs(err[high_days]).mean()) if high_days.any() else None,
                station_macro_mae=float(pd.Series(np.abs(err)).groupby(meta.station_id).mean().mean()))


def estimator(family, params):
    if family in ('random_forest', 'residual_rf'):
        return RandomForestRegressor(n_estimators=180, n_jobs=4, random_state=SEED, **params)
    return HistGradientBoostingRegressor(max_iter=180, learning_rate=.05,
                                        early_stopping=False, random_state=SEED, **params)


def predict_bundle(bundle, features):
    """Inference API: features come from make_features with measurements available then."""
    baseline = features[bundle['idw_column']].to_numpy()
    if bundle['family'] == 'idw':
        return baseline
    correction = bundle['model'].predict(features[bundle['features']])
    return np.maximum(0, baseline + bundle['alpha'] * correction
                      if bundle['family'].startswith('residual') else correction)


def main():
    if (OUT / 'selection.json').exists() or (OUT / 'test_results.json').exists():
        raise RuntimeError('Selection already frozen; use a new experiment directory, never retune on this test.')
    start = time.monotonic()
    paths = [ROOT / p for p in ['spatial_ai.py', 'station_map.py', 'app.py',
             'data/spatial_model.joblib', 'data/spatial_results.json',
             'data/spatial_test_predictions.csv', 'data/almaty_stations_pm25.csv.gz']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    old = joblib.load(ROOT / 'data/spatial_model.joblib')
    old_result = json.loads((ROOT / 'data/spatial_results.json').read_text())
    train_ids, test_ids = set(old['reference_site_ids']), set(old['heldout_site_ids'])
    assert len(train_ids) == 148 and len(test_ids) == 37 and not train_ids & test_ids
    cutoff = old_result['test_from']
    # Read only old evaluation keys. No saved test values/predictions used in selection.
    test_keys = pd.read_csv(ROOT / 'data/spatial_test_predictions.csv',
                            usecols=['date', 'station_id'], dtype=str)
    df = daily_stations()
    df.station_id = df.station_id.astype(str)
    pre = df[(df.date < cutoff) & df.station_id.isin(train_ids)].copy()
    active = sorted(pre.station_id.unique())
    fit_ids, val_ids = train_test_split(active, test_size=.25, random_state=SEED)
    fit_ids, val_ids = set(fit_ids), set(val_ids)
    days = sorted(pre.date.unique())
    val_from = days[int(len(days)*.75)]
    fit_refs = pre[pre.station_id.isin(fit_ids)]
    fit_targets = fit_refs[fit_refs.date < val_from]
    val_targets = pre[pre.station_id.isin(val_ids) & (pre.date >= val_from)]
    assert not fit_ids & val_ids and not (fit_ids | val_ids) & test_ids
    manifest = dict(seed=SEED, training_station_ids=sorted(train_ids),
                    test_station_ids=sorted(test_ids), test_from=cutoff,
                    test_dates=sorted(test_keys.date.unique()),
                    validation_station_ids=sorted(val_ids), fit_station_ids=sorted(fit_ids),
                    validation_from=val_from, validation_until_exclusive=cutoff,
                    active_training_station_count=len(active), file_sha256=hashes,
                    test_key_sha256=hashlib.sha256(test_keys.to_csv(index=False).encode()).hexdigest(),
                    high_pollution_threshold=HIGH, selection_metric='validation station-day MAE',
                    prediction_time='after completed local-day reference aggregates are available',
                    environment=dict(python=sys.version, executable=sys.executable,
                                     sklearn=sklearn.__version__, pandas=pd.__version__,
                                     numpy=np.__version__, platform=platform.platform()))
    (OUT / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Active training stations: {len(active)} of 148; fit {len(fit_ids)}, validation {len(val_ids)}; validation starts {val_from}', flush=True)
    xf, mf = make_features(fit_targets, fit_refs[fit_refs.date < val_from])
    xv, mv = make_features(val_targets, fit_refs[fit_refs.date >= val_from])
    print(f'Eligible fit rows {len(xf)}, validation rows {len(xv)} ({mv.station_id.nunique()} reporting stations)', flush=True)
    records, configs, predictions = [], {}, {}

    def record(name, config, pred):
        configs[name] = config
        predictions[name] = pred
        records.append(dict(name=name, family=config['family'], **metrics(mv, pred)))

    for k in KS:
        for p in POWERS:
            for floor in FLOORS:
                name = idw_name(k, p, floor)
                record(name, dict(family='idw', idw_column=name, features=[], alpha=0, params={}), xv[name].to_numpy())
    best_idw = min(records, key=lambda r: (r['mae'], r['name']))['name']
    # Limited, predeclared model grids. All candidates use exactly the same rows.
    feature_cols = [c for c in xf if not c.startswith('idw_')] + list(dict.fromkeys([BASELINE, best_idw, idw_name(4, 2, .15)]))
    grid = {
        'random_forest': [dict(min_samples_leaf=leaf, max_depth=depth, max_features=.8)
                          for leaf, depth in [(8,12),(25,10),(60,8)]],
        'hist_gradient_boosting': [dict(max_leaf_nodes=leaves, min_samples_leaf=40, l2_regularization=reg, loss=loss)
                                  for leaves, reg, loss in [(7,10,'squared_error'),(15,20,'squared_error'),(7,10,'absolute_error')]],
        'residual_rf': [dict(min_samples_leaf=leaf, max_depth=depth, max_features=.8)
                        for leaf, depth in [(25,8),(60,6)]],
        'residual_hgb': [dict(max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=20, loss=loss)
                         for loss in ['squared_error','absolute_error']]}
    (OUT / 'search_plan.json').write_text(json.dumps(dict(idw_k=KS, idw_p=POWERS,
        idw_floor=FLOORS, model_grid=grid, residual_alpha=[.1,.25,.5,1.], features=feature_cols), indent=2))
    for family, params_list in grid.items():
        for i, params in enumerate(params_list):
            model = estimator(family, params)
            target = mf.actual.to_numpy()
            if family.startswith('residual'):
                target = target - xf[best_idw].to_numpy()
            model.fit(xf[feature_cols], target)
            pred = model.predict(xv[feature_cols])
            for alpha in ([.1,.25,.5,1.] if family.startswith('residual') else [1.]):
                name = f'{family}_{i}_alpha{alpha:g}'
                values = np.maximum(0, xv[best_idw].to_numpy() + alpha * pred if family.startswith('residual') else pred)
                record(name, dict(family=family, params=params, alpha=alpha,
                                  features=feature_cols, idw_column=best_idw), values)
            print(f'Finished {family} configuration {i}', flush=True)
    comparison = pd.DataFrame(records).sort_values(['mae','name'])
    comparison.to_csv(OUT / 'validation_comparison.csv', index=False)
    winner = comparison.iloc[0]['name']
    chosen = configs[winner]
    # Irrevocably record decision before constructing or scoring any test features.
    selection = dict(name=winner, **chosen, validation=metrics(mv, predictions[winner]))
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    val_preds = mv.copy()
    for family in comparison.family.unique():
        name = comparison[comparison.family == family].iloc[0]['name']
        val_preds[name] = predictions[name]
    val_preds['original_idw'] = predictions[BASELINE]
    val_preds.to_csv(OUT / 'validation_predictions.csv', index=False)
    print('FROZEN SELECTION:', winner, flush=True)

    # Refit only the selected method, on all original training stations and dates.
    xa, ma = make_features(pre, pre)
    bundle = dict(chosen, name=winner, model=None, reference_site_ids=sorted(train_ids),
                  heldout_site_ids=sorted(test_ids), cutoff=cutoff, high_threshold=HIGH,
                  prediction_time=manifest['prediction_time'], schema_version=1)
    if chosen['family'] != 'idw':
        fitted = estimator(chosen['family'], chosen['params'])
        y = ma.actual.to_numpy()
        if chosen['family'].startswith('residual'):
            y = y - xa[best_idw].to_numpy()
        fitted.fit(xa[feature_cols], y)
        bundle['model'] = fitted
    joblib.dump(bundle, OUT / 'selected_model.joblib')
    # The old model is diagnostic only, never a selection candidate: it saw validation sites.
    old_cols = old['features']
    legacy_features = xa.rename(columns={BASELINE:'idw', idw_name(4,2,.15):'idw4'})[old_cols]
    old_train_pred = np.maximum(0, xa[BASELINE].to_numpy() + old['model'].predict(legacy_features))
    audit = dict(fit_candidate_rows=len(fit_targets), fit_used_rows=len(xf),
                 validation_candidate_rows=len(val_targets), validation_used_rows=len(xv),
                 refit_rows=len(xa), original_model_training=metrics(ma,old_train_pred),
                 original_idw_training=metrics(ma,xa[BASELINE].to_numpy()),
                 training_target_mean=float(ma.actual.mean()),
                 training_nearest_km_mean=float(xa.nearest_km.mean()),
                 training_reference_count_mean=float(xa.reference_count.mean()),
                 colocated_training_rows=int((xa.nearest_km < .01).sum()),
                 validation_station_counts=mv.station_id.value_counts().to_dict(),
                 training_source_counts=pre.station_id.str.split('_').str[0].value_counts().to_dict())
    # Test gate: exactly the previously evaluated keys, no newly selected test cohort.
    test_targets = test_keys.merge(df, on=['date','station_id'], how='left', validate='one_to_one')
    assert test_targets.pm25.notna().all()
    assert (test_targets.date >= cutoff).all() and set(test_targets.station_id) <= test_ids
    xt, mt = make_features(test_targets, df[df.station_id.isin(train_ids) & (df.date >= cutoff)])
    assert set(zip(mt.date,mt.station_id)) == set(zip(test_keys.date,test_keys.station_id))
    assert len(xt) == old_result['test_examples']
    pred = predict_bundle(bundle, xt)
    base = xt[BASELINE].to_numpy()
    test_result = dict(selected_name=winner, selected=metrics(mt,pred), original_idw=metrics(mt,base),
                       evaluation_count=1, decision_changed_after_test=False)
    assert round(test_result['original_idw']['mae'],3) == old_result['baseline_idw_mae']
    pd.DataFrame(dict(date=mt.date,station_id=mt.station_id,actual=mt.actual,
                      original_idw=base, selected_prediction=pred)).to_csv(OUT / 'test_predictions.csv',index=False)
    (OUT / 'test_results.json').write_text(json.dumps(test_result,indent=2))
    audit.update(test_target_mean=float(mt.actual.mean()),
                 test_nearest_km_mean=float(xt.nearest_km.mean()),
                 test_reference_count_mean=float(xt.reference_count.mean()),
                 colocated_test_rows=int((xt.nearest_km < .01).sum()), elapsed_seconds=time.monotonic()-start)
    (OUT / 'data_audit.json').write_text(json.dumps(audit,indent=2))
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in paths}, 'Working artifacts changed'
    print(comparison.groupby('family',sort=False).head(1).to_string(index=False),flush=True)
    print(json.dumps(test_result,indent=2),flush=True)
    print('Original source, data and website artifacts unchanged (SHA256 verified).',flush=True)


if __name__ == '__main__':
    main()
