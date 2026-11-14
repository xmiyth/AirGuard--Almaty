"""Read-only audit of station identity evidence. Outputs are separate from production."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import os

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments' / 'station_alias_audit_20261114'
(OUT / 'tmp').mkdir(parents=True, exist_ok=True)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT / 'tmp')
import numpy as np
import pandas as pd
from spatial_ai import haversine
from station_map import daily_stations


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def norm(s):
    return re.sub(r'\W+', ' ', unicodedata.normalize('NFKC', str(s)).casefold()).strip()


def stats(a,b):
    both = pd.concat([a.rename('a'),b.rename('b')],axis=1,join='inner').dropna()
    if both.empty:
        return dict(n=0,mae=None,correlation=None,equal_fraction=None,start=None,end=None)
    return dict(n=len(both),mae=float(abs(both.a-both.b).mean()),
                correlation=float(both.a.corr(both.b)) if len(both)>2 and both.a.std()>0 and both.b.std()>0 else None,
                equal_fraction=float(np.isclose(both.a,both.b,rtol=0,atol=1e-6).mean()),
                start=str(both.index.min()),end=str(both.index.max()))


def main():
    if (OUT/'pair_evidence.csv').exists():
        raise SystemExit('Audit already exists; refusing overwrite')
    protected = [p for p in ROOT.iterdir() if p.is_file() and p.name in
                 ['app.py','spatial_ai.py','station_map.py','airguard.py','requirements.txt']]
    protected += [p for p in (ROOT/'data').iterdir() if p.is_file()]
    before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    (OUT/'production_sha256_before.json').write_text(json.dumps(before,indent=2))
    raw = pd.read_csv(ROOT/'data/almaty_stations_pm25.csv.gz')
    raw['row_number'] = np.arange(len(raw))+2
    raw['timestamp'] = pd.to_datetime(raw.datetime_utc,utc=True,errors='coerce')
    raw['date'] = raw.timestamp.dt.tz_convert('Asia/Almaty').dt.strftime('%Y-%m-%d')
    raw['hour_utc'] = raw.timestamp.dt.floor('h')
    daily = daily_stations()
    daily.to_csv(OUT/'original_daily_cache.csv.gz',index=False)
    registry = pd.read_csv(OUT/'evidence/stations.csv').set_index('station_id')
    first = json.loads((ROOT/'experiments/spatial_20261114/split_manifest.json').read_text())
    safe = json.loads((ROOT/'experiments/spatial_20261114_alias_safe/split_manifest.json').read_text())
    t,v,oldv = set(first['training_station_ids']),set(safe['validation_station_ids']),set(first['validation_station_ids'])
    test = set(first['test_station_ids'])
    rows=[]
    for sid,g in raw.groupby('station_id'):
        d=daily[daily.station_id==sid]
        pos=g[['lat','lon']].drop_duplicates().dropna()
        mid=g[['lat','lon']].median()
        meta=registry.loc[sid] if sid in registry.index else pd.Series(dtype=object)
        rows.append(dict(station_id=sid,names=' | '.join(g.station_name.dropna().unique()),
            sources=' | '.join(sorted(g.source.unique())),lat=mid.lat,lon=mid.lon,
            coordinate_variants=len(pos),max_displacement_km=float(haversine(*mid,pos.to_numpy()).max()) if len(pos) else None,
            registry_name=meta.get('name',''),operator=meta.get('operator',''),
            raw_rows=len(g),raw_start=g.date.min(),raw_end=g.date.max(),
            subhour_rows=int((g.timestamp!=g.hour_utc).sum()),
            duplicate_station_hours=int(g.duplicated('hour_utc').sum()),
            daily_rows=len(d),daily_start=d.date.min() if len(d) else '',daily_end=d.date.max() if len(d) else '',
            original_partition='train' if sid in t else 'test_inspected' if sid in test else 'not_in_original_split',
            first_validation_partition='validation' if sid in oldv else 'fit' if sid in first['fit_station_ids'] else 'inactive_or_test',
            previous_validation_partition='validation' if sid in v else 'fit' if sid in safe['fit_station_ids'] else 'inactive_or_test'))
    inventory=pd.DataFrame(rows).set_index('station_id')
    inventory.to_csv(OUT/'station_inventory.csv')
    hour={sid:g.groupby('hour_utc').value_ugm3.mean() for sid,g in raw.groupby('station_id')}
    day={sid:g.set_index('date').pm25 for sid,g in daily.groupby('station_id')}
    pairs=[]
    ids=list(inventory.index)
    for i,a in enumerate(ids):
        x=inventory.loc[a]
        for b in ids[i+1:]:
            y=inventory.loc[b]
            dist=float(haversine(x.lat,x.lon,np.array([[y.lat,y.lon]]))[0])
            same_name=bool(norm(x.names)==norm(y.names) and len(norm(x.names))>8)
            embassy={a,b}=={'oaq_8876','waqi_12774'}
            if dist>.1 and not (same_name and dist<=1) and not embassy:
                continue
            row=dict(a=a,b=b,distance_m=dist*1000,same_name=same_name,
                name_a=x.names,name_b=y.names,operator_a=x.operator,operator_b=y.operator,
                original_a=x.original_partition,original_b=y.original_partition,
                first_validation_a=x.first_validation_partition,first_validation_b=y.first_validation_partition,
                previous_validation_a=x.previous_validation_partition,previous_validation_b=y.previous_validation_partition,
                candidate_reason='embassy_feed' if embassy else 'within_100m' if dist<=.1 else 'same_name_within_1km')
            for label,aa,bb in [('hour_all',hour[a],hour[b]),
                ('hour_pre',hour[a][hour[a].index<pd.Timestamp('2025-04-17T19:00:00Z')],hour[b][hour[b].index<pd.Timestamp('2025-04-17T19:00:00Z')]),
                ('day_all',day.get(a,pd.Series(dtype=float)),day.get(b,pd.Series(dtype=float)))]:
                row.update({label+'_'+k:v for k,v in stats(aa,bb).items()})
            pairs.append(row)
    pairs=pd.DataFrame(pairs).sort_values(['distance_m','a','b'])
    pairs.to_csv(OUT/'pair_evidence.csv',index=False)
    periods=pd.read_csv(OUT/'evidence/station_periods.csv')
    summary=dict(raw_rows=len(raw),raw_station_ids=len(inventory),raw_start=raw.date.min(),raw_end=raw.date.max(),
        daily_rows=len(daily),daily_station_ids=daily.station_id.nunique(),daily_start=daily.date.min(),daily_end=daily.date.max(),
        screened_pairs=len(pairs),raw_duplicate_station_timestamp=int(raw.duplicated(['station_id','timestamp']).sum()),
        subhour_rows=int((raw.timestamp!=raw.hour_utc).sum()),
        registry_missing_ids=sorted(set(ids)-set(registry.index)),
        upstream_period_rows_for_local_ids=int(periods.station_id.isin(ids).sum()),
        max_station_coordinate_displacement_km=float(inventory.max_displacement_km.max()),
        protected_files_unchanged=before=={str(p.relative_to(ROOT)):sha(p) for p in protected},
        status='exploratory: existing test labels inspected previously; no fresh holdout certified')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    print(pairs[['a','b','distance_m','operator_a','operator_b','hour_all_n','hour_all_mae','hour_all_correlation','original_a','original_b']].to_string(index=False))


if __name__=='__main__':
    main()
