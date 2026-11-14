"""Screen every recorded coordinate, and exceptionally similar daily streams."""
from station_alias_audit import ROOT, OUT, haversine, stats
import pandas as pd
import numpy as np
import json

if (OUT/'additional_candidates.csv').exists():
    raise SystemExit('Refusing overwrite')
r=pd.read_csv(ROOT/'data/almaty_stations_pm25.csv.gz')
pos=r[['station_id','lat','lon']].drop_duplicates().dropna()
periods=r.groupby(['station_id','lat','lon'],as_index=False).agg(
    first_observed_utc=('datetime_utc','min'),last_observed_utc=('datetime_utc','max'),rows=('station_id','size'))
periods.to_csv(OUT/'coordinate_observation_ranges.csv',index=False)
d=pd.read_csv(OUT/'original_daily_cache.csv.gz')
pivot=d.pivot(index='date',columns='station_id',values='pm25')
corr=pivot.corr(min_periods=30)
existing=pd.read_csv(OUT/'pair_evidence.csv')
seen={tuple(sorted([a,b])) for a,b in zip(existing.a,existing.b)}
coords={sid:g[['lat','lon']].to_numpy() for sid,g in pos.groupby('station_id')}
ids=sorted(coords)
hour={sid:g.assign(hour=pd.to_datetime(g.datetime_utc,utc=True).dt.floor('h')).groupby('hour').value_ugm3.mean()
      for sid,g in r.groupby('station_id')}
rows=[]
for i,a in enumerate(ids):
    for b in ids[i+1:]:
        if (a,b) in seen:
            continue
        distance=min(haversine(*loc,coords[b]).min() for loc in coords[a])*1000
        value=corr.loc[a,b] if a in corr.index and b in corr.columns else np.nan
        near=distance<=100
        similar=value>=.995
        if not (near or similar):
            continue
        row=dict(a=a,b=b,minimum_recorded_distance_m=float(distance),daily_correlation=value,
                 reason='any_historical_coordinates_within_100m' if near else 'daily_correlation_ge_0.995_30_days')
        row.update({'hour_all_'+k:v for k,v in stats(hour[a],hour[b]).items()})
        rows.append(row)
pd.DataFrame(rows,columns=['a','b','minimum_recorded_distance_m','daily_correlation','reason',
    'hour_all_n','hour_all_mae','hour_all_correlation','hour_all_equal_fraction','hour_all_start','hour_all_end']).to_csv(OUT/'additional_candidates.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
