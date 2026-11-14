"""Evidence-led curation and EXPLORATORY validation. Never scores old test targets.

Raw data, website and production artifacts remain read-only. Every station-day
and raw row receives a disposition. Identity inference and split isolation are
separate concepts: spatial grouping does not assert identical hardware.
"""
from station_alias_audit import ROOT, OUT, sha, norm
import os
os.environ['OMP_NUM_THREADS']='4'
import json
import itertools
import joblib
import pandas as pd
import numpy as np
from spatial_experiments import make_features, metrics, estimator, BASELINE, idw_name


class Groups:
    def __init__(self, ids):
        self.parent={x:x for x in ids}

    def root(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.root(self.parent[x])
        return self.parent[x]

    def join(self,a,b):
        a,b=sorted([self.root(a),self.root(b)])
        self.parent[b]=a


def create_policy(inv,pairs,extra):
    decisions=[]
    identity=Groups(inv.index)
    site=Groups(inv.index)
    for row in itertools.chain(pairs.to_dict('records'),extra.to_dict('records')):
        a,b=row['a'],row['b']
        x,y=inv.loc[a],inv.loc[b]
        distance=row.get('distance_m',row.get('minimum_recorded_distance_m'))
        same_name=norm(x.names)==norm(y.names)
        operators={str(x.operator),str(y.operator)}
        alias=False
        if {a,b} in [{'oaq_2812550','waqi_dqihb6321'},{'oaq_2812831','waqi_dxkuv4229'}]:
            classification='high_confidence_feed_alias_inferred'
            reason='Both provider pages identify Clarity; same name and sub-metre coordinates; 883+/2873 aligned hours with correlation >0.995. No published serial crosswalk.'
            evidence='evidence/provider/openaq_arena.html; evidence/provider/openaq_road.html; evidence/provider/waqi_clarity_arena.html; evidence/provider/waqi_clarity_road.html; pair_evidence.csv'
            alias=True
        elif {a,b}=={'waqi_-410893','waqi_82475'}:
            classification='probable_feed_rename_no_serial_confirmation'
            reason='Identical name/coordinates/operator (sensor.community); sequential periods without overlap. Treat as one lineage for split isolation, not a proven serial match.'
            evidence='evidence/stations.csv; station_inventory.csv; pair_evidence.csv'
            alias=True
        elif same_name and distance<1 and ('AirGradient' in operators):
            classification='documented_provider_transition_inferred_station_match'
            reason='Upstream documents OpenAQ-to-direct-AirGradient transition; identical station name/coordinates. Sparse overlap cannot independently certify hardware identity.'
            evidence='evidence/README.md:261; evidence/stations.csv; pair_evidence.csv'
            alias=True
        elif same_name and distance<1 and operators=={'AirKaz','AirKaz.org'}:
            classification='same_provider_site_lineage_unconfirmed_serial'
            reason='Same AirKaz name and coordinates, separated historical periods. Conservative lineage link; no measurements blended.'
            evidence='evidence/stations.csv; station_inventory.csv'
            alias=True
        elif {a,b}=={'oaq_8876','waqi_12774'}:
            classification='unresolved_embassy_feed_identity_and_location'
            reason='Both identify the Almaty US monitor lineage, but coordinates differ by 7.35 km and hourly correlation is only 0.750. Quarantine together, prefer documented OpenAQ location, exclude old WAQI.'
            evidence='evidence/README.md:305-307; evidence/provider/waqi_embassy.html; evidence/stations.csv; pair_evidence.csv'
        elif distance<=100:
            classification='shared_site_or_position_uncertainty_not_proven_alias'
            reason='At least one recorded coordinate pair lies within 100 m. Isolate as one site group; different providers/names or distinct readings do not establish identical instruments.'
            evidence='coordinate_observation_ranges.csv; pair_evidence.csv; additional_candidates.csv'
        else:
            classification='retain_separate_no_alias_evidence'
            reason='Same area/name or daily correlation alone is insufficient. Providers/coordinates differ or hourly series are distinct; do not merge.'
            evidence='evidence/stations.csv; pair_evidence.csv; additional_candidates.csv'
        group=alias or distance<=100 or {a,b}=={'oaq_8876','waqi_12774'}
        if alias:
            identity.join(a,b)
        if group:
            site.join(a,b)
        decisions.append(dict(a=a,b=b,classification=classification,identity_group_link=alias,
            split_group_link=group,measurement_merge=False,reason=reason,evidence=evidence,distance_m=distance))
    return pd.DataFrame(decisions),identity,site


def main():
    if (OUT/'exploratory_metrics.csv').exists() or (OUT/'resolution_policy.json').exists():
        raise SystemExit('Frozen run exists; refusing overwrite')
    before=json.loads((OUT/'production_sha256_before.json').read_text())
    inv=pd.read_csv(OUT/'station_inventory.csv').set_index('station_id').fillna('')
    pairs=pd.read_csv(OUT/'pair_evidence.csv')
    extra=pd.read_csv(OUT/'additional_candidates.csv')
    decisions,identity,site=create_policy(inv,pairs,extra)
    decisions.to_csv(OUT/'alias_decisions.csv',index=False)
    old=json.loads((ROOT/'experiments/spatial_20261114/split_manifest.json').read_text())
    safe=json.loads((ROOT/'experiments/spatial_20261114_alias_safe/split_manifest.json').read_text())
    train,test,val=map(set,[old['training_station_ids'],old['test_station_ids'],safe['validation_station_ids']])
    site_map={sid:'site_'+site.root(sid) for sid in inv.index}
    test_groups={site_map[s] for s in test}
    val_groups={site_map[s] for s in val}-test_groups
    fit_groups={site_map[s] for s in train}-test_groups-val_groups
    inv['identity_group']=['lineage_'+identity.root(s) for s in inv.index]
    inv['site_group']=[site_map[s] for s in inv.index]
    inv['resolved_partition']=['test_quarantine' if g in test_groups else 'validation' if g in val_groups
                               else 'fit' if g in fit_groups else 'unassigned' for g in inv.site_group]
    inv.to_csv(OUT/'station_group_registry.csv')
    crossings=[]
    for group,g in inv.groupby('site_group'):
        original=set(g.original_partition)-{'not_in_original_split'}
        first=set(g.first_validation_partition)-{'inactive_or_test'}
        prev=set(g.previous_validation_partition)-{'inactive_or_test'}
        if len(original)>1 or len(first)>1 or len(prev)>1:
            crossings.append(dict(site_group=group,members='|'.join(g.index),
                original_train_test_crossing=len(original)>1,first_fit_validation_crossing=len(first)>1,
                previous_fit_validation_crossing=len(prev)>1,resolution=g.resolved_partition.iloc[0]))
    pd.DataFrame(crossings).to_csv(OUT/'split_crossings.csv',index=False)
    # Freeze evidence rules and fixed model configurations before new scores.
    tuned=idw_name(4,1,.05)
    configs=[
        dict(name='original_idw_settings',family='idw',column=BASELINE),
        dict(name='previously_selected_tuned_idw',family='idw',column=tuned),
        dict(name='fixed_random_forest',family='random_forest',params=dict(min_samples_leaf=25,max_depth=10,max_features=.8)),
        dict(name='fixed_hist_gradient_boosting',family='hist_gradient_boosting',params=dict(max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=10,loss='absolute_error')),
        dict(name='fixed_residual_rf',family='residual_rf',alpha=.1,params=dict(min_samples_leaf=60,max_depth=6,max_features=.8)),
        dict(name='fixed_residual_hgb',family='residual_hgb',alpha=.1,params=dict(max_leaf_nodes=7,min_samples_leaf=60,l2_regularization=20,loss='absolute_error'))]
    policy=dict(status='EXPLORATORY; no untouched historical holdout certified',
        test_evaluations=0,raw_measurements_merged=0,group_distance_m=100,
        grouping='Identity lineage links separate from conservative site groups; transitive across all recorded positions.',
        precedence='Inspected-test groups take precedence over validation, then fit. No split reassignment to make scores improve.',
        excluded_quality='All WAQI before 2026-03-17: upstream unexplained earlier-import discrepancies; no attempted recalibration.',
        representative='One original station-day per site-day; source priority kgmt, kgmt_auto, airgradient, openaq, airkaz, waqi, then more samples, then lexical station ID. Never average aliases or backfill labels.',
        original_cutoff='2025-04-18',fit_until_exclusive='2024-01-22',
        validation_from='2024-01-22',validation_until_exclusive='2025-04-18',
        fixed_models=configs,metadata_retrieved='2026-11-14',
        evaluation_scope='Original training IDs only, with full-site exclusions; no test target evaluation.',
        future_holdout='Proposed 2026-11-15 through 2027-02-28 local Almaty dates; not yet collected or scored.')
    (OUT/'resolution_policy.json').write_text(json.dumps(policy,indent=2))
    d=pd.read_csv(OUT/'original_daily_cache.csv.gz')
    d['original_station_id']=d.station_id
    d['site_group']=d.station_id.map(site_map)
    d['partition']=d.station_id.map(inv.resolved_partition)
    d['sources']=d.station_id.map(inv.sources)
    d['exclude_waqi_quality']=d.sources.eq('waqi') & (d.date<'2026-03-17')
    d['exclude_inspected_test_group']=d.site_group.isin(test_groups)
    d['exclude_not_original_training_id']=~d.station_id.isin(train)
    d['outside_training_period']=d.date>='2025-04-18'
    flags=['exclude_waqi_quality','exclude_inspected_test_group','exclude_not_original_training_id','outside_training_period']
    d['preeligible']=~d[flags].any(axis=1)
    source_order={'kgmt':0,'kgmt | kgmt_auto':0,'kgmt_auto':0,'airgradient':1,'openaq':2,'airkaz':3,'waqi':4}
    d['priority']=d.sources.map(source_order).fillna(9)
    candidates=d[d.preeligible].sort_values(['date','site_group','priority','hours','station_id'],ascending=[True,True,True,False,True])
    chosen=candidates.drop_duplicates(['date','site_group'])
    chosen_keys=set(zip(chosen.date,chosen.station_id))
    d['exclude_redundant_site_day']=d.preeligible & ~pd.Series(list(zip(d.date,d.station_id))).isin(chosen_keys)
    flags.append('exclude_redundant_site_day')
    d['curation_reason']=d.apply(lambda r:';'.join(f for f in flags if r[f]) or 'retain_original_daily_value',axis=1)
    # Keep rows from distinct site groups and use site IDs for target exclusion.
    clean=chosen.copy()
    clean.station_id=clean.site_group
    assert not clean.site_group.isin(test_groups).any()
    assert not clean.duplicated(['date','station_id']).any()
    clean.to_csv(OUT/'curated_training_period_daily.csv.gz',index=False)
    fit=clean[(clean.partition=='fit')&(clean.date<'2024-01-22')]
    validation=clean[(clean.partition=='validation')&(clean.date>='2024-01-22')]
    refs=clean[clean.partition=='fit']
    assert not set(fit.station_id)&set(validation.station_id)
    xf,mf=make_features(fit,refs[refs.date<'2024-01-22'],exclude_colocated=True)
    xv,mv=make_features(validation,refs[refs.date>='2024-01-22'],exclude_colocated=True)
    eligible_keys=set(zip(mf.date,mf.station_id))|set(zip(mv.date,mv.station_id))
    candidate_keys=set(zip(fit.date,fit.station_id))|set(zip(validation.date,validation.station_id))
    d['evaluation_role']=np.where((d.partition=='fit')&(d.date<'2024-01-22'),'fit_candidate',
        np.where((d.partition=='validation')&(d.date>='2024-01-22')&(d.date<'2025-04-18'),'validation_candidate','outside_target_window'))
    def support_reason(row):
        if row.curation_reason!='retain_original_daily_value':
            return 'policy_excluded'
        key=(row.date,row.site_group)
        if key in eligible_keys:
            return 'evaluated'
        if key in candidate_keys:
            return 'insufficient_spatial_support_min3_sites_nearest5km'
        return 'reference_only_or_outside_target_window'
    d['evaluation_disposition']=d.apply(support_reason,axis=1)
    d.to_csv(OUT/'station_day_disposition.csv.gz',index=False)
    # Trace all raw input rows to policy decisions, including upstream-loader exclusions.
    raw=pd.read_csv(ROOT/'data/almaty_stations_pm25.csv.gz')
    ts=pd.to_datetime(raw.datetime_utc,utc=True,errors='coerce')
    raw['date']=ts.dt.tz_convert('Asia/Almaty').dt.strftime('%Y-%m-%d')
    trace=pd.DataFrame(dict(raw_csv_line=np.arange(len(raw))+2,station_id=raw.station_id,date=raw.date))
    trace=trace.merge(d[['station_id','date','site_group','curation_reason','evaluation_disposition']],on=['station_id','date'],how='left',validate='many_to_one')
    trace['curation_reason']=trace.curation_reason.fillna('not_in_original_daily_loader_output')
    trace['evaluation_disposition']=trace.evaluation_disposition.fillna('original_loader_excluded')
    trace.to_csv(OUT/'raw_row_disposition.csv.gz',index=False)
    hour_counts=raw.assign(hour=ts.dt.floor('h')).groupby(['date','station_id']).hour.nunique()
    original_keys=pd.MultiIndex.from_frame(d[['date','station_id']])
    under12=int((hour_counts.reindex(original_keys)<12).sum())
    features=[c for c in xf if not c.startswith('idw_')]+[BASELINE,tuned,idw_name(4,2,.15)]
    results=[]
    predictions=mv.copy()
    for config in configs:
        if config['family']=='idw':
            pred=xv[config['column']].to_numpy()
        else:
            model=estimator(config['family'],config['params'])
            y=mf.actual.to_numpy()
            residual=config['family'].startswith('residual')
            if residual:
                y=y-xf[tuned].to_numpy()
            model.fit(xf[features],y)
            p=model.predict(xv[features])
            pred=np.maximum(0,xv[tuned].to_numpy()+config['alpha']*p if residual else p)
        results.append(dict(name=config['name'],status='exploratory_validation',**metrics(mv,pred)))
        predictions[config['name']]=pred
    pd.DataFrame(results).to_csv(OUT/'exploratory_metrics.csv',index=False)
    predictions.to_csv(OUT/'exploratory_validation_predictions.csv',index=False)
    bysite=[]
    for sid,g in predictions.groupby('station_id'):
        for name in [c['name'] for c in configs]:
            bysite.append(dict(site_group=sid,method=name,n=len(g),date_from=g.date.min(),date_to=g.date.max(),mae=float(abs(g[name]-g.actual).mean())))
    pd.DataFrame(bysite).to_csv(OUT/'validation_by_site.csv',index=False)
    # A proposed prospective roster, selected on past coverage/metadata only.
    recent=d[(d.date>='2026-10-09')&(d.date<='2026-11-13')&d.original_station_id.isin(test)]
    coverage=recent.groupby('original_station_id').date.nunique()
    proposed=[]
    for sid,n in coverage.items():
        if n>=20 and inv.loc[sid,'sources']=='airgradient' and inv.loc[sid,'max_displacement_km']<.1:
            proposed.append(dict(station_id=sid,site_group=site_map[sid],historical_coverage_days=int(n),
                proposed_start='2026-11-15',proposed_end='2027-02-28',status='proposed_only_not_scored'))
    pd.DataFrame(proposed).to_csv(OUT/'prospective_roster_proposed.csv',index=False)
    active_before=set(d.loc[(d.date<'2025-04-18')&d.original_station_id.isin(train),'original_station_id'])
    station_counts=[]
    for name,frame in [('original_training_period',d[(d.date<'2025-04-18')&d.original_station_id.isin(train)]),
                       ('curated_training_period',clean),('fit_candidates',fit),('validation_candidates',validation)]:
        station_counts.append(dict(stage=name,station_ids=int(frame.original_station_id.nunique()),site_groups=int(frame.site_group.nunique()),
            rows=len(frame),date_from=frame.date.min(),date_to=frame.date.max()))
    for name,frame in [('fit_evaluated',mf),('validation_evaluated',mv)]:
        station_counts.append(dict(stage=name,station_ids=int(frame.station_id.nunique()),site_groups=int(frame.station_id.nunique()),
            rows=len(frame),date_from=frame.date.min(),date_to=frame.date.max()))
    pd.DataFrame(station_counts).to_csv(OUT/'station_counts_and_dates.csv',index=False)
    summary=dict(evaluation_status='EXPLORATORY',new_test_scores=0,untouched_historical_test_certified=False,
        original_train_ids=len(train),original_test_ids=len(test),raw_station_ids=len(inv),
        lineage_groups=int(inv.identity_group.nunique()),conservative_site_groups=int(inv.site_group.nunique()),
        original_train_test_crossing_groups=sum(x['original_train_test_crossing'] for x in crossings),
        first_fit_validation_crossing_groups=sum(x['first_fit_validation_crossing'] for x in crossings),
        previous_fit_validation_crossing_groups=sum(x['previous_fit_validation_crossing'] for x in crossings),
        active_original_training_ids=len(active_before),active_train_ids_quarantined_for_test_group=sorted(active_before&set(inv[inv.resolved_partition=='test_quarantine'].index)),
        original_daily_rows_with_fewer_than_12_distinct_clock_hours=under12,
        proposed_prospective_station_ids=len(proposed),proposed_prospective_site_groups=len({r['site_group'] for r in proposed}),
        protected_files_unchanged=all(sha(ROOT/p)==digest for p,digest in before.items()),
        no_production_or_experimental_models_written=True,
        policy_sha256=sha(OUT/'resolution_policy.json'))
    (OUT/'resolution_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(station_counts).to_string(index=False))
    print(pd.DataFrame(results)[['name','n','stations','mae','rmse','high_mae']].to_string(index=False))


if __name__=='__main__':
    main()
