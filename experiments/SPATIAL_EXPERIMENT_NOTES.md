# AirGuard spatial experiments — 2026-11-14

No ML improvement over IDW was demonstrated. After correcting a possible
cross-provider station-alias problem, tuned IDW won validation. The website,
original model, original scores, station data and original test predictions
were left unchanged; SHA256 checks verified this. Backups are in
`spatial_20261114/backups/`.

## Important protocol limitation

The first run excluded target station IDs correctly but missed possible aliases
between OpenAQ and WAQI. This was discovered in the data audit **after** its one
test evaluation. Its validation result must not be treated as alias-safe.
The follow-up repaired this using training-period coordinates only and ran
**validation only**. It did not evaluate the test set again. Consequently the
final alias-safe model has **no test score**. The requested clean, single final
test confirmation was not achieved; the two runs below are deliberately not
presented as one clean experiment. No test result was used to choose parameters
or change the validation ranking.

## Why the original correction fails

- `spatial_ai.py` fits a 90-tree residual Random Forest directly, with no separate
  validation. Latitude/longitude can encode station-specific biases. Its
  in-sample MAE is 6.214 versus IDW's 14.225, yet the previously saved test MAEs
  are 11.934 versus 6.304. This supports overfitting / failure to transfer, though
  it does not prove which feature causes it.
- Only **21 of the nominal 148 training/reference stations** have any data
  before 2025-04-18; only 19 contribute to the original 9,434 eligible fit rows.
  The remaining 127 cannot contribute training labels before the cutoff.
- Mean available reference count changes from **8.16 in training to 108.10 in
  test**; mean nearest-station distance drops from **3.283 km to 1.466 km**.
  Large corrections learned in a sparse historical network need not transfer
  to a denser network where local IDW is more accurate.
- Original training uses squared-error fitting and adds the full correction.
  This can emphasize unusually high readings and sensor-specific residuals.
  Regularization, absolute-error loss and correction shrinkage were tested.
- `use_ml = ml_mae < baseline_mae` in the old code uses test results to choose
  the displayed method. That is test-driven model selection. New selection is
  frozen in `selection.json` before the test gate.
- No direct target-ID inclusion was found in original training: its
  `skip_id` logic works. However, physical station aliases are not handled.

## Station-data findings

The unchanged loader produces 29,957 station-days across 191 IDs, from
2020-04-10 to 2026-11-13. The raw file contains 727,357 records and no duplicate
`(station_id, datetime_utc)` keys. Sources include KGMT, OpenAQ, WAQI,
AirGradient and AirKaz, so IDs do not establish independent physical sensors.

Training-period coordinate audit found two possible alias pairs:

| IDs | Median-coordinate separation | Overlapping days | Mean absolute reading difference |
|---|---:|---:|---:|
| oaq_2812550 / waqi_dqihb6321 | 0.165 m | 134 | 0.749 |
| oaq_2812831 / waqi_dxkuv4229 | 0.524 m | 39 | 0.374 |

No paired daily readings were exactly equal, so alias identity is suspected,
not proven. The safe run groups these IDs conservatively and excludes all
references within 10 m of each target. Other undocumented aliases remain
possible; authoritative physical-sensor metadata would strengthen this audit.

`station_map.py` correctly converts UTC timestamps to Asia/Almaty dates, filters
invalid values/coordinates and aggregates daily readings. Its `hours` count is
actually distinct timestamps, not distinct hourly bins. Thus its “12 hours”
description assumes hourly source sampling; that assumption was not certified.
Aggregation was kept unchanged to preserve the existing test cohort and labels.
Same-day full-day averages are available only after that day has completed and
the feeds have arrived. These experiments do not establish live intraday or
next-day forecasting accuracy; arrival timestamps are absent.

## Fixed evaluation protocol

- Original 148 reference IDs and 37 held-out IDs come from the saved artifact,
  not from a newly randomized split. No held-out station is used as a training
  target or reference, even during earlier dates.
- Original cutoff remains **2025-04-18**. Evaluation keys come from the saved
  prediction file: exactly **2,112 rows, 34 reporting held-out stations**, on
  their original dates from **2025-05-17 through 2026-11-13**. Three held-out
  stations have no eligible evaluation rows. The cutoff and first eligible
  test date are different and have not been changed.
- Original eligibility was computed over the whole historical file by the old
  code. This inherited cohort is preserved, not reselected based on outcomes.
- New station split uses seed **20261114**, drawing 25% of the 21 active
  training-period stations. Validation dates are **2024-01-22 to 2025-04-17**;
  fitting uses earlier dates only. Validation stations are absent from fitting
  and reference pools on every date. Safe grouping expands 6 held-out IDs to 7;
  14 IDs remain in its fit/reference pool.
- The safe run has **1,946 eligible fitting rows**, **612 validation rows from
  3 reporting stations**, and **9,088 final refit rows**. Other
  targets fail the inherited minimum-three-references / nearest-within-5-km
  support rule. Comparison methods use exactly the same rows within each run.
- Features contain target coordinates and calendar phase, same-day neighboring
  PM2.5, distances, neighbor coordinates, counts and spread. They contain no
  station IDs, target PM2.5 history or future measurements. Target PM2.5 is used
  only as the supervised label and for post-prediction metric strata.
- Grid: 54 IDW configurations; 3 direct RF fits; 3 direct HGB fits; 2 residual
  RF and 2 residual HGB fits, each tested at correction multipliers
  0.1/0.25/0.5/1.0. There are 76 scored configurations, including duplicate IDW
  predictions when fewer than the requested neighbors are available.
  No random-row HGB early stopping is used. Ranking uses validation MAE only.
- High station-days mean actual target PM2.5 >=35 µg/m³. High days mean the
  validation/test target cohort's daily average >=35. These are descriptive
  strata, not citywide exposure labels or official AQI classifications. The
  safe validation has 180 high station-days and 213 rows on high days.

## Alias-safe validation results — final selection

All errors are µg/m³. Each model-family row is its best MAE configuration.

| Method | MAE | RMSE | High station-day MAE | High-day MAE |
|---|---:|---:|---:|---:|
| Original IDW settings | 11.511 | 18.762 | 21.790 | 20.921 |
| idw | 10.987 | 17.102 | 18.923 | 19.827 |
| residual_rf | 11.065 | 17.037 | 18.627 | 19.538 |
| residual_hgb | 11.085 | 17.281 | 19.223 | 19.846 |
| random_forest | 13.438 | 20.478 | 23.129 | 23.283 |
| hist_gradient_boosting | 15.310 | 25.247 | 33.685 | 30.224 |

Selected: **idw_k4_p1_floor0.05**, four nearest available neighbors, inverse-distance
power 1 and distance floor 0.05 km. The saved artifact is an IDW configuration,
so it deliberately contains no fitted sklearn estimator. All tested ML
families lost on the primary validation MAE metric. Some have slightly better
RMSE or high-pollution error; the selection criterion was not changed afterward.
Only three validation stations contribute, and station-days are correlated;
these results do not establish broad spatial generalization or significance.

## First-run validation — possible alias contamination

651 eligible validation rows, four stations. One possible alias pair crossed
the split. These scores are retained for audit, not as the final selection.

| Method | MAE | RMSE | High station-day MAE | High-day MAE |
|---|---:|---:|---:|---:|
| Original IDW settings | 10.780 | 18.052 | 21.653 | 20.787 |
| residual_hgb | 10.204 | 17.159 | 20.436 | 20.160 |
| idw | 10.348 | 16.921 | 20.217 | 20.581 |
| residual_rf | 10.444 | 16.817 | 19.877 | 20.255 |
| random_forest | 13.165 | 20.072 | 23.196 | 23.101 |
| hist_gradient_boosting | 14.993 | 25.382 | 34.649 | 31.195 |

That run selected residual HGB with absolute-error loss, seven leaf nodes,
minimum leaf size 60, L2=20, 180 iterations, learning rate .05 and a 0.25
correction multiplier on tuned IDW (k=12, power=1, floor=.05 km).

## Single test evaluation — first-run frozen model only

| Method | MAE | RMSE | High station-day MAE | High-day MAE |
|---|---:|---:|---:|---:|
| Original IDW | 6.304 | 9.370 | 9.263 | 8.576 |
| First-run selected residual HGB | 6.767 | 10.234 | 10.819 | 9.198 |

The residual method's test MAE is 7.35% worse than the original IDW baseline.
It improves over the old residual forest's saved 11.934 MAE, but does not beat
IDW. No other experimental candidate was scored on test. The original saved
6.304 IDW result was reproduced. There is no claimed test result for the final
alias-safe tuned-IDW artifact, and it was not promoted to the website.

## Outputs and verification

- `../spatial_experiments.py`: experiment implementation. Existing completed
  directories are guarded against overwriting / accidental repeat evaluation.
- `spatial_20261114_alias_safe/selected_model.joblib`: final validation-selected
  IDW artifact, with reference IDs, features/configuration and alias guard flag.
- Each run: complete `validation_comparison.csv`, compact
  `family_comparison.csv`, `validation_predictions.csv`, `split_manifest.json`,
  `search_plan.json`, `selection.json`, `data_audit.json` and selected artifact.
- First run only: `test_results.json`, `test_predictions.csv`, `run.log`,
  backed-up production files and the exact first-run experiment source.
- Synthetic regression tests passed for target/alias-value invariance,
  exclusion of other dates and IDW inference.
- Gradio imports with 38 blocks; existing spatial map (2 traces), measured map
  (30 table rows), and result-summary callback (12 rows) passed. No browser
  interaction test was performed.
- Execution used `D:/AirGuard/almaty-airguard/.venv/Scripts/python.exe`.
  TEMP, TMP, joblib temporary files and artifacts were kept under D:.

To use an artifact without changing the website, load it with joblib, construct
target rows containing date/station_id/lat/lon and a placeholder pm25 value,
filter same-day references to `reference_site_ids`, call
`make_features(targets, references, exclude_colocated=bundle['exclude_colocated'])`,
then `predict_bundle(bundle, features)`. Only eligible rows are returned, with
their corresponding keys in the second result from `make_features`.

Recommendation: keep the current IDW website behavior. Obtain a genuinely new
future evaluation period and better physical-station identifiers before
confirming or deploying a revised model. Do not reuse this test to tune again.
