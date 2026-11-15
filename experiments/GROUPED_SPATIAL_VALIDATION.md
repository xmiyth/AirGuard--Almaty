# AirGuard spatial PM2.5 validation

## Result

Tuned inverse-distance weighting (IDW) had the lowest validation MAE and RMSE.
Its MAE was 2.96% lower than the original IDW settings (0.198 µg/m³).
The closest ML candidate, residual Histogram Gradient Boosting without absolute
coordinates, was 0.0034 µg/m³ worse on MAE. That difference is small, and the
candidate had lower MAE at 53 of the 110 evaluated sites. The comparison
supports retaining IDW for the current map and keeping the tuned configuration
as a candidate for future confirmation.

| Method | MAE | RMSE | MAE on high station-days | Site-average MAE |
| --- | ---: | ---: | ---: | ---: |
| Previously tuned IDW | 6.494 | 10.529 | 9.086 | 6.992 |
| Residual HistGradientBoosting, relative features | 6.497 | 10.540 | 9.097 | 6.998 |
| Residual HistGradientBoosting | 6.505 | 10.552 | 9.109 | 7.005 |
| Residual Random Forest | 6.508 | 10.556 | 9.111 | 7.008 |
| Original IDW settings | 6.692 | 10.756 | 9.416 | 7.203 |
| Random Forest | 8.427 | 15.323 | 12.778 | 8.851 |
| HistGradientBoosting | 8.773 | 15.908 | 13.411 | 9.207 |

Errors are in µg/m³. High station-days are the 1,813 target readings at or
above 35 µg/m³, a descriptive analysis threshold. Site-average MAE weights
each evaluated site equally. See [full metrics](grouped_validation/comparison.csv),
[per-site metrics](grouped_validation/by_site.csv), and the
[split manifest](grouped_validation/split_manifest.json).

## Data and exclusions

- [AirData.kz station snapshot](https://github.com/qazybekb/AirDatakz-OpenData/blob/98ad95f7f60cd4517e45947a429d92626d238889/almaty/pm25.csv.gz),
  commit `98ad95f7f60cd4517e45947a429d92626d238889`. The gzip SHA-256 is
  `69f2dccf5feb827c24f2f50a0031c0aee1ad4d5344ff310a2d0634e551e55947`.
  The project loader reproduced all 29,957 audited local station-days
  from 191 reporting IDs when checked against the committed daily cache.
  The spatial experiment uses these station measurements. The separate
  Open-Meteo/CAMS citywide forecast is outside this comparison.
- Reused the [station grouping registry](station_alias_audit_20261114/station_group_registry.csv)
  and [alias evidence](station_alias_audit_20261114/alias_decisions.csv).
  The audit found plausible feed aliases and two site groups crossing the
  original train/test split. Some site groups reflect proximity or uncertain
  positions. Membership controls separation without asserting hardware
  identity.
- Started the experiment window on March 17, 2026, the cutoff the audit
  used for earlier WAQI import discrepancies. Of 10,480 rows in the date
  window, removed 2,268 from sites linked to the old test and 184 from
  two IDs with recorded coordinate displacement of at least 100 m.
  Retained one original station-day per site-day using the published source
  priority, then sample count and station ID. This removed one redundant
  row and left 8,027 rows across 127 site groups. Measurements were preserved
  in the input files without averaging feeds or filling in missing data.
  The manifest lists excluded IDs and input checksums.

## Split and model selection

- Five seeded folds assign entire site groups to validation. A held-out
  site's labels and same-day readings are excluded from that fold's fit and
  reference pools. When constructing fit examples, the fit target site itself
  is removed from that day's reference set. The feature builder also drops
  potential references within 10 m of each target.
- Fit dates are March 17 through September 30, 2026. Eligible fit rows in
  practice span March 19 through April 25. Each fold used 2,813–3,025 fit
  site-days from 81–87 sites. Validation dates are October 1 through
  November 13, 2026, with 3,537 of 3,910 candidate rows scored across
  110 sites on 35 reporting dates. All methods use identical scored rows.
  Eligibility requires at least three other reference sites and a nearest
  reference within 5 km.
- The configurations come from the prior audit policy. The additional
  residual HGB candidate uses the same settings as its fixed counterpart
  but omits target and neighbor latitude and longitude. It retains relative
  distances, neighbor readings and counts, seasonal features, and IDW
  estimates. This probes sensitivity to location-specific learning.
  Residual methods add 10% of the predicted correction to tuned IDW.
  No configurations were tuned against these folds.
- This is exploratory historical validation. The grouping registry uses the
  audited snapshot including the validation dates, and coverage and network
  composition change by date. The historical test remains quarantined.
  Future dates and a site roster frozen before collection would provide
  independent confirmation. Errors across sites and nearby dates can be
  correlated.

## Where IDW works and fails

The tuned configuration uses up to four neighbors with weights proportional
to inverse distance, floored at 0.05 km. It needs the target coordinates and
quality-checked readings and coordinates from other sites for the completed
local day. Full-day averages become available after reporting completes.
The evaluation covers same-day interpolation. Feed arrival times and
intraday reliability are unknown.

Validation MAE increased from 4.85 µg/m³ when the nearest reference was
within 1 km to 8.97 µg/m³ when it was 2–5 km away. This association can also
reflect differences in site conditions. On November 8, `site_ag_167635`
reported 150.44 µg/m³ while tuned IDW estimated 57.59 µg/m³ from neighbors at
least 3.16 km away. On October 27, it overestimated `site_ag_167651` by
70.05 µg/m³. A weighted average tends to miss local peaks and can transfer a
neighbor's elevation in PM2.5 to a quieter site. Validation used on average
89 reporting reference sites per scored row, so these errors do not
characterize coverage in a sparser network or beyond the support rule.

## Reproduce

From the repository root on Windows, with the dependencies installed:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest -v test_spatial_experiments
.\.venv\Scripts\python.exe -X utf8 grouped_spatial_validation.py --output data/grouped_validation_recheck
```

The script uses the audited cache committed in the repository and writes a
comparison, site-level metrics, a split manifest, and per-example predictions.
Use a fresh output path. This run used Python 3.14.7, pandas 2.3.3, NumPy
2.5.4, and scikit-learn 1.9.1.
