"""Compare frozen PM2.5 interpolation candidates on an exploratory site/time split."""

import argparse
import json
import platform
from pathlib import Path

from spatial_experiments import BASELINE, HIGH, digest, estimator, idw_name, make_features, metrics

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "experiments" / "station_alias_audit_20261114"
DAILY = AUDIT / "original_daily_cache.csv.gz"
REGISTRY = AUDIT / "station_group_registry.csv"
POLICY = AUDIT / "resolution_policy.json"
FIT_START = "2026-03-17"
VALIDATION_START = "2026-10-01"
VALIDATION_END = "2026-11-14"  # exclusive, Almaty local dates
SOURCE_PRIORITY = {
    "kgmt": 0,
    "kgmt | kgmt_auto": 0,
    "kgmt_auto": 0,
    "airgradient": 1,
    "openaq": 2,
    "airkaz": 3,
    "waqi": 4,
}
SEED = 20261114
FOLDS = 5
TUNED = idw_name(4, 1, 0.05)
RELATIVE_MODEL = "relative_residual_hgb"


def prepare_data():
    registry = pd.read_csv(REGISTRY, usecols=[
        "station_id", "site_group", "sources", "resolved_partition", "max_displacement_km",
    ])
    daily = pd.read_csv(DAILY).merge(
        registry, on="station_id", how="left", validate="many_to_one",
    )
    if daily.site_group.isna().any():
        raise ValueError("The alias registry is missing station IDs")
    if daily.duplicated(["date", "station_id"]).any():
        raise ValueError("The audited snapshot has duplicate station-days")

    # Apply the audited source ordering and retain one measurement per site-day.
    # Remove old-test sites and mobile/uncertain IDs before creating either pool.
    window = daily[daily.date.between(FIT_START, VALIDATION_END, inclusive="left")]
    unquarantined = window[window.resolved_partition.ne("test_quarantine")]
    fixed = unquarantined[unquarantined.max_displacement_km.lt(0.1)].copy()
    fixed["priority"] = fixed.sources.map(SOURCE_PRIORITY).fillna(9)
    curated = (
        fixed.sort_values(
            ["date", "site_group", "priority", "hours", "station_id"],
            ascending=[True, True, True, False, True],
        )
        .drop_duplicates(["date", "site_group"])
        .rename(columns={"station_id": "source_station_id", "site_group": "station_id"})
    )
    audit = {
        "snapshot_rows": len(daily),
        "snapshot_station_ids": daily.station_id.nunique(),
        "window_rows": len(window),
        "quarantined_rows": len(window) - len(unquarantined),
        "quarantined_station_ids": sorted(set(window.station_id) - set(unquarantined.station_id)),
        "mobile_or_uncertain_rows": len(unquarantined) - len(fixed),
        "mobile_or_uncertain_station_ids": sorted(set(unquarantined.station_id) - set(fixed.station_id)),
        "redundant_site_days": len(fixed) - len(curated),
        "curated_rows": len(curated),
        "curated_sites": curated.station_id.nunique(),
    }
    return curated, audit


def predict_methods(train, train_meta, validation, configs):
    features = [col for col in train if not col.startswith("idw_")]
    features += [BASELINE, TUNED, idw_name(4, 2, 0.15)]
    relative_features = [
        col for col in features if col not in {"lat", "lon"} and not col.endswith(("_lat", "_lon"))
    ]
    predictions = {}
    for config in configs:
        name, family = config["name"], config["family"]
        if family == "idw":
            predictions[name] = validation[config["column"]].to_numpy()
            continue

        columns = relative_features if name == RELATIVE_MODEL else features
        y = train_meta.actual.to_numpy()
        residual = family.startswith("residual")
        if residual:
            y = y - train[TUNED].to_numpy()
        model = estimator(family, config["params"])
        model.fit(train[columns], y)
        values = model.predict(validation[columns])
        if residual:
            values = validation[TUNED].to_numpy() + config["alpha"] * values
        predictions[name] = np.maximum(0, values)
    return predictions


def run(output):
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")

    daily, audit = prepare_data()
    sites = np.array(sorted(daily.station_id.unique()))
    fit_period = daily[daily.date.lt(VALIDATION_START)]
    validation_period = daily[daily.date.ge(VALIDATION_START)]
    configs = json.loads(POLICY.read_text(encoding="utf-8"))["fixed_models"]
    residual_config = next(config for config in configs if config["name"] == "fixed_residual_hgb")
    configs.append({**residual_config, "name": RELATIVE_MODEL})
    names = [config["name"] for config in configs]
    folds, fold_predictions = [], []
    for fold, (fit_indices, held_indices) in enumerate(
        KFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(sites), start=1,
    ):
        fit_sites = sites[fit_indices]
        held_sites = sites[held_indices]
        fitting = fit_period[fit_period.station_id.isin(fit_sites)]
        targets = validation_period[validation_period.station_id.isin(held_sites)]
        references = validation_period[validation_period.station_id.isin(fit_sites)]
        if set(targets.station_id) & (set(fitting.station_id) | set(references.station_id)):
            raise ValueError("A held-out site appears in fitting or reference data")

        train, train_meta = make_features(fitting, fitting, exclude_colocated=True)
        validation, validation_meta = make_features(targets, references, exclude_colocated=True)
        predictions = predict_methods(train, train_meta, validation, configs)
        folds.append({
            "fold": fold,
            "fit_site_pool": fit_sites.tolist(),
            "held_out_sites": held_sites.tolist(),
            "fit_candidates": len(fitting),
            "fit_rows": len(train_meta),
            "fit_label_sites": sorted(train_meta.station_id.unique()),
            "fit_start": train_meta.date.min(),
            "fit_end": train_meta.date.max(),
            "validation_candidates": len(targets),
            "validation_rows": len(validation_meta),
            "validation_label_sites": sorted(validation_meta.station_id.unique()),
            "validation_start": validation_meta.date.min(),
            "validation_end": validation_meta.date.max(),
        })
        fold_predictions.append(validation_meta.assign(
            fold=fold, nearest_km=validation.nearest_km, reference_count=validation.reference_count,
            **predictions,
        ))
        print(f"Fold {fold}: {len(train_meta)} fit rows, {len(validation_meta)} validation rows", flush=True)

    scored = pd.concat(fold_predictions, ignore_index=True)
    if scored.duplicated(["date", "station_id"]).any():
        raise ValueError("A site-day was scored by more than one fold")

    comparison = pd.DataFrame([
        {"method": name, **metrics(scored, scored[name].to_numpy())} for name in names
    ]).sort_values("mae")
    site_rows = []
    for site, rows in scored.groupby("station_id"):
        for name in names:
            errors = rows[name] - rows.actual
            site_rows.append({
                "site_group": site, "method": name, "n": len(rows),
                "mae": errors.abs().mean(), "rmse": np.sqrt((errors ** 2).mean()),
            })
    manifest = {
        "status": "exploratory",
        "source_snapshot_commit": "98ad95f7f60cd4517e45947a429d92626d238889",
        "inputs_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (DAILY, REGISTRY, POLICY)},
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "fit_from_inclusive": FIT_START,
        "validation_from_inclusive": VALIDATION_START,
        "validation_until_exclusive": VALIDATION_END,
        "seed": SEED,
        "high_station_day_threshold_ugm3": HIGH,
        "audit": audit,
        "folds": folds,
        "model_configs": configs,
        "validation_days": int(scored.date.nunique()),
        "validation_eligible_rows": len(scored),
        "validation_candidate_rows": len(validation_period),
    }

    output.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(site_rows).to_csv(output / "by_site.csv", index=False)
    scored.to_csv(output / "predictions.csv", index=False)
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(comparison[["method", "n", "stations", "mae", "rmse", "high_mae", "station_macro_mae"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "grouped_validation")
    run(parser.parse_args().output)
