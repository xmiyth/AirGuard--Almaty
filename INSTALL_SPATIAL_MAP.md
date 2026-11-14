# AirGuard — AI Spatial Map update

This updates your **existing** `D:\AirGuard\almaty-airguard` project. Do not delete `data`, `.venv`, or `airguard.py`.

## Installation

1. Stop your running Gradio server in VS Code with `Ctrl+C`.
2. Extract `airguard-spatial-update.zip` and copy `app.py`, `station_map.py`, and `spatial_ai.py` into `D:\AirGuard\almaty-airguard`. Choose **Replace** for `app.py` and `station_map.py`.
3. In the VS Code PowerShell terminal inside that folder, run:

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir scipy joblib
.\.venv\Scripts\python.exe spatial_ai.py train
```

4. Read the MAE results in the terminal. They are **real only when run against your local AirData.kz file**; no scores are assumed in this update.
5. Run your website:

```powershell
.\.venv\Scripts\python.exe app.py
```

6. Open `http://127.0.0.1:7860`, select **AI Spatial Map — estimated areas**, click **Load spatial estimates**, and then **Show held-out station results**.

## What the model does

- Uses AirData.kz station-days with **12 or more hourly readings**.
- Holds out **20% of qualifying station IDs entirely**, so the model cannot learn their historical readings.
- Trains on the **first 80% of calendar dates** and evaluates on the **latest 20% of dates** at the hidden stations.
- Uses nearby *other* reporting stations' **same-day** PM2.5 to compute inverse-distance-weighted (IDW) interpolation.
- Trains a random forest to correct IDW residual errors from spatial coordinates, day of year, distance, and reference-station patterns.
- Reports IDW versus ML-corrected mean absolute error in µg/m³ on the held-out test set.
- Displays the ML-corrected map **only if the model beats IDW**. Otherwise displays the honest baseline.
- Restricts plotted estimates to within the **convex hull** of reference stations and **3 km** from at least one reporting reference station.

## Scientific limitations

The spatial map is **not tomorrow's neighborhood forecast**. It is a same-day estimate from available sensors. The original next-day forecasting model is still in the **Try the AI** tab and uses a different pollution dataset (CAMS/Open-Meteo estimates). Do not describe these two models as one trained system. Ground observations from low-cost sensors can contain bias or calibration errors, and results may change when selecting a different set of held-out stations.

For the hackathon, show actual model-vs-baseline scores, one failure case, and clearly describe the test design. A negative AI improvement is valid to report.

Data source: https://github.com/qazybekb/AirDatakz-OpenData
