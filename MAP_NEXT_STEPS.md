# AirGuard Pollution Map — measured station layer (Phase 1)

This extension adds real measured station PM2.5 values across Almaty to your existing trained AirGuard forecasting app.

## Install and download (PowerShell, project directory)

```powershell
$env:TEMP="D:\pip-temp"
$env:TMP="D:\pip-temp"
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe station_map.py download
.\.venv\Scripts\python.exe station_map.py check
.\.venv\Scripts\python.exe app.py
```

Click **Pollution Map — measured stations**, then **Load latest station map**.

## Scientific limitations

- The dots are station locations with at least 12 valid hourly PM2.5 readings on the selected local Almaty date. These are *daily averages*, not live current air quality.
- The colors show measured relative concentrations, not an official AQI risk classification. The map background uses external OpenStreetMap tiles.
- This is NOT a machine-learned interpolation map yet. Do not claim it predicts pollution between sensors or for every neighborhood.
- AirData.kz aggregates quality-checked data from official and low-cost stations. Different sensors can have different calibration and reporting characteristics.
- Data source and methodology: https://github.com/qazybekb/AirDatakz-OpenData

## Next step to make it an actual novel AI research project

Train a *spatial interpolation* model using many dates of station readings. Evaluate on stations completely held out from training; compare against inverse-distance weighted interpolation and show MAE across held-out stations. Show continuous predictions ONLY after confirming spatial holdout accuracy and displaying the uncertainty/coverage limitations. Never train and test on the same sensor locations.
