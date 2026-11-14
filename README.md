# Almaty AirGuard — Veritas AI Hackathon 2026

**Track:** AI + Climate → next-day air-quality forecast (4.3.2)

An ML-based predictor of the **following day's average PM2.5** in Almaty. Built in Python with a Gradio UI. The model receives today's daily PM2.5, yesterday's PM2.5, weather variables, and season and estimates tomorrow's PM2.5 (µg/m³).

## Run in VS Code (Windows)

1. Install Python 3.11+ from python.org if needed.
2. Extract this ZIP, then open its `almaty-airguard` folder in VS Code.
3. Open **Terminal → New Terminal**, and run:

```powershell
python -m pip install -r requirements.txt
python airguard.py download
python airguard.py train
python app.py
```

4. Open the local `http://127.0.0.1:7860` URL printed by Gradio. Running the app does not make it publicly accessible; record a screen demo, or deploy separately.

On some Windows systems, use `py` instead of `python`.

## What to demonstrate

- Forecasts from entered weather + PM2.5 values, with a date label.
- Held-out chronologically separated test data.
- Persistence baseline (tomorrow's PM2.5 equals today's) vs linear regression and random forest, all in **MAE µg/m³**.
- One of the five highest-error days from `data/results.json`, describing where prediction fails.
- Your own original addition: an Almaty schools outdoor-activity message at a demonstrative 35 µg/m³ daily-PM2.5 threshold. Evaluate its false warnings and missed warnings if you develop it further.

## Scientific validity and limitations

- **PM2.5 training target** is *CAMS Global model output* delivered by Open-Meteo, NOT direct sensor/ground observations. It is intended as a working prototype and is limited by the underlying atmospheric-model estimates and resolution. Do not claim local monitoring-station accuracy.
- Historical weather comes from Open-Meteo archive (model/reanalysis data).
- Training features for day D predict target day D+1. Data are split chronologically 80/20 without shuffling. Target future values aren't included as features.
- Data are fetched up to seven days before the current day by default to limit missing data. The app preloads the last available historical day. **The prefilled data may be old.** To forecast tomorrow, enter today's real measurements, or later add a live-input feature. Do not label a historical-day forecast 'tomorrow'.
- This model is compared to naive persistence; there is no guarantee it improves upon it.
- It is a demo and not a regulatory air-quality forecast, live health warning, or medical advice. The threshold is for a demo decision, not an official AQI conversion.
- Check applicable data attribution and terms when deploying commercially.

## File map

- `airguard.py`: download daily data; clean; train and evaluate models; save results.
- `app.py`: interactive Gradio demo.
- `data/daily.csv`: created when data are downloaded.
- `data/results.json`, `data/test_predictions.csv`, `data/model.joblib`: created after training.

## Sources and tools used

- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Gradio](https://www.gradio.app/)
- AI coding assistance: OpenAI ChatGPT (state your team's other actual AI tools when submitting).

## Hackathon submission checklist

- [ ] Run the real dataset download and training; keep actual results.
- [ ] Check the app starts and predicts from at least one input.
- [ ] Inspect a wrong test prediction; explain limitations.
- [ ] Publish code to a public/readable GitHub repo and test link in incognito.
- [ ] Prepare ~1 page write-up with model, real test MAE, baseline, source credits, user, improvement and limitation.
- [ ] Optionally record <2-minute demonstration.
- [ ] Submit code link, write-up link and team details by deadline.
