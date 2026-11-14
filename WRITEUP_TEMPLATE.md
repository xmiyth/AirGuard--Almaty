# Almaty AirGuard

**Track:** AI + Climate  
**Team:** [Your team members]

## The problem
Outdoor activities in Almaty are sometimes affected by fine particle pollution. This project helps a school sports organizer decide whether to review tomorrow's outdoor plans.

## What we built
An AI model that inputs daily PM2.5 and weather conditions and outputs a next-day PM2.5 estimate (µg/m³). A Gradio app demonstrates predictions and a school-sports message.

## What we added
A school-focused Almaty demonstration and a comparison of a simple and a nonlinear regression model. [Describe any additional original test you completed.]

## How it works
Data: Open-Meteo CAMS Global **modelled PM2.5** and historical weather in Almaty (NOT ground observations). [Enter date range and training/testing row counts.]  
Models: persistence baseline; linear regression; random forest. Train on older 80% of dates and evaluate on newer 20%.

## Results
[Copy actual numbers from `data/results.json`, don't guess.]
- Baseline MAE: **__ µg/m³**
- Linear regression MAE: **__ µg/m³**
- Random forest MAE: **__ µg/m³**
- Our strongest model was **__** and [did/did not] outperform persistence.
- A failure case occurred on **__**, with actual **__** and predicted **__** µg/m³. [Explain a plausible cause without asserting certainty.]

## What we'd do next
Compare estimates against real Almaty ground monitoring data, and evaluate school warning false positives/negatives separately. This project is a research prototype, not an official air quality advisory.

## Sources
Open-Meteo Air Quality API; Open-Meteo Historical Weather API; scikit-learn; Gradio; OpenAI ChatGPT (and other tools actually used).
