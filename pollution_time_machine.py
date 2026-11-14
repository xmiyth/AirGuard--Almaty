"""Pollution Time Machine for Almaty AirGuard.

Shows *historical measured daily station averages*, never forecasts.
Depends on the existing station_map.daily_stations() and its local dataset.
"""
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from station_map import STATION_FILE, daily_stations

MAP_CENTER = {"lat": 43.2389, "lon": 76.8897}
PM25_SCALE_MAX = 75  # Consistent scale on every day; higher values clamp to top color.
MIN_STATIONS = 3


@lru_cache(maxsize=2)
def _load_station_days(file_mtime):
    df = daily_stations().copy()
    df["date"] = df["date"].astype(str)
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25", "lat", "lon"])
    days = (df.groupby("date")["station_id"].nunique().sort_index())
    valid_dates = days[days >= MIN_STATIONS].index.tolist()
    if not valid_dates:
        raise ValueError("No dates with at least three eligible monitoring stations.")
    return df, valid_dates


def dataset():
    if not Path(STATION_FILE).exists():
        raise FileNotFoundError("Station data not found. Run: .\\.venv\\Scripts\\python.exe station_map.py download")
    return _load_station_days(Path(STATION_FILE).stat().st_mtime_ns)


def plot_for_date(date, data):
    chosen = data[data["date"] == date].copy()
    fig = go.Figure()
    hover = [
        f"<b>{name}</b><br>Measured daily PM2.5: {pm:.1f} µg/m³"
        f"<br>{date}<br>{hours} hourly reports"
        for name, pm, hours in zip(chosen["station_name"], chosen["pm25"], chosen["hours"])
    ]
    fig.add_trace(go.Scattermap(
        lat=chosen["lat"], lon=chosen["lon"], mode="markers",
        text=hover, hoverinfo="text", name="Measured daily averages",
        marker=dict(size=12, color=chosen["pm25"], colorscale="RdYlGn_r",
                    cmin=0, cmax=PM25_SCALE_MAX, showscale=True,
                    colorbar=dict(title="PM2.5", ticksuffix=" µg/m³")),
    ))
    fig.update_layout(
        map=dict(style="open-street-map", center=MAP_CENTER, zoom=10),
        height=510, margin=dict(t=4, b=4, l=0, r=0),
        showlegend=False, uirevision="airguard-time-machine",
    )
    return fig


def render_day(index):
    try:
        df, dates = dataset()
        i = min(max(int(index), 0), len(dates) - 1)
        date = dates[i]
        chosen = df[df.date == date]
        average = chosen.pm25.mean()  # Station-average, NOT population or citywide exposure.
        summary = (f"### {date} — Historical measured PM2.5\n"
                   f"**{len(chosen)} stations** · station mean **{average:.1f} µg/m³** · "
                   f"range **{chosen.pm25.min():.1f}–{chosen.pm25.max():.1f} µg/m³**\n\n"
                   "*Measurements are daily means from reporting stations, not a full-city "
                   "measurement, a live feed, or an AI forecast. The number of reporting "
                   "stations changes by date. Marker colors use the same scale across dates.*")
        table = (chosen.sort_values("pm25", ascending=False)
                 [["station_name", "pm25", "hours"]]
                 .rename(columns={"station_name":"Station", "pm25":"PM2.5 (µg/m³)", "hours":"Hourly reports"})
                 .round({"PM2.5 (µg/m³)": 1}).head(25))
        return summary, plot_for_date(date, df), table
    except Exception as exc:
        return f"**Could not load time machine:** {exc}", None, pd.DataFrame()


def dataset_description():
    try:
        _, dates = dataset()
        return (f"Explore **{len(dates):,} dates**, from **{dates[0]}** to **{dates[-1]}**. "
                "Move the slider to change days. This uses historical monitoring-station readings.")
    except Exception as exc:
        return f"Unable to load available dates: {exc}"


def end_index():
    try:
        return len(dataset()[1]) - 1
    except Exception:
        return 0


def build_time_machine():
    """Call INSIDE an existing `with gr.Tab(...):` block, or a Gradio Blocks app."""
    gr.Markdown("## 🕰️ Pollution Time Machine\nTravel through Almaty's *historical measured* air pollution, day by day.")
    gr.Markdown(dataset_description())
    last = end_index()
    with gr.Row():
        back = gr.Button("← Previous day", size="sm")
        forward = gr.Button("Next day →", size="sm")
        latest = gr.Button("Latest available", size="sm")
    slider = gr.Slider(minimum=0, maximum=max(1,last), step=1, value=last,
                       label="Timeline (drag to change date)", interactive=True)
    summary = gr.Markdown()
    map_plot = gr.Plot(label="Daily station measurements")
    table = gr.Dataframe(label="Highest-reading stations on selected day", interactive=False)
    slider.change(render_day, inputs=slider, outputs=[summary,map_plot,table], show_progress="minimal")
    back.click(lambda i: max(0,int(i)-1), inputs=slider, outputs=slider)
    forward.click(lambda i: min(last,int(i)+1), inputs=slider, outputs=slider)
    latest.click(lambda: last, outputs=slider)
    gr.Markdown("**Interpretation note:** Stations may appear/disappear between dates. "
                "Do not interpret changes in the average of all reporting stations as a "
                "true citywide trend without controlling for station coverage. "
                "No model is trained or changed by this tool.")
    # Loading the tab avoids a large startup render; can also refresh the map via the slider.
    return slider, summary, map_plot, table


if __name__ == "__main__":
    with gr.Blocks(title="AirGuard — Pollution Time Machine") as demo:
        slider, summary, map_plot, table = build_time_machine()
        demo.load(lambda: render_day(end_index()), outputs=[summary, map_plot, table])
    demo.launch(server_port=7861)
