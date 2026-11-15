"""AirGuard presentation layer: read-only observations and existing model inference.

Run with the existing D: virtual environment. No training, downloads or writes
to scientific artifacts occur here. Styling lives in ui_theme.css.
"""
import json
import base64
import re
from html import escape
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import Delaunay
from airguard import DATA, predict_from_inputs, read_daily
from station_map import daily_stations, map_figure
from spatial_ai import latest_map, results_summary
from pollution_time_machine import build_time_machine, render_day, end_index
from ui_maps import basemap_style, apply_basemap

RESULTS = DATA/'results.json'
COLORS = [(0, (255, 247, 188)), (.25, (254, 196, 79)),
          (.5, (244, 109, 67)), (.75, (190, 36, 87)), (1, (83, 35, 112))]
COLOR_SCALE = [[p, f'rgb{rgb}'] for p, rgb in COLORS]
CSS = Path(__file__).with_name('ui_theme.css').read_text(encoding='utf-8')
PANEL = '#101d2d'
ACCENT = '#8cebd7'
THEME_VALUES = dict(body_background_fill='#09111d', body_text_color='#ecf3fa',
    body_text_color_subdued='#a8b9cb', background_fill_primary=PANEL,
    background_fill_secondary='#152638', border_color_primary='#2b4054',
    block_background_fill=PANEL, block_border_color='#2b4054',
    block_label_text_color='#c3d5e4', block_title_text_color='#edf5fc',
    block_info_text_color='#a3b9cb', block_label_background_fill='transparent',
    input_background_fill='#0c1828', input_background_fill_focus='#102335',
    input_border_color='#354d63', input_border_color_focus='#8cebd7',
    input_placeholder_color='#8ea6ba', button_primary_background_fill='#8cebd7',
    button_primary_text_color='#092520', button_primary_border_color='#8cebd7',
    button_secondary_background_fill='#172b3e', button_secondary_text_color='#d5e8f5',
    button_secondary_border_color='#35546a', checkbox_label_background_fill='#102030',
    checkbox_label_background_fill_selected='#193f40', checkbox_label_text_color='#b9cbdc',
    checkbox_label_text_color_selected='#a6f4df', checkbox_label_border_color='#354d63',
    checkbox_label_border_color_selected='#5da395', table_even_background_fill='#112132',
    table_odd_background_fill='#0e1b2b', table_border_color='#30465a',
    slider_color='#8cebd7', loader_color='#8cebd7')
THEME = gr.themes.Base(primary_hue='teal', neutral_hue='slate',
    font=['Segoe UI', 'system-ui', 'sans-serif']).set(
        **THEME_VALUES, **{k+'_dark': v for k, v in THEME_VALUES.items()},
        block_radius='16px', input_radius='10px', button_large_radius='10px')
# Native Gradio handles navigation; the small JS hook repairs plot sizing and
# reports basemap errors without changing data or making backend requests.
UI_JS = """() => {
  document.documentElement.classList.add('dark');
  if (window.airguardTooltipsReady) return;
  window.airguardTooltipsReady = true;
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') document.querySelectorAll('.tip-wrap').forEach(el => el.classList.add('dismissed'));
  });
  document.addEventListener('focusin', event => event.target.closest?.('.tip-wrap')?.classList.remove('dismissed'));
  document.addEventListener('pointerover', event => event.target.closest?.('.tip-wrap')?.classList.remove('dismissed'));
  const plots = new WeakSet(), maps = new WeakSet();
  const resize = new ResizeObserver(entries => {
    for (const {target, contentRect} of entries) {
      if (contentRect.width && contentRect.height && target._fullLayout && window.Plotly)
        window.Plotly.Plots.resize(target).catch(() => {});
    }
  });
  const bind = () => document.querySelectorAll('[data-testid="plotly"]').forEach(plot => {
    if (plots.has(plot) || !plot.on) return;
    plots.add(plot);
    resize.observe(plot);
    const bindMap = () => {
      const map = plot._fullLayout?.map?._subplot?.map;
      if (!map || maps.has(map)) return;
      maps.add(map);
      // MapLibre must remeasure its canvas after a hidden tab becomes visible.
      requestAnimationFrame(() => map.resize());
      if (!plot.closest('#pollution-map')) return;
      const status = () => document.getElementById('map-network-status');
      map.on('error', () => {
        if (status()) status().textContent = 'Some map tiles or labels could not load. Check your connection, switch basemap, or refresh the map. Local pollution layers remain available.';
      });
      map.on('idle', () => {
        if (map.areTilesLoaded() && status()) status().textContent = '';
      });
    };
    plot.on('plotly_afterplot', bindMap);
    bindMap();
  });
  let scheduled = false;
  new MutationObserver(() => {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { scheduled = false; bind(); });
  }).observe(document.querySelector('.gradio-container') || document.body, {childList: true, subtree: true});
  bind();
}"""


def file_stamp(name):
    path = DATA / name
    return path.stat().st_mtime_ns if path.exists() else 0


@lru_cache(maxsize=1)
def station_snapshot(stamp):
    """Cache only in memory; invalidate when the local file changes."""
    return daily_stations()


def measured_day():
    df = station_snapshot(file_stamp('almaty_stations_pm25.csv.gz'))
    counts = df.groupby('date').station_id.nunique()
    good = counts[counts >= 3]
    if good.empty:
        raise ValueError('No day has at least three reporting stations.')
    day = good.index[-1]
    return day, df[df.date == day].copy()


def chart_style(fig, height=300):
    fig.update_layout(template='plotly_dark', paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=height, font=dict(family='Segoe UI, sans-serif', size=12, color='#b7cada'),
        margin=dict(l=12, r=16, t=22, b=40),
        hoverlabel=dict(bgcolor='#1d3347', bordercolor='#567286', font_color='#edf5fc'),
        legend=dict(orientation='h', y=1.12, x=0, font_size=11),
        xaxis=dict(gridcolor='#24384a', zeroline=False, fixedrange=False),
        yaxis=dict(gridcolor='#24384a', zeroline=False, fixedrange=False))
    return fig


def empty_chart(message='Data will appear here', height=280):
    fig = chart_style(go.Figure(), height)
    fig.add_annotation(text=escape(message), showarrow=False, x=.5, y=.5, xref='paper', yref='paper',
                       font=dict(color='#94adbf', size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def style_map(fig, estimated=False):
    """Presentation only: color existing outputs, without changing their values."""
    trace = fig.data[0] if fig.data else None
    if trace is None:
        return fig
    values = np.asarray(trace.marker.color, dtype=float)
    ceiling = max(100, float(np.ceil(np.max(values) / 50) * 50))
    trace.marker.update(colorscale=COLOR_SCALE, cmin=0, cmax=ceiling, showscale=True,
        colorbar=dict(title=dict(text='PM2.5<br>µg/m³'), thickness=16, len=.75,
                      tickformat='.0f', outlinewidth=0))
    trace.marker.colorbar.update(orientation='h', x=.5, xanchor='center', y=-.06,
        yanchor='top', len=.8, thickness=10, title=dict(text='PM2.5 · µg/m³', side='top'),
        tickfont=dict(color='#c3d6e5', size=11), title_font=dict(color='#c3d6e5', size=11))
    fig.update_layout(template='plotly_dark', height=570, paper_bgcolor='#0c1725',
        font=dict(family='Segoe UI, sans-serif', color='#d6e5f0'),
        hoverlabel=dict(bgcolor='#152c3f', bordercolor='#638397', font=dict(size=12, color='#f3f8fc')),
        margin=dict(l=0, r=0, t=0, b=80), showlegend=False,
        modebar_add=['zoomInMap', 'zoomOutMap', 'resetViewMap'],
        map=dict(style=basemap_style()), uirevision='airguard-map',
        clickmode='event+select')
    if estimated:
        fig.data[1].name = 'Measured reference stations'
        fig.data[1].marker.update(size=9, color='#173a4a')
    else:
        trace.name = 'Measured station averages'
        trace.marker.size = 10
    return fig


def continuous_layer(fig):
    """Rasterize existing grid estimates; transparent outside supported coverage.

    Interpolation is display smoothing, not a new fitted spatial model. Keep
    holes in the grid, and enforce the original hull and 3 km reference limit.
    Use Web Mercator pixel spacing so the image aligns with the map's projection.
    """
    estimates, stations = fig.data[:2]
    lat, lon = np.asarray(estimates.lat, float), np.asarray(estimates.lon, float)
    ys, xs = np.unique(lat), np.unique(lon)
    if min(len(xs), len(ys)) < 2:
        return False
    grid = np.full((len(ys), len(xs)), np.nan)
    grid[np.searchsorted(ys, lat), np.searchsorted(xs, lon)] = np.asarray(estimates.marker.color, float)
    px = np.linspace(xs[0], xs[-1], 320)
    mercator = np.log(np.tan(np.pi / 4 + np.radians([ys[-1], ys[0]]) / 2))
    py = np.degrees(2 * np.arctan(np.exp(np.linspace(*mercator, 320))) - np.pi / 2)
    xx, yy = np.meshgrid(px, py)
    pixels = np.c_[yy.ravel(), xx.ravel()]
    surface = RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=np.nan)(pixels)
    refs = np.c_[np.asarray(stations.lat, float), np.asarray(stations.lon, float)]
    supported = Delaunay(refs).find_simplex(pixels) >= 0
    a, o = np.radians(pixels[:, 0]), np.radians(pixels[:, 1])
    nearest = np.full(len(pixels), np.inf)
    for ra, ro in np.radians(refs):
        distance = 12742 * np.arcsin(np.sqrt(np.clip(
            np.sin((a-ra)/2)**2 + np.cos(a)*np.cos(ra)*np.sin((o-ro)/2)**2, 0, 1)))
        nearest = np.minimum(nearest, distance)
    supported &= (nearest <= 3) & np.isfinite(surface)
    if not supported.any():
        return False
    scaled = np.clip(np.nan_to_num(surface) / estimates.marker.cmax, 0, 1)
    rgb = np.column_stack([np.interp(scaled, [p for p, _ in COLORS], [c[i] for _, c in COLORS])
                           for i in range(3)])
    rgba = np.c_[rgb, np.where(supported, 210, 0)].astype(np.uint8).reshape(320, 320, 4)
    buffer = BytesIO()
    Image.fromarray(rgba).save(buffer, format='PNG')
    source = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    fig.update_layout(map_layers=[dict(sourcetype='image', source=source, type='raster',
        coordinates=[[xs[0], ys[-1]], [xs[-1], ys[-1]], [xs[-1], ys[0]], [xs[0], ys[0]]],
        below='traces', opacity=1)])
    # Keep original sampled estimates as hover targets, plus their colorbar.
    estimates.marker.update(size=12, opacity=.01)
    estimates.showlegend = False
    return True


def station_hover(trace, rows, day, reference=False):
    trace.customdata = [[escape(str(r.station_name)), float(r.pm25), str(day), str(r.station_id)]
                        for r in rows.itertuples()]
    label = 'MEASURED REFERENCE' if reference else 'MEASURED STATION'
    trace.hovertemplate = ('<b>%{customdata[0]}</b><br>' + label +
        '<br><b>%{customdata[1]:.1f} µg/m³</b> PM2.5<br>%{customdata[2]} · local-day average<extra></extra>')
    trace.name = 'Measured reference stations' if reference else 'Measured station averages'
    trace.selected = dict(marker=dict(size=16, opacity=1))
    trace.unselected = dict(marker=dict(opacity=.45))


def marker_halo(fig, trace):
    """A separate non-hoverable ring; Scattermap has no marker.line property."""
    halo = go.Scattermap(lat=trace.lat, lon=trace.lon, mode='markers',
        marker=dict(size=trace.marker.size+4, color='#eaf6ff', opacity=.9),
        hoverinfo='skip', showlegend=False, name='Station outlines')
    original = tuple(fig.data)
    fig.add_trace(halo)
    index = next(i for i, item in enumerate(original) if item is trace)
    fig.data = original[:index] + (fig.data[-1],) + original[index:]


def safe_map():
    try:
        day, shown = measured_day()
        fig = style_map(map_figure(shown))
        station_hover(fig.data[0], shown, day)
        marker_halo(fig, fig.data[0])
        summary = (f'**Measured observations · {day}** · {len(shown)} reporting stations. '
                   'Daily station averages from AirData.kz; not a citywide exposure estimate or live feed.')
        table = shown.sort_values('pm25', ascending=False)[['station_name', 'pm25', 'date', 'hours']].head(30)
        return summary, fig, table.round({'pm25': 1})
    except Exception as exc:
        return f'Measured map unavailable: {exc}', empty_chart('Measured map unavailable · try Refresh map', 570), pd.DataFrame()


@lru_cache(maxsize=1)
def spatial_snapshot(data_stamp, model_stamp):
    summary, fig = latest_map()
    day = re.search(r'\d{4}-\d{2}-\d{2}', summary).group()
    artifact = joblib.load(DATA/'spatial_model.joblib')
    df = station_snapshot(data_stamp)
    refs = df[(df.date == day) & df.station_id.astype(str).isin(artifact['reference_site_ids'])]
    assert len(refs) == len(fig.data[1].lat)
    assert np.allclose(refs[['lat','lon']].to_numpy(), np.c_[fig.data[1].lat, fig.data[1].lon])
    style_map(fig, estimated=True)
    estimates = fig.data[0]
    estimates.customdata = [[day] for _ in estimates.lat]
    estimates.hovertemplate = ('<b>Interpolated location</b><br>ESTIMATED · not measured'
        '<br><b>%{marker.color:.1f} µg/m³</b> PM2.5<br>%{customdata[0]} · same-day estimate'
        '<br>%{lat:.4f}° N, %{lon:.4f}° E<extra></extra>')
    station_hover(fig.data[1], refs, day, reference=True)
    if not continuous_layer(fig):
        summary += '\n\nCoverage is too sparse for a continuous layer; showing supported estimate points.'
    marker_halo(fig, fig.data[1])
    return summary, fig


def safe_spatial():
    try:
        summary, fig = spatial_snapshot(file_stamp('almaty_stations_pm25.csv.gz'), file_stamp('spatial_model.joblib'))
        return summary, go.Figure(fig)  # Per-request copy; never mutate cached views.
    except Exception as exc:
        return f'Estimated map unavailable: {exc}. Choose Measured stations to view observations.', empty_chart('Estimate unavailable · choose Measured stations', 570)


def load_explorer(mode, basemap='Dark', labels=False):
    if mode == 'Measured stations':
        summary, fig, table = safe_map()
    else:
        summary, fig = safe_spatial()
        table = pd.DataFrame()
    if fig is not None:
        apply_basemap(fig, basemap)
        for trace in fig.data:
            if trace.name in ('Measured station averages', 'Measured reference stations'):
                trace.mode = 'markers+text' if labels else 'markers'
                trace.text = [str(row[0]) for row in trace.customdata]
                trace.textposition = 'top right'
                trace.textfont = dict(size=11, color='#edf6ff' if basemap == 'Dark' else '#153347')
    return summary, fig, table


def forecasting(pm25, yesterday, temp, wind, rain, humidity, observation_date):
    try:
        d = pd.Timestamp(observation_date)
        nextday = (d + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        vals = [pm25, yesterday, temp, wind, rain, humidity]
        if any(v is None for v in vals) or pm25 < 0 or yesterday < 0 or wind < 0 or rain < 0 or not 0 <= humidity <= 100:
            return 'Enter valid readings (PM2.5/wind/rain >= 0 and humidity 0–100%).'
        value = predict_from_inputs(*vals, forecast_date=observation_date)
        msg = ('Above 35 µg/m³: consider reviewing outdoor activity plans.' if value > 35 else
               'At or below 35 µg/m³: still check local official air-quality guidance.')
        return (f'### Forecast for {nextday}: {value:.1f} µg/m³ PM2.5\n\n'
                f'{msg}\n\n**Prototype warning:** this is a statistical estimate, not live measured air quality, '
                'medical guidance, or a validated health alert. The 35 µg/m³ cutoff is a demonstration threshold.')
    except Exception as e:
        return f'Unable to predict: {e}'


def latest_defaults():
    try:
        d = read_daily().dropna(subset=['pm25','temperature','wind','rain','humidity'])
        a, b = d.iloc[-1], d.iloc[-2]
        return [round(float(a.pm25),1), round(float(b.pm25),1), round(float(a.temperature),1),
                round(float(a.wind),1), round(float(a.rain),1), round(float(a.humidity),1),
                a.date.strftime('%Y-%m-%d')]
    except Exception:
        return [None,None,None,None,None,None,'2026-11-13']


def results_text():
    if not RESULTS.exists():
        return 'Train the model first to see real held-out test results.'
    r = json.loads(RESULTS.read_text())
    return (f"**Test dates:** {r['test_start']} to {r['test_end']} ({r['test_count']} days)\n\n"
            f"**Persistence baseline MAE:** {r['scores']['persistence_baseline']:.2f} µg/m³\n\n"
            f"**Linear regression MAE:** {r['scores']['linear_regression']:.2f} µg/m³\n\n"
            f"**Random forest MAE:** {r['scores']['random_forest']:.2f} µg/m³\n\n"
            f"**Selected model:** {r['selected_model']}\n\n"
            f"**Beats persistence baseline:** {'Yes' if r['beats_baseline'] else 'No'}")


def preview():
    path = DATA/'test_predictions.csv'
    if path.exists():
        return pd.read_csv(path).tail(30)
    return pd.DataFrame()


def metric_card(label, value, unit, note, explanation):
    ident = 'tip-' + re.sub(r'[^a-z0-9]', '-', label.lower())
    return (f'<article class="metric-card"><div class="metric-label">{escape(label)}'
        f'<span class="tip-wrap"><button class="tip-button" type="button" '
        f'aria-label="About {escape(label)}" aria-describedby="{ident}">i</button>'
        f'<span class="tip-content" role="tooltip" id="{ident}">{escape(explanation)}</span></span></div>'
        f'<div class="metric-value">{escape(str(value))}<small>{escape(unit)}</small></div>'
        f'<div class="metric-note">{escape(note)}</div></article>')


def metric_grid(cards):
    return '<div class="metric-grid">' + ''.join(metric_card(*card) for card in cards) + '</div>'


def notice(text, kind=''):
    return f'<div class="notice {kind}" role="{"alert" if kind == "error" else "note"}">{escape(text)}</div>'


def intro(title, description):
    return f'<div class="section-intro"><h2>{escape(title)}</h2><p>{escape(description)}</p></div>'


def load_overview():
    try:
        day, shown = measured_day()
        cards = metric_grid([
            ('Station median', f'{shown.pm25.median():.1f}', 'µg/m³', 'Measured · latest available day',
             'The middle PM2.5 reading across reporting stations. This is not a population-weighted citywide exposure.'),
            ('Highest reading', f'{shown.pm25.max():.1f}', 'µg/m³', 'Highest measured station average',
             'The largest daily average at a reporting station. A local reading does not represent the whole city.'),
            ('Reporting stations', str(len(shown)), 'stations', 'At least 12 valid samples per day',
             'Distinct reporting station IDs. Some provider IDs may represent the same physical instrument.'),
            ('Lowest reading', f'{shown.pm25.min():.1f}', 'µg/m³', 'Lowest measured station average',
             'The smallest daily average among reporting stations. Low-cost sensors have measurement uncertainty.')])
        df = station_snapshot(file_stamp('almaty_stations_pm25.csv.gz'))
        start = pd.Timestamp(day) - pd.Timedelta(days=29)
        recent = df[(df.date >= start.strftime('%Y-%m-%d')) & (df.date <= day)]
        daily = recent.groupby('date').pm25.agg(median='median', low=lambda x:x.quantile(.25),
                                               high=lambda x:x.quantile(.75), count='size')
        # Plain ISO date lists plus an explicit date axis prevent numeric/time
        # inference across Plotly/Gradio versions. Missing calendar days stay gaps.
        days = pd.date_range(start, day).strftime('%Y-%m-%d').tolist()
        daily = daily.reindex(days)
        trend = go.Figure()
        trend.add_trace(go.Scatter(x=days, y=daily.high, mode='lines', line_width=0,
                                  hoverinfo='skip', showlegend=False, connectgaps=False))
        trend.add_trace(go.Scatter(x=days, y=daily.low, mode='lines', line_width=0,
            fill='tonexty', fillcolor='rgba(110,231,203,.12)', name='Middle 50% of stations',
            hoverinfo='skip', connectgaps=False))
        trend.add_trace(go.Scatter(x=days, y=daily['median'], mode='lines',
            line=dict(color=ACCENT, width=3), name='Station median', connectgaps=False,
            customdata=daily[['count']].to_numpy(),
            hovertemplate='<b>%{x|%d %b %Y}</b><br>Measured station median: %{y:.1f} µg/m³<br>%{customdata[0]:.0f} reporting IDs<extra></extra>'))
        chart_style(trend, 320)
        trend.update_xaxes(type='date', tickformat='%d %b', hoverformat='%d %b %Y',
                           tick0=days[0], dtick=7*86400000, range=[days[0], days[-1]],
                           title_text=f'Local observation date · {start:%Y}', showspikes=True,
                           spikecolor=ACCENT, spikethickness=1)
        trend.update_yaxes(title_text='PM2.5 · µg/m³', rangemode='tozero')
        top = shown.nlargest(8, 'pm25').sort_values('pm25')
        ranking = go.Figure(go.Bar(x=top.pm25, y=[escape(str(x))[:24] for x in top.station_name],
            orientation='h', marker_color='#65bfb1', text=top.pm25.round(1), textposition='outside',
            customdata=[[escape(str(r.station_name)), day] for r in top.itertuples()],
            hovertemplate='<b>%{customdata[0]}</b><br>MEASURED · %{customdata[1]}<br>%{x:.1f} µg/m³<extra></extra>'))
        chart_style(ranking, 320)
        ranking.update_layout(margin=dict(l=8, r=30, t=15, b=40), showlegend=False)
        ranking.update_xaxes(title_text='PM2.5 · µg/m³', range=[0, max(1,float(top.pm25.max())*1.25)])
        ranking.update_yaxes(tickfont_size=10)
        status = (f'<div class="status-line"><span><span class="status-dot"></span>Historical observations</span>'
                  f'<span><strong>{escape(day)}</strong> · Almaty local date</span><span>Source: AirData.kz</span></div>')
        return cards, trend, ranking, status
    except Exception as exc:
        return notice(f'Observations unavailable: {exc}', 'error'), empty_chart(), empty_chart(), notice('No observation date available.')


INITIAL_FORECAST = ('<div class="forecast-card"><span class="badge accent">NEXT-DAY MODEL</span>'
    '<h3>Your forecast will appear here</h3><div class="forecast-value">— <small>µg/m³</small></div>'
    '<p>Review the dated inputs, then calculate a forecast. This uses the existing saved model; '
    'it is separate from the measured station map.</p><div class="forecast-meta">No live data · No calibrated uncertainty interval</div></div>')


def reset_forecast():
    return INITIAL_FORECAST, empty_chart('Your inputs and forecast will appear here', 245)


def forecast_dashboard(pm25, yesterday, temp, wind, rain, humidity, observation_date):
    try:
        values = [pm25, yesterday, temp, wind, rain, humidity]
        if any(v is None or not np.isfinite(float(v)) for v in values):
            raise ValueError('Complete all six numeric fields with finite values.')
        if pm25 < 0 or yesterday < 0 or wind < 0 or rain < 0 or not 0 <= humidity <= 100:
            raise ValueError('Pollution, wind and rain must be nonnegative; humidity must be between 0 and 100%.')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(observation_date)):
            raise ValueError('Use a valid observation date in YYYY-MM-DD format.')
        day = pd.Timestamp(observation_date)
        target = day + pd.Timedelta(days=1)
        value = predict_from_inputs(*values, forecast_date=observation_date)
        if not np.isfinite(value):
            raise ValueError('The saved model did not return a finite prediction.')
        card = (f'<div class="forecast-card forecast-ready" role="status"><span class="badge accent">FORECAST · NOT MEASURED</span>'
                f'<h3>Estimated PM2.5 for {target:%d %b %Y}</h3><div class="forecast-value">{value:.1f} <small>µg/m³</small></div>'
                '<p>A statistical next-day forecast from the saved CAMS/Open-Meteo model. '
                'This is not a prediction for a specific street or a medical recommendation.</p>'
                f'<div class="forecast-meta"><span>Input date: {day:%Y-%m-%d}</span>'
                f'<span>Input PM2.5: {pm25:.1f} µg/m³</span></div></div>')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[day-pd.Timedelta(days=1), day], y=[yesterday, pm25],
            mode='lines+markers', name='Your inputs', line=dict(color=ACCENT, width=2), marker_size=10,
            hovertemplate='%{x|%Y-%m-%d}<br>USER INPUT: %{y:.1f} µg/m³<extra></extra>'))
        fig.add_trace(go.Scatter(x=[day, target], y=[pm25, value], mode='lines',
            line=dict(color='#f3bf7b', width=2, dash='dot'), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=[target], y=[value], mode='markers', name='Model forecast',
            marker=dict(color='#f3bf7b', size=13, symbol='diamond'),
            hovertemplate='%{x|%Y-%m-%d}<br>FORECAST · not measured: %{y:.1f} µg/m³<extra></extra>'))
        chart_style(fig, 245)
        fig.update_yaxes(title_text='PM2.5 · µg/m³', rangemode='tozero')
        fig.update_xaxes(type='date', tickformat='%d %b', dtick=86400000)
        return card, fig
    except Exception as exc:
        return notice(f'Please check your inputs. {exc}', 'error'), empty_chart('No forecast calculated', 245)


def evaluation_dashboard(metric='MAE'):
    try:
        r = json.loads((DATA/'spatial_results.json').read_text(encoding='utf-8'))
        suffix = 'mae' if metric == 'MAE' else 'rmse'
        definition = ('Mean absolute error: average prediction error in µg/m³. Lower is better.' if suffix == 'mae'
                      else 'Root mean squared error: emphasizes larger prediction errors. Lower is better; also in µg/m³.')
        baseline, correction = r['baseline_idw_'+suffix], r['ml_corrected_'+suffix]
        cards = metric_grid([
            ('IDW '+metric, f'{baseline:.3f}', 'µg/m³', 'Saved spatial baseline', definition),
            ('ML correction '+metric, f'{correction:.3f}', 'µg/m³', 'Saved residual model', definition),
            ('Test observations', f"{r['test_examples']:,}", 'station-days', 'Original historical evaluation',
             'One reporting station on one day. Repeated days and nearby stations are correlated.'),
            ('Test station IDs', str(r['test_unique_sites']), 'reporting IDs', 'Original held-out IDs with eligible rows',
             'Distinct IDs do not guarantee distinct physical sensors. Alias findings limit the independence of these scores.')])
        fig = go.Figure(go.Bar(x=[baseline,correction], y=['IDW baseline','ML correction'], orientation='h',
            marker_color=[ACCENT,'#df9ea8'], text=[f'{baseline:.3f}',f'{correction:.3f}'], textposition='outside',
            hovertemplate='%{y}<br>Saved '+metric+': %{x:.3f} µg/m³<extra></extra>'))
        chart_style(fig, 230)
        fig.update_xaxes(title_text=f'{metric} · µg/m³ · lower is better', range=[0,max(baseline,correction)*1.3])
        fig.update_layout(showlegend=False)
        text, table = results_summary()
        return cards, fig, text, table
    except Exception as exc:
        return notice(f'Saved spatial results unavailable: {exc}', 'error'), empty_chart(), 'No saved results to display.', pd.DataFrame()


def forecast_evaluation_chart():
    try:
        r = json.loads(RESULTS.read_text(encoding='utf-8'))
        labels = {'persistence_baseline':'Repeat today’s value', 'linear_regression':'Linear regression', 'random_forest':'Random forest'}
        pairs = list(r['scores'].items())
        fig = go.Figure(go.Bar(x=[score for _,score in pairs], y=[labels.get(name,name) for name,_ in pairs],
            orientation='h', marker_color=['#6d879b' if name!=r['selected_model'] else ACCENT for name,_ in pairs],
            text=[f'{score:.3f}' for _,score in pairs], textposition='outside',
            hovertemplate='%{y}<br>Saved MAE: %{x:.3f} µg/m³<extra></extra>'))
        chart_style(fig, 250)
        fig.update_xaxes(title_text='MAE · µg/m³ · lower is better', range=[0,max(r['scores'].values())*1.3])
        return fig
    except Exception:
        return empty_chart('Saved forecast results unavailable')

def grouped_validation_panel():
    """Read saved exploratory scores only; never fit or select a map model."""
    labels = {
        'previously_selected_tuned_idw': 'Previously tuned IDW',
        'relative_residual_hgb': 'Residual HGB · relative features',
        'fixed_residual_hgb': 'Residual HGB',
        'fixed_residual_rf': 'Residual Random Forest',
        'original_idw_settings': 'Original IDW settings',
        'fixed_random_forest': 'Random Forest',
        'fixed_hist_gradient_boosting': 'HistGradientBoosting',
    }
    try:
        folder = Path(__file__).parent / 'experiments' / 'grouped_validation'
        scores = pd.read_csv(folder / 'comparison.csv').sort_values('mae')
        manifest = json.loads((folder / 'split_manifest.json').read_text(encoding='utf-8'))
        if set(scores.method) != set(labels) or len(scores) != 7:
            raise ValueError('Expected the seven documented spatial methods.')
        table = scores[['method', 'mae', 'rmse']].copy()
        table['method'] = table.method.map(labels)
        table.columns = ['Spatial method', 'MAE · µg/m³', 'RMSE · µg/m³']
        fig = go.Figure()
        for key, color in [('mae', ACCENT), ('rmse', '#93b5f2')]:
            fig.add_trace(go.Bar(x=scores[key].tolist(), y=table['Spatial method'].tolist(),
                orientation='h', name=key.upper(), marker_color=color,
                text=[f'{v:.3f}' for v in scores[key]], textposition='outside', cliponaxis=False,
                hovertemplate='%{y}<br>'+key.upper()+': %{x:.3f} µg/m³<extra></extra>'))
        chart_style(fig, 420)
        fig.update_layout(barmode='group', bargap=.26, margin=dict(l=8, r=50, t=45, b=45))
        fig.update_yaxes(autorange='reversed', automargin=True, tickfont_size=11)
        fig.update_xaxes(title_text='Error · µg/m³ · lower is better',
                         range=[0, float(scores.rmse.max())*1.18])
        end = pd.Timestamp(manifest['validation_until_exclusive']) - pd.Timedelta(days=1)
        summary = (f"**Exploratory grouped validation · {manifest['validation_from_inclusive']} to {end:%Y-%m-%d}**\n\n"
            f"Five folds holding out entire site groups · **{int(scores.n.iloc[0]):,} station-days** · "
            f"**{int(scores.stations.iloc[0])} sites** · identical scored rows across all seven methods.\n\n"
            '**Previously tuned IDW has the lowest MAE and RMSE. ML did not outperform it.** '
            'The closest residual HGB candidate is only 0.0034 µg/m³ worse on MAE; '
            'this is not evidence of a meaningful performance difference.\n\n'
            'These newer results are **separate from the original evaluation below**, not a fresh independent test. '
            'The grouping registry uses the audited snapshot, including validation dates; network coverage changes over time. '
            'The historical test remains quarantined. Future dates and a site roster frozen before collection are needed for confirmation. '
            'The deployed model and interpolation settings are unchanged.')
        return summary, fig, table.round(3)
    except Exception as exc:
        return f'Grouped validation unavailable: {exc}', empty_chart('Saved grouped scores unavailable'), pd.DataFrame()


with gr.Blocks(title='Almaty AirGuard', analytics_enabled=False) as demo:
    gr.HTML('<header class="brandbar"><div class="brand"><div class="brand-mark" aria-hidden="true">'
        '<svg width="25" height="25" viewBox="0 0 25 25" fill="none" stroke="#9af2db" stroke-width="1.8" stroke-linecap="round">'
        '<path d="M3 8h13c5 0 5-5 1-5M3 13h18M3 18h11c5 0 5 5 1 5"/></svg></div>'
        '<div><div class="brand-name">AirGuard<span style="color:#8cebd7">.</span></div>'
        '<div class="brand-sub">Environmental intelligence</div></div></div>'
        '<div class="location">Almaty, Kazakhstan<small>Veritas AI · Research prototype</small></div></header>', js_on_load=None)
    with gr.Tabs() as navigation:
        with gr.Tab('Overview', id='overview'):
            gr.HTML('<section class="hero"><div class="eyebrow">A city in focus</div>'
                '<h1>A clearer view of<br><em>Almaty’s air.</em></h1>'
                '<p>From individual sensors to patterns across the city. Explore what is measured, '
                'understand what is estimated, and see the science behind each view.</p>'
                '<span class="badge">Historical observations · Not a live monitoring service</span></section>', js_on_load=None)
            with gr.Row():
                overview_status = gr.HTML('<div class="status-line">Loading the latest available local observations…</div>', scale=3, js_on_load=None)
                refresh = gr.Button('Refresh local data', size='sm', scale=0, min_width=140)
                open_map = gr.Button('Explore the map →', variant='primary', size='sm', scale=0, min_width=160)
            overview_cards = gr.HTML(metric_grid([(label,'—',unit,'Waiting for observations',tip) for label,unit,tip in [
                ('Station median','µg/m³','Median of reporting station averages, not citywide exposure.'),
                ('Highest reading','µg/m³','Highest reporting station average.'),
                ('Reporting stations','stations','Reporting IDs may include shared physical sensors.'),
                ('Lowest reading','µg/m³','Lowest reporting station average.')]]), js_on_load=None)
            with gr.Row(equal_height=True):
                with gr.Column(scale=3, min_width=320, elem_classes='panel'):
                    gr.Markdown('### The last 30 days\nMeasured station median and the middle 50% of readings.')
                    trend_plot = gr.Plot(empty_chart('Loading station history…', 320), show_label=False, elem_classes='plot-panel')
                with gr.Column(scale=2, min_width=300, elem_classes='panel'):
                    gr.Markdown('### Highest station readings\nEight highest daily averages on the latest available day.')
                    ranking_plot = gr.Plot(empty_chart('Loading measured readings…', 320), show_label=False, elem_classes='plot-panel')
            gr.HTML(notice('These are reporting-station summaries, not population exposure estimates. '
                'The reporting network changes over time. The shaded band shows variation between stations, not forecast uncertainty.'), js_on_load=None)
        with gr.Tab('Map explorer', id='map') as map_tab:
            gr.HTML(intro('See the city, layer by layer.', 'Compare observed station readings with estimates between sensors. Every view is dated and labeled.'), js_on_load=None)
            with gr.Row(equal_height=True):
                map_mode = gr.Radio(['Estimated heatmap','Measured stations'], value='Estimated heatmap', label='Pollution layer', scale=3)
                basemap = gr.Dropdown(['Dark','Street'], value='Dark', label='Basemap', scale=1, min_width=140)
                station_labels = gr.Checkbox(False, label='Station names', scale=1, min_width=130)
                map_btn = gr.Button('Refresh map', variant='primary', scale=0, min_width=135)
            map_summary = gr.Markdown('Open this tab to load the latest supported map. Data is historical, not live.')
            with gr.Group(elem_classes='map-shell'):
                map_plot = gr.Plot(empty_chart('Preparing your map…', 570), show_label=False, elem_id='pollution-map')
            gr.HTML('<div id="map-network-status" class="map-network-status" role="status" aria-live="polite"></div>', js_on_load=None)
            gr.HTML('<div class="map-key"><span><i class="gradient" aria-hidden="true"></i>Heatmap: estimated concentration</span>'
                '<span><i class="dot" aria-hidden="true"></i>Ring markers: measured stations</span>'
                '<span><i class="blank" aria-hidden="true"></i>Blank: no supported estimate</span></div>', js_on_load=None)
            gr.HTML(notice('Higher PM2.5 runs from pale yellow through orange to purple; these colors are not an official AQI. '
                'Hover or tap markers for values and dates; click a station to highlight it (Shift-click to add another). '
                'In the measured view, marker colors show actual station readings. '
                'Use the map’s zoom and reset controls; unsupported areas remain transparent.'), js_on_load=None)
            with gr.Accordion('Coverage, smoothing & measurement limitations', open=False):
                gr.Markdown('The heatmap smooths the **existing prediction grid**, without fitting a new model. '
                    'Pixels outside the station convex hull, farther than **3 km** from a reference, or in grid gaps stay transparent. '
                    'Hover values are the original sampled grid estimates. **Street-level accuracy is not established.**\n\n'
                    'Reference markers are actual daily station averages; some feeds may represent the same physical sensor. '
                    'Full-day averages require the day’s measurements to be available. The map is **not a next-day forecast**. '
                    'Basemap tiles require internet; local data does not refresh from the internet.')
            with gr.Accordion('Measured view: 30 highest station readings', open=False):
                station_table = gr.Dataframe(interactive=False, label='Measured station averages · table available in measured view')
        with gr.Tab('Time Machine', id='time-machine') as time_tab:
            gr.HTML(intro('Travel through Almaty’s air-quality history.',
                'Explore daily measurements from reporting stations. This is historical data, not a forecast.'), js_on_load=None)
            time_slider, time_summary, time_plot, time_table = build_time_machine()
        with gr.Tab('Forecast', id='forecast'):
            gr.HTML(intro('A look at the next day.', 'Use pollution and weather inputs to run the existing statistical forecast model.'), js_on_load=None)
            gr.HTML(notice('Defaults come from the latest historical CAMS/Open-Meteo record, not live station observations. '
                'Review the date and values before calculating. The forecast model is separate from the station map.'), js_on_load=None)
            defaults = latest_defaults()
            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes='panel'):
                    gr.Markdown('### Your inputs')
                    day = gr.Textbox(label='Observation date', value=defaults[6], placeholder='YYYY-MM-DD', info='The forecast is for the next calendar day.')
                    with gr.Row():
                        today = gr.Number(label='PM2.5 · input day', value=defaults[0], info='µg/m³', minimum=0, min_width=140)
                        prev = gr.Number(label='PM2.5 · previous day', value=defaults[1], info='µg/m³', minimum=0, min_width=140)
                    with gr.Row():
                        temp = gr.Number(label='Temperature', value=defaults[2], info='°C', min_width=140)
                        wind = gr.Number(label='Wind speed', value=defaults[3], info='km/h', minimum=0, min_width=140)
                    with gr.Row():
                        rain = gr.Number(label='Rainfall', value=defaults[4], info='mm', minimum=0, min_width=140)
                        humid = gr.Number(label='Relative humidity', value=defaults[5], info='%', minimum=0, maximum=100, min_width=140)
                    forecast_btn = gr.Button('Calculate next-day forecast →', variant='primary')
                    restore_btn = gr.Button('Restore historical inputs', size='sm')
                with gr.Column(scale=1, min_width=300):
                    forecast_card = gr.HTML(INITIAL_FORECAST, js_on_load=None)
                    with gr.Group(elem_classes='panel'):
                        gr.Markdown('### Inputs → forecast')
                        forecast_plot = gr.Plot(empty_chart('Your inputs and forecast will appear here', 245), show_label=False, elem_classes='plot-panel')
                    gr.HTML(notice('No calibrated uncertainty interval is available. This prototype is not an operational air-quality alert or medical recommendation.'), js_on_load=None)
        with gr.Tab('Model insights', id='insights'):
            gr.HTML(intro('The evidence behind the estimates.', 'Explore saved model results in their original units. Lower prediction error is better.'), js_on_load=None)
            with gr.Column(elem_classes='panel'):
                gr.Markdown('### Seven methods, one grouped comparison')
                grouped_text, grouped_chart, grouped_table = grouped_validation_panel()
                gr.Markdown(grouped_text)
                gr.Plot(grouped_chart, show_label=False, elem_classes='plot-panel')
                with gr.Accordion('All seven methods · scores rounded to 3 decimals', open=False):
                    gr.Dataframe(grouped_table, interactive=False, label='Exploratory spatial validation')
                    gr.Markdown('Source: `experiments/grouped_validation/comparison.csv` and `split_manifest.json`. '
                                'Protocol: `experiments/GROUPED_SPATIAL_VALIDATION.md`. HGB = Histogram Gradient Boosting.')
            gr.Markdown('### Original saved evaluations')
            gr.HTML(notice('Historical, exploratory results. The old test set has already been inspected, and some station feeds share sites or possible sensor identities. '
                'These saved scores are not fresh, independent test confirmation.', 'warning'), js_on_load=None)
            metric = gr.Radio(['MAE','RMSE'], value='MAE', label='Spatial error metric')
            initial_cards, initial_chart, initial_details, initial_table = evaluation_dashboard()
            eval_cards = gr.HTML(initial_cards, js_on_load=None)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=300, elem_classes='panel'):
                    gr.Markdown('### Spatial interpolation\nSame-day estimates at held-out station IDs, using later dates.')
                    eval_plot = gr.Plot(initial_chart, show_label=False, elem_classes='plot-panel')
                    gr.Markdown('The saved ML correction did **not** beat the IDW baseline. The map uses the method stored in the existing artifact.')
                with gr.Column(scale=1, min_width=300, elem_classes='panel'):
                    gr.Markdown('### Next-day model\nA separate evaluation on modelled CAMS/Open-Meteo data.')
                    gr.Plot(forecast_evaluation_chart(), show_label=False, elem_classes='plot-panel')
                    gr.Markdown('Different targets and datasets: the errors in these two charts are **not directly comparable**.')
            with gr.Accordion('How to read MAE, RMSE and IDW', open=False):
                gr.Markdown('**MAE — mean absolute error:** the average size of the prediction error. '
                    'An MAE of 6 µg/m³ means predictions differ from targets by 6 µg/m³ on average.\n\n'
                    '**RMSE — root mean squared error:** also measured in µg/m³, but gives larger errors more weight. '
                    'Neither metric guarantees accuracy at a specific location or on a future day.\n\n'
                    '**IDW — inverse-distance weighting:** nearby reporting stations contribute more strongly to an estimate. '
                    'An ML correction tries to adjust that estimate; it does not automatically improve it.')
            with gr.Accordion('Original spatial results & largest saved errors', open=False):
                eval_details = gr.Markdown(initial_details)
                eval_table = gr.Dataframe(value=initial_table, label='Saved correction errors', interactive=False)
            with gr.Accordion('Original next-day model results & saved predictions', open=False):
                gr.Markdown(results_text())
                gr.Dataframe(value=preview(), label='Last 30 saved test cases', interactive=False)
    gr.HTML('<footer class="footer-note"><span>AirGuard · Built for Veritas AI<br>Research prototype. Measurements, interpolation and forecasts are distinct.</span>'
        '<span>Sources: <a href="https://github.com/qazybekb/AirDatakz-OpenData" target="_blank" rel="noopener">AirData.kz</a> · '
        '<a href="https://open-meteo.com/en/docs/air-quality-api" target="_blank" rel="noopener">Open-Meteo</a><br>'
        'Basemaps: © OpenStreetMap contributors</span></footer>', js_on_load=None)
    overview_outputs = [overview_cards, trend_plot, ranking_plot, overview_status]
    demo.load(load_overview, outputs=overview_outputs, show_progress='minimal', concurrency_id='station-read', concurrency_limit=1)
    refresh.click(load_overview, outputs=overview_outputs, show_progress='minimal', concurrency_id='station-read', concurrency_limit=1)
    open_map.click(lambda: gr.Tabs(selected='map'), outputs=navigation, queue=False)
    map_inputs, map_outputs = [map_mode, basemap, station_labels], [map_summary, map_plot, station_table]
    gr.on(triggers=[map_tab.select, map_btn.click, map_mode.change, basemap.change, station_labels.change],
          fn=load_explorer, inputs=map_inputs, outputs=map_outputs, show_progress='minimal',
          show_progress_on=map_plot, concurrency_id='station-read', concurrency_limit=1, trigger_mode='always_last')
    # Render the historical map only when the user opens this tab.
    time_tab.select(fn=render_day, inputs=time_slider,
        outputs=[time_summary, time_plot, time_table], show_progress='minimal',
        concurrency_id='time-read', concurrency_limit=1, trigger_mode='always_last')
    forecast_inputs = [today,prev,temp,wind,rain,humid,day]
    forecast_outputs = [forecast_card,forecast_plot]
    forecast_btn.click(forecast_dashboard, forecast_inputs, forecast_outputs, show_progress='minimal', show_progress_on=forecast_card)
    restore_btn.click(latest_defaults, outputs=forecast_inputs).then(reset_forecast, outputs=forecast_outputs, show_progress='hidden')
    gr.on(triggers=[component.input for component in forecast_inputs], fn=reset_forecast,
          outputs=forecast_outputs, show_progress='hidden', queue=False)
    metric.change(evaluation_dashboard, metric, [eval_cards,eval_plot,eval_details,eval_table], show_progress='minimal')

if __name__ == '__main__':
    demo.queue().launch(theme=THEME, css=CSS, js=UI_JS)
