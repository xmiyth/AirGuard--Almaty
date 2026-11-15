"""Presentation-only basemaps shared by the explorer and historical viewer."""


def basemap_style(name='Dark'):
    """Inline raster styles avoid remote style, glyph and sprite dependencies.

    Tiles still require internet. A matching background keeps missing tiles
    legible while the local observation/estimate layers remain visible.
    """
    dark = name == 'Dark'
    return {
        'version': 8,
        'sources': {'basemap': {
            'type': 'raster', 'tileSize': 256, 'maxzoom': 19,
            'tiles': ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            'attribution': '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
        }},
        # Needed only when the user enables station-name labels.
        'glyphs': 'https://fonts.openmaptiles.org/{fontstack}/{range}.pbf',
        'layers': [
            {'id': 'background', 'type': 'background',
             'paint': {'background-color': '#0c1725' if dark else '#e4e9e8'}},
            {'id': 'basemap', 'type': 'raster', 'source': 'basemap',
             # Invert luminance on the basemap ONLY; pollution colors and
             # measurement values remain unchanged. No API key is required.
             'paint': {'raster-fade-duration': 0, 'raster-saturation': -1 if dark else 0,
                       'raster-brightness-min': .75 if dark else 0,
                       'raster-brightness-max': .08 if dark else 1}},
        ],
    }


def apply_basemap(fig, name='Dark'):
    """Keep image overlays inside the style during Plotly.react style changes.

    Plotly's separate map.layers cache can reference removed image layers after
    a style replacement. Including them in the style makes rebuilding atomic.
    """
    style = basemap_style(name)
    for index, layer in enumerate(fig.layout.map.layers or []):
        if layer.sourcetype != 'image':
            raise ValueError('Expected a presentation image overlay.')
        ident = f'pollution-{index}'
        style['sources'][ident] = {'type': 'image', 'url': layer.source,
                                  'coordinates': [list(point) for point in layer.coordinates]}
        style['layers'].append({'id': ident, 'type': 'raster', 'source': ident,
                                'paint': {'raster-opacity': 1, 'raster-fade-duration': 0}})
    fig.update_layout(map_style=style, map_layers=[])
    return fig
