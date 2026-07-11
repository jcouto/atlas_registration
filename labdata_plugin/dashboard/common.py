import base64
import io
import numpy as np
import pandas as pd
import streamlit as st

_ANNOTATED_COLOR = '#d4f5d4'  # light green for brains that have a ProbeTrack


def orientation_axes_safe(orientation='asr'):
    '''``atlas_registration.orientation_axes`` with an 'asr' fallback.'''
    from atlas_registration import orientation_axes
    try:
        return orientation_axes(orientation)
    except Exception:
        return orientation_axes('asr')


def _normalize_with_pct(img, lo_pct=2.0, hi_pct=99.5):
    img = np.asarray(img, dtype=float)
    lo = np.percentile(img, lo_pct)
    hi = np.percentile(img, hi_pct)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def annotation_rgb(slice2d, lookup):
    '''Colour a 2D annotation slice with the atlas region RGB triplets.'''
    slice2d = np.asarray(slice2d)
    ids = np.unique(slice2d)
    palette = np.zeros((len(ids), 3), dtype=np.uint8)
    for i, rid in enumerate(ids):
        palette[i] = lookup['id_to_rgb'].get(int(rid), [0, 0, 0])
    idx = np.searchsorted(ids, slice2d)
    return palette[idx]


def to_base64(arr):
    '''Encode a grayscale or RGB array as a base64 PNG data URL.'''
    from PIL import Image as PILImage
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = _normalize_with_pct(arr)
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype != np.uint8:
        arr = _normalize_with_pct(arr)
    buf = io.BytesIO()
    PILImage.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def altair_image(url, w, h, title='', width=400, height=None):
    import altair as alt
    if height is None:
        height = max(1, int(width * h / w))
    img_df = pd.DataFrame([{'x': 0, 'y': 0, 'x2': w, 'y2': h, 'url': url}])
    return (alt.Chart(img_df).mark_image(aspect=False).encode(
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, w]), axis=None),
        y=alt.Y('y:Q', scale=alt.Scale(domain=[h, 0]), axis=None),
        x2=alt.X2('x2'), y2=alt.Y2('y2'), url='url:N',
        tooltip=alt.value(None))
        .properties(width=width, height=height, title=title).interactive())


def altair_clickable(url, img_w, img_h, width=380, title='', overlay=None,
                     line=None, point_size=45):
    '''Altair image with a transparent grid overlay for click detection.

    ``overlay`` is an optional DataFrame with ``x``/``y`` columns drawn as yellow
    markers (e.g. section-local annotation points); ``line`` is an optional
    DataFrame with ``x``/``y`` columns drawn as a red path (e.g. the fitted
    trajectory projected onto this plane).
    '''
    import altair as alt
    x_sc = alt.Scale(domain=[0, img_w])
    y_sc = alt.Scale(domain=[img_h, 0])
    # cap the click-grid density so reruns stay fast on small (downsampled) images
    step = max(2, min(img_w, img_h) // 50)
    xs = np.arange(0, img_w, step, dtype=float)
    ys = np.arange(0, img_h, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    height = max(150, int(width * img_h / img_w))
    img_layer = (alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': img_w,
                                          'y2': img_h, 'url': url}]))
                 .mark_image(aspect=False)
                 .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                         y=alt.Y('y:Q', scale=y_sc, axis=None),
                         x2='x2:Q', y2='y2:Q', url='url:N',
                         tooltip=alt.value(None)))
    click_sel = alt.selection_point(name='pt_click', on='click',
                                    nearest=True, encodings=['x', 'y'])
    grid_layer = (alt.Chart(pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel()}))
                  .mark_point(opacity=0, size=step * step * 4)
                  .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                          y=alt.Y('y:Q', scale=y_sc, axis=None))
                  .add_params(click_sel))
    layers = [img_layer, grid_layer]
    if line is not None and len(line):
        layers.append(alt.Chart(line).mark_line(
            color='red', strokeWidth=1.5, opacity=0.7).encode(
            x=alt.X('x:Q', scale=x_sc, axis=None),
            y=alt.Y('y:Q', scale=y_sc, axis=None),
            order='order:Q'))
    if overlay is not None and len(overlay):
        layers.append(alt.Chart(overlay).mark_point(
            color='yellow', size=point_size, filled=True, opacity=0.9).encode(
            x=alt.X('x:Q', scale=x_sc, axis=None),
            y=alt.Y('y:Q', scale=y_sc, axis=None)))
    return (alt.layer(*layers)
            .properties(width=width, height=height, title=title).interactive())


def extract_click(event):
    '''Return (x, y) from an Altair on_select event, or None.'''
    sel = (event.selection or {}).get('pt_click', [])
    if sel:
        x, y = sel[0].get('x'), sel[0].get('y')
        if x is not None and y is not None:
            return float(x), float(y)
    return None


def extract_click_y(event, name):
    '''Return the clicked depth (y) from a depth-click selection, or None.'''
    sel = (event.selection or {}).get(name, [])
    if sel and sel[0].get('y') is not None:
        return float(sel[0]['y'])
    return None


def altair_view(url, img_w, img_h, width=300, title='', line=None, marker=None):
    '''Non-interactive image view with an optional projected line and a marker
    (used by the read-only orthogonal projections).'''
    import altair as alt
    x_sc = alt.Scale(domain=[0, img_w])
    y_sc = alt.Scale(domain=[img_h, 0])
    height = max(150, int(width * img_h / img_w))
    layers = [alt.Chart(pd.DataFrame([{'x': 0, 'y': 0, 'x2': img_w, 'y2': img_h,
                                       'url': url}]))
              .mark_image(aspect=False)
              .encode(x=alt.X('x:Q', scale=x_sc, axis=None),
                      y=alt.Y('y:Q', scale=y_sc, axis=None),
                      x2='x2:Q', y2='y2:Q', url='url:N', tooltip=alt.value(None))]
    if line is not None and len(line):
        layers.append(alt.Chart(line).mark_line(color='red', strokeWidth=1.2,
                                                opacity=0.7).encode(
            x=alt.X('x:Q', scale=x_sc, axis=None),
            y=alt.Y('y:Q', scale=y_sc, axis=None), order='order:Q'))
    if marker is not None and len(marker):
        layers.append(alt.Chart(marker).mark_point(
            color='cyan', size=90, filled=True, opacity=0.9,
            shape='diamond').encode(
            x=alt.X('x:Q', scale=x_sc, axis=None),
            y=alt.Y('y:Q', scale=y_sc, axis=None)))
    return (alt.layer(*layers).properties(width=width, height=height, title=title)
            .interactive())


def depth_scale(depth_domain):
    '''A depth y-scale with **zero at the top** (cortex/surface up).'''
    import altair as alt
    lo, hi = depth_domain
    return alt.Scale(domain=[lo, hi], reverse=True)


def region_depth_column(regions, height=520, width=150, depth_domain=None,
                        rules=None, clickable=False, click_name='dclick',
                        title='Regions along track'):
    '''Colored region blocks vs depth (µm), acronym-labelled, **cortex on top**.

    ``depth_domain`` is a ``(lo, hi)`` tuple to share the depth axis with other
    panels; defaults to ``(0, max exit)``.  ``rules`` is a list of
    ``(depth, color)`` horizontal reference lines.  When ``clickable`` a
    transparent y-grid adds a click selection named ``click_name`` (read back
    with :func:`extract_click_y`).
    '''
    import altair as alt
    rows = []
    for _, r in regions.iterrows():
        if r['length_um'] <= 0:
            continue
        rgb = r.get('rgb')
        rgb = [120, 120, 120] if np.ndim(rgb) == 0 else rgb   # None/NaN -> gray
        rows.append(dict(acronym=r['acronym'], name=r.get('name', ''),
                         entry=float(r['entry_um']), exit=float(r['exit_um']),
                         mid=float((r['entry_um'] + r['exit_um']) / 2),
                         color=f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})',
                         x=0.0, x2=1.0))
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({'entry': [0], 'exit': [1]})).mark_rect()
    if depth_domain is None:
        depth_domain = (0.0, float(df['exit'].max()))
    y_scale = depth_scale(depth_domain)
    span = depth_domain[1] - depth_domain[0]
    blocks = alt.Chart(df).mark_rect(stroke='white', strokeWidth=0.3).encode(
        y=alt.Y('entry:Q', scale=y_scale, title='Depth (µm)'), y2='exit:Q',
        x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]), axis=None), x2='x2:Q',
        color=alt.Color('color:N', scale=None, legend=None),
        tooltip=['acronym', 'name', 'entry', 'exit'])
    labels = alt.Chart(df[df['exit'] - df['entry'] >= span * 0.015]).mark_text(
        align='left', dx=4, fontSize=9, color='white').encode(
        y=alt.Y('mid:Q', scale=y_scale), x=alt.value(2), text='acronym:N')
    layers = [blocks, labels]
    if rules:
        rdf = pd.DataFrame([{'d': float(d), 'c': c} for d, c in rules])
        layers.append(alt.Chart(rdf).mark_rule(strokeWidth=2).encode(
            y=alt.Y('d:Q', scale=y_scale),
            color=alt.Color('c:N', scale=None, legend=None)))
    if clickable:
        ys = np.linspace(depth_domain[0], depth_domain[1], 240)
        sel = alt.selection_point(name=click_name, on='click', nearest=True,
                                  encodings=['y'])
        layers.append(alt.Chart(pd.DataFrame({'y': ys, 'x': [0.5] * len(ys)}))
                      .mark_point(opacity=0, size=180)
                      .encode(y=alt.Y('y:Q', scale=y_scale, axis=None),
                              x=alt.X('x:Q', scale=alt.Scale(domain=[0, 1]),
                                      axis=None))
                      .add_params(sel))
    return alt.layer(*layers).properties(height=height, width=width, title=title)


def tab_cache_factory(refresh_key):
    '''Decorator like ``st.cache_data`` wired to a per-tab refresh button.'''
    do_clear = st.session_state.pop(refresh_key, False)

    def cache(func):
        cached = st.cache_data(func)
        if do_clear:
            cached.clear()
        return cached
    return cache


def refresh_button(refresh_key, label='↻ Refresh'):
    if st.button(label, key=f'{refresh_key}_btn',
                 help="Reload this tab's data from the database"):
        st.session_state[refresh_key] = True
        st.rerun()


def schema_reference(*names):
    '''Render a collapsed quick-reference of the given pluginschema classes.'''
    from ..pluginschema import SCHEMA_REFERENCE
    with st.expander('Schema reference (pluginschema)'):
        for name in names:
            entry = SCHEMA_REFERENCE.get(name)
            if entry is None:
                continue
            desc, methods, parts = entry
            st.markdown(f"**`{name}`** — {desc}")
            for sig, purpose in methods.items():
                st.markdown(f"- `{sig}` — {purpose}")
            for part, purpose in parts.items():
                st.markdown(f"- part `{name}.{part}` — {purpose}")
