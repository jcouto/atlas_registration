import numpy as np
import pandas as pd
import streamlit as st

def _spike_fr_profile(surface_depths, domain, duration, n_depth=280,
                      smooth_bins=1.5):
    '''Firing-rate-vs-depth profile: the depth marginal of the drift raster,
    i.e. the SAME spikes on the SAME depth bins (``n_depth`` over ``domain``,
    matching ``_drift_hist``), divided by duration and only lightly smoothed.
    Using identical bins guarantees the FR line tracks the drift density (no
    peak shift from smoothing).  Returns (bin_centers, firing_rate_Hz).'''
    edges = np.linspace(domain[0], domain[1], n_depth + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    if not len(surface_depths) or duration <= 0:
        return centers, np.zeros_like(centers)
    cnt, _ = np.histogram(surface_depths, bins=edges)
    fr = cnt / float(duration)
    if smooth_bins:
        try:
            from scipy.ndimage import gaussian_filter1d
            fr = gaussian_filter1d(fr, smooth_bins)
        except Exception:
            pass
    return centers, fr


def _unit_density_profile(depths, domain, n_depth=140, smooth_bins=1.0):
    '''Unit-count-vs-depth profile (linear): number of units per depth bin,
    lightly smoothed.  Returns (bin_centers, unit_count).'''
    edges = np.linspace(domain[0], domain[1], n_depth + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    if not len(depths):
        return centers, np.zeros_like(centers)
    cnt, _ = np.histogram(depths, bins=edges)
    cnt = cnt.astype(float)
    if smooth_bins:
        try:
            from scipy.ndimage import gaussian_filter1d
            cnt = gaussian_filter1d(cnt, smooth_bins)
        except Exception:
            pass
    return centers, cnt


def _drift_hist(times, surface_depths, domain, n_time=1000, n_depth=356):
    '''Downsampled drift density (depth × time) for a grayscale heatmap, plus the
    time-bin and depth-bin centers.  Spikes are already subsampled by
    ``get_spikes``; here we bin them into a 2D histogram.  The grid is kept
    modest (n_time × n_depth) because this heatmap is re-rendered in the browser
    on every rerun (e.g. editing a reference pair) — a smaller payload = a
    shorter grayed-out pause.'''

    if not len(times):
        return None, None, None
    tb = np.linspace(times.min(), times.max(), n_time + 1)
    db = np.linspace(domain[0], domain[1], n_depth + 1)
    H, _, _ = np.histogram2d(times, surface_depths, bins=[tb, db])
    z = np.sqrt(H.T)                              # (depth, time), soften the scale
    return z, (tb[:-1] + tb[1:]) / 2, (db[:-1] + db[1:]) / 2


def _region_text_color(rgb):
    return 'white' if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 140 else 'black'


def _depth_col(coords, shanks):
    '''Which channel_coords column is the along-shank DEPTH axis.

    Not the global argmax span — for a multi-shank probe the lateral (shank)
    axis can span more than the recorded depth.  Instead pick the axis with the
    larger typical *within-shank* span.'''

    coords = np.asarray(coords, float)
    shanks = np.asarray(shanks).ravel()
    best_col, best_span = 0, -1.0
    for c in (0, 1):
        spans = [float(np.ptp(coords[shanks == s, c]))
                 for s in np.unique(shanks) if int((shanks == s).sum()) > 1]
        m = float(np.median(spans)) if spans else float(np.ptp(coords[:, c]))
        if m > best_span:
            best_span, best_col = m, c
    return best_col


_CLICK_Y = np.nan   # sentinel; the invisible click markers span this many points
_N_CLICK = 200


def _add_region_column(fig, regions, col, span):
    '''Draw region blocks + acronym labels into subplot ``col`` (regions carry
    ``entry_um``/``exit_um`` in that subplot's depth coordinate).'''
    xref = 'x domain' if col == 1 else f'x{col} domain'
    yref = 'y' if col == 1 else f'y{col}'
    for _, r in regions.iterrows():
        lo, hi = float(r['entry_um']), float(r['exit_um'])
        if hi - lo <= 0:
            continue
        rgb = r.get('rgb') or [120, 120, 120]
        fig.add_shape(type='rect', xref=xref, yref=yref, x0=0, x1=1, y0=lo, y1=hi,
                      line_width=0.3, line_color='white',
                      fillcolor=f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})',
                      layer='below')
        if hi - lo >= span * 0.012:
            fig.add_annotation(xref=xref, yref=yref, x=0.05, y=(lo + hi) / 2,
                               text=r['acronym'], showarrow=False, xanchor='left',
                               font=dict(size=9, color=_region_text_color(rgb)))


def _build_figure(regions, regions_aligned, snr_df, grid, fr, u_grid, u_dens,
                  drift, domain, insertion_depth, feature_ref, track_ref):
    '''Plotly figure with ONE shared depth axis (0 = brain surface at top).

    The ephys panels (SNR, FR, Firing, Units, Drift) are held FIXED at the depth
    from the insertion (``surf = insertion_depth − electrode``); the reference
    pairs warp the **Aligned** region column (right) so it slides onto the
    features, while the left **Regions** column stays at the raw atlas depths.

    Hover stays explicit: every ephys panel shows the electrode depth (from tip)
    and the aligned atlas depth.  Clicking Regions sets an atlas depth; clicking
    an ephys panel sets the electrode depth (carried in customdata).
    '''
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from atlas_registration import electrode_to_atlas

    fig = make_subplots(
        rows=1, cols=7, shared_yaxes=True,
        column_widths=[0.048, 0.056, 0.035, 0.035, 0.10, 0.10, 0.55],
        horizontal_spacing=0.01,
        subplot_titles=['Regions', 'Aligned', 'SNR', 'FR', 'Firing', 'Units',
                        'Drift raster'])

    span = domain[1] - domain[0]
    yy = np.linspace(domain[0], domain[1], _N_CLICK)
    yy_e = insertion_depth - yy                        # electrode of each surf y
    yy_al = electrode_to_atlas(yy_e, insertion_depth, feature_ref, track_ref)

    def _feat_cd(surf_vals):
        '''customdata [tag, electrode, aligned_atlas] for surf positions.'''
        e = insertion_depth - np.asarray(surf_vals, float)
        al = electrode_to_atlas(e, insertion_depth, feature_ref, track_ref)
        return np.stack([np.array(['feature'] * len(e), dtype=object), e, al], axis=1)

    def _add_feature_markers(col, xpositions):
        xp = np.atleast_1d(np.asarray(xpositions, float))
        xs = np.repeat(xp, len(yy))
        cd = np.tile(_feat_cd(yy), (len(xp), 1))
        fig.add_trace(go.Scatter(x=xs, y=np.tile(yy, len(xp)), mode='markers',
                                 marker=dict(size=16, color='rgba(0,0,0,0)'),
                                 customdata=cd, hoverinfo='skip', showlegend=False),
                      row=1, col=col)

    # these traces carry a 2-element customdata [electrode, aligned_atlas]
    _EHOVER = ('electrode %{customdata[0]:.0f} µm<br>'
               'atlas %{customdata[1]:.0f} µm')

    # region columns side by side: raw atlas (col 1) and aligned/warped (col 2).
    # The markers span the FULL region extent (not just the electrode domain) so
    # every drawn region is hoverable/clickable, even outside the ephys span.
    for col, reg in ((1, regions), (2, regions_aligned)):
        _add_region_column(fig, reg, col, span)
        lo = float(min(domain[0], reg['entry_um'].min()))
        hi = float(max(domain[1], reg['exit_um'].max()))
        yr = np.linspace(lo, hi, int(_N_CLICK * max(1, (hi - lo) / max(span, 1))))
        # track depth for a click: atlas y directly on col 1; on the aligned
        # column, map the warped position back to atlas via the alignment
        if col == 1:
            tag_atlas = yr
        else:
            tag_atlas = electrode_to_atlas(insertion_depth - yr, insertion_depth,
                                           feature_ref, track_ref)
        hov = []
        for y in yr:
            hit = reg[(reg['entry_um'] <= y) & (reg['exit_um'] > y)]
            r0 = hit.iloc[0] if len(hit) else None
            hov.append('' if r0 is None else
                       f"{r0['acronym']} — {r0.get('name', '')}<br>"
                       f"{r0['entry_um']:.0f}–{r0['exit_um']:.0f} µm")
        cd = np.stack([np.array(['track'] * len(yr), dtype=object), tag_atlas], axis=1)
        fig.add_trace(go.Scatter(x=[0.5] * len(yr), y=yr, mode='markers',
                                 marker=dict(size=22, color='rgba(0,0,0,0)'),
                                 customdata=cd, text=hov,
                                 hovertemplate='%{text}<extra></extra>',
                                 showlegend=False), row=1, col=col)

    # SNR (col 3)
    if not snr_df.empty:
        sd = snr_df.groupby('surf', as_index=False)['snr'].mean().sort_values('surf')
        sy = sd['surf'].to_numpy()
        fig.add_trace(go.Heatmap(z=sd['snr'].to_numpy()[:, None], x=[0], y=sy,
                                 colorscale='Viridis', showscale=False,
                                 customdata=_feat_cd(sy)[:, 1:],
                                 hovertemplate=_EHOVER + '<br>SNR %{z:.1f}<extra></extra>'),
                      row=1, col=3)
    _add_feature_markers(3, [0.0])

    # FR colormap (col 4) — LOG colour
    if len(grid) and fr.max() > 0:
        floor = max(fr.max() * 1e-3, 1e-2)
        cd = _feat_cd(grid)[:, 1:].reshape(len(grid), 1, 2)
        fig.add_trace(go.Heatmap(z=np.log10(np.maximum(fr, floor))[:, None], x=[0],
                                 y=grid, colorscale='Magma', showscale=False, customdata=cd,
                                 hovertemplate=_EHOVER + '<extra></extra>'),
                      row=1, col=4)
    _add_feature_markers(4, [0.0])

    # Firing line (col 5) — LINEAR x
    fr_max = float(np.nanmax(fr)) if len(fr) else 1.0
    if len(grid):
        fig.add_trace(go.Scatter(x=fr, y=grid, mode='lines', line=dict(color='black'),
                                 customdata=_feat_cd(grid)[:, 1:],
                                 hovertemplate=_EHOVER + '<br>FR %{x:.1f} Hz<extra></extra>',
                                 showlegend=False), row=1, col=5)
    _add_feature_markers(5, np.linspace(0, max(fr_max, 1.0), 4))

    # Unit-density line (col 6) — LINEAR x
    u_max = float(np.nanmax(u_dens)) if len(u_dens) else 1.0
    if len(u_grid):
        fig.add_trace(go.Scatter(x=u_dens, y=u_grid, mode='lines', line=dict(color='teal'),
                                 customdata=_feat_cd(u_grid)[:, 1:],
                                 hovertemplate=_EHOVER + '<br>units %{x:.1f}<extra></extra>',
                                 showlegend=False), row=1, col=6)
    _add_feature_markers(6, np.linspace(0, max(u_max, 1.0), 4))

    # Drift raster (col 7).  Rasterize to a grayscale bitmap and draw it as a
    # go.Image — a go.Heatmap ships all cells and re-renders them in the browser
    # on every remount (slow); an image is one <image> element (fast).  The full
    # raster is a compact PNG (no per-pixel hover data → tiny payload); a small
    # overlay carries the electrode/atlas hover for only the first 30 s.
    if drift is not None:
        from .common import to_base64
        z, tc, dc = drift
        al = electrode_to_atlas(insertion_depth - dc, insertion_depth,
                                feature_ref, track_ref)
        zmax = float(z.max()) or 1.0
        gray = (255.0 * (1.0 - z / zmax)).clip(0, 255).astype(np.uint8)  # dense = dark
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        dx = (tc[-1] - tc[0]) / (len(tc) - 1) if len(tc) > 1 else 1.0
        dy = (dc[-1] - dc[0]) / (len(dc) - 1) if len(dc) > 1 else 1.0
        fig.add_trace(go.Image(source=to_base64(rgb), x0=float(tc[0]), dx=float(dx),
                               y0=float(dc[0]), dy=float(dy), hoverinfo='skip'),
                      row=1, col=7)
        # hover overlay: only the first 30 s of the raster carries customdata
        n30 = int(np.searchsorted(tc, tc[0] + 30.0)) + 1
        n30 = max(1, min(n30, z.shape[1]))
        cd = np.stack([np.tile((insertion_depth - dc)[:, None], (1, n30)),
                       np.tile(al[:, None], (1, n30))], axis=-1)
        fig.add_trace(go.Image(z=rgb[:, :n30], x0=float(tc[0]), dx=float(dx),
                               y0=float(dc[0]), dy=float(dy), customdata=cd,
                               hovertemplate='t %{x:.0f}s<br>electrode %{customdata[0]:.0f} µm'
                               '<br>atlas %{customdata[1]:.0f} µm<extra></extra>'),
                      row=1, col=7)
        # go.Image locks pixels square by default; let it stretch to fill the panel
        fig.update_xaxes(range=[float(tc[0]), float(tc[-1])], row=1, col=7)
        fig.update_yaxes(scaleanchor=None, autorange=False, row=1, col=7)
        _add_feature_markers(7, np.linspace(tc[0], tc[-1], 12))

    # reference lines: atlas track_ref on the raw Regions column; the electrode
    # feature_ref (drawn at its surf position) across the Aligned + ephys panels
    for d in (track_ref or []):
        fig.add_hline(y=d, line_dash='dash', line_color='deepskyblue',
                      line_width=1.2, row=1, col=1)
    for e in (feature_ref or []):
        for c in (2, 3, 4, 5, 6, 7):
            fig.add_hline(y=insertion_depth - e, line_dash='dash',
                          line_color='deepskyblue', line_width=1.2, row=1, col=c)

    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False,
                     title_text=None)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False,
                     title_text=None)
    fig.update_yaxes(range=[domain[1], domain[0]])       # 0 surface on top, linked
    fig.update_layout(height=620, margin=dict(l=6, r=6, t=30, b=10),
                      dragmode='zoom', plot_bgcolor='white')
    return fig


def _channel_figure(coords, shanks, sel_shank, depth_col):
    '''Independent channel-map scatter (physical geometry).'''
    import plotly.graph_objects as go
    hcol, xcol = coords[:, depth_col], coords[:, 1 - depth_col]
    sel = shanks == sel_shank
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xcol[~sel], y=hcol[~sel], mode='markers',
                             marker=dict(size=5, color='lightgray'), showlegend=False))
    fig.add_trace(go.Scatter(x=xcol[sel], y=hcol[sel], mode='markers',
                             marker=dict(size=5, color='crimson'), showlegend=False,
                             hovertemplate='x %{x:.0f}<br>height %{y:.0f}<extra></extra>'))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(height=620, margin=dict(l=6, r=6, t=30, b=10),
                      title='Channel map', plot_bgcolor='white')
    return fig


def _clicked_depths(event):
    '''Return {'track': y, 'feature': y} from a Plotly on_select event.
    Defensive: never raises on an unexpected event shape.'''
    out = {}
    try:
        sel = getattr(event, 'selection', None)
        if sel is None and isinstance(event, dict):
            sel = event.get('selection')
        points = (sel or {}).get('points', []) if sel else []
        for pt in points:
            if not isinstance(pt, dict):
                continue
            cd = pt.get('customdata')
            # customdata is [tag, value]: value is the electrode depth (feature)
            # or the atlas depth (track).  Fall back to y for a bare tag.
            if isinstance(cd, (list, tuple)) and len(cd) >= 2:
                tag, val = cd[0], cd[1]
            else:
                tag = cd[0] if isinstance(cd, (list, tuple)) and cd else cd
                val = pt.get('y')
            if tag in ('track', 'feature') and val is not None:
                out[tag] = float(val)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Module-level cached fetches.  Keeping these out of the tab function (and
# passing a hashable ``ck`` = sorted sort_key items) avoids the stale-result
# pitfalls of caching nested closures, so every selector change re-fetches.
# ---------------------------------------------------------------------------
def _ck(sort_key):
    def native(v):
        return int(v) if isinstance(v, (np.integer, int)) else str(v)
    return tuple(sorted((k, native(v)) for k, v in sort_key.items()))


@st.cache_data(show_spinner=False)
def _fetch_sortings(subject_name, probe_id):
    from ..pluginschema import SpikeSorting, EphysRecording, ProbeConfiguration
    q = (SpikeSorting & dict(subject_name=subject_name)
         & (EphysRecording.ProbeSetting * ProbeConfiguration
            & dict(probe_id=probe_id)))
    return q.fetch('subject_name', 'session_name', 'dataset_name', 'probe_num',
                   'parameter_set_num', as_dict=True)


@st.cache_data(show_spinner=False)
def _fetch_channel_map(ck):
    from ..pluginschema import ProbeConfiguration, EphysRecording
    rows = (ProbeConfiguration * EphysRecording.ProbeSetting & dict(ck)).fetch(
        'probe_id', 'channel_coords', 'channel_shank', as_dict=True)
    if not rows:
        return None, None, None
    c = rows[0]
    return (c['probe_id'], np.asarray(c['channel_coords'], float),
            np.asarray(c['channel_shank']).ravel().astype(int))


@st.cache_data(show_spinner=False)
def _fetch_avail_crit(ck):
    from ..pluginschema import UnitCount
    return sorted((UnitCount & dict(ck)).fetch('unit_criteria_id').tolist())


@st.cache_data(show_spinner=False)
def _fetch_config_id(ck):
    '''Probe configuration_id used by this sorting (the channel geometry the
    alignment depends on).'''
    from ..pluginschema import ProbeConfiguration, EphysRecording
    r = (ProbeConfiguration * EphysRecording.ProbeSetting & dict(ck)).fetch(
        'configuration_id')
    return int(r[0]) if len(r) else None


@st.cache_data(show_spinner=False)
def _fetch_insertion_depth(ck):
    from ..pluginschema import ProbeInsertion
    sk = dict(ck)
    pid, coords, _ = _fetch_channel_map(ck)
    d = (ProbeInsertion & dict(subject_name=sk['subject_name'],
                               probe_id=pid)).fetch('insertion_depth')
    if len(d):
        return float(d[0])
    _, coords_all, shanks_all = _fetch_channel_map(ck)
    depth_col = _depth_col(coords_all, shanks_all)
    # no ProbeInsertion: default so the shallowest electrode sits at the surface
    return float(coords_all[:, depth_col].max())


@st.cache_data(show_spinner='Computing SNR…')
def _fetch_snr(ck, shank):
    from ..pluginschema import SpikeSorting
    sk = dict(ck)
    _, coords, shanks = _fetch_channel_map(ck)
    if coords is None:
        return pd.DataFrame()
    depth_col = _depth_col(coords, shanks)
    scc, sci = (SpikeSorting & sk).fetch1('sorting_channel_coords',
                                          'sorting_channel_indices')
    sci = np.asarray(sci).ravel().astype(int)
    scc = np.asarray(scc, float)
    if sci.max() >= len(shanks):
        return pd.DataFrame()
    mask = shanks[sci] == int(shank)
    heights = scc[:, depth_col][mask]
    segs = (SpikeSorting.Segment & sk).fetch('segment', as_dict=True)
    if not segs:
        return pd.DataFrame()
    acc = np.zeros(int(mask.sum()))
    for s in segs:
        seg = np.asarray(s['segment'], float)
        if seg.shape[1] != len(sci):
            seg = seg.T
        x = seg[:, mask]
        dev = x - np.median(x, axis=0)
        noise = 1.4826 * np.median(np.abs(dev), axis=0) + 1e-9
        snr = np.empty(x.shape[1])
        for c in range(x.shape[1]):
            dc = dev[:, c]
            tr = dc[dc < -5.0 * noise[c]]
            sig = -tr.mean() if tr.size > 5 else np.percentile(-dc, 99.9)
            snr[c] = sig / noise[c]
        acc += snr
    acc /= len(segs)
    return pd.DataFrame({'height': heights, 'snr': acc})


@st.cache_data(show_spinner='Loading spikes…')
def _fetch_spikes(ck, shank, crit):
    from ..pluginschema import (SpikeSorting, UnitMetrics, UnitCount,
                                EphysRecording)
    sk = dict(ck)
    _, coords, shanks = _fetch_channel_map(ck)
    depth_col = _depth_col(coords, shanks)
    if crit is None:
        umids = set(int(u) for u in
                    (UnitMetrics & sk & dict(shank=int(shank))).fetch('unit_id'))
    else:
        umids = set(int(u) for u in
                    (UnitCount.Unit * UnitMetrics & sk
                     & dict(unit_criteria_id=int(crit), shank=int(shank))
                     & 'passes = 1').fetch('unit_id'))
    srate = float((EphysRecording.ProbeSetting & sk).fetch1('sampling_rate'))
    rows = (SpikeSorting.Unit & sk).fetch(
        'unit_id', 'spike_times', 'spike_positions', 'spike_amplitudes',
        as_dict=True)
    T, D, A = [], [], []
    for u in rows:
        if int(u['unit_id']) not in umids or u['spike_positions'] is None:
            continue
        tt = np.asarray(u['spike_times'], float) / srate
        pos = np.asarray(u['spike_positions'], float)
        dep = pos[:, depth_col] if pos.ndim == 2 else pos
        am = (np.asarray(u['spike_amplitudes'], float)
              if u['spike_amplitudes'] is not None else np.ones(len(tt)))
        n = min(len(tt), len(dep), len(am))
        T.append(tt[:n]); D.append(dep[:n]); A.append(am[:n])
    if not T:
        return np.array([]), np.array([]), np.array([])
    T, D, A = np.concatenate(T), np.concatenate(D), np.concatenate(A)
    if len(T) > 300000:
        idx = np.random.default_rng(0).choice(len(T), 300000, replace=False)
        T, D, A = T[idx], D[idx], A[idx]
    return T, D, A


@st.cache_data(show_spinner=False)
def _fetch_unit_depths(ck, shank, crit):
    '''Per-unit depth (height above the tip, micron) for the shank, optionally
    filtered by a unit-count criteria.'''
    from ..pluginschema import UnitMetrics, UnitCount
    sk = dict(ck)
    if crit is None:
        d = (UnitMetrics & sk & dict(shank=int(shank))).fetch('depth')
    else:
        d = (UnitCount.Unit * UnitMetrics & sk
             & dict(unit_criteria_id=int(crit), shank=int(shank))
             & 'passes = 1').fetch('depth')
    return np.asarray([float(x) for x in d if x is not None], dtype=float)


@st.cache_data(show_spinner='Sampling track…')
def _fetch_samples(tck):
    '''Per-voxel ``sample_annotation_along_track`` for a track.  Only depends on
    the track key, so cache it — otherwise it reloads the whole atlas annotation
    and re-fits the track on every rerun (e.g. when adding a reference pair).'''
    from ..pluginschema import ProbeTrack
    return (ProbeTrack & dict(tck)).get_samples()


@st.cache_data(show_spinner=False)
def _fetch_lookup(atlas):
    from atlas_registration import get_structure_lookup
    return get_structure_lookup(atlas)


# The drift / firing-rate / unit-density panels depend only on the spikes and the
# depth domain (not on the reference pairs), so cache them by small scalar keys —
# otherwise editing a pair re-bins millions of spikes on every rerun.
@st.cache_data(show_spinner='Building drift raster…')
def _compute_drift(ck, shank, crit, insertion_depth, dom):
    T, D, _ = _fetch_spikes(ck, shank, crit)
    surf = (insertion_depth - D) if len(T) else np.array([])
    return _drift_hist(T, surf, dom)


@st.cache_data(show_spinner='Computing firing rate…')
def _compute_fr(ck, shank, crit, insertion_depth, dom):
    T, D, _ = _fetch_spikes(ck, shank, crit)
    surf = (insertion_depth - D) if len(T) else np.array([])
    duration = float(T.max() - T.min()) if len(T) else 0.0
    return _spike_fr_profile(surf, dom, duration)


@st.cache_data(show_spinner='Computing unit density…')
def _compute_unit_density(ck, shank, crit, insertion_depth, dom):
    ud = _fetch_unit_depths(ck, shank, crit)
    surf = (insertion_depth - ud) if len(ud) else np.array([])
    return _unit_density_profile(surf, dom)


_FETCHERS = (_fetch_sortings, _fetch_channel_map, _fetch_avail_crit,
             _fetch_config_id, _fetch_insertion_depth, _fetch_snr, _fetch_spikes,
             _fetch_unit_depths, _fetch_samples, _fetch_lookup,
             _compute_drift, _compute_fr, _compute_unit_density)


def _alignment_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                   AtlasRegistrationAnnotation, ProbeTrack, ProbeAlignment):
    from ..pluginschema import (SpikeSorting, UnitMetrics, UnitCount,
                                ProbeConfiguration, EphysRecording,
                                ProbeInsertion, Subject)
    sel_key = st.session_state.get('ar_selected_key')
    if not sel_key:
        st.info('Select a registration in the Sessions tab first.')
        return
    if st.button('↻ Refresh data', key='refresh_ar_align_btn',
                 help='Clear cached spikes/SNR/channel maps and re-fetch'):
        for f in _FETCHERS:
            f.clear()
        st.rerun()
    subject = sel_key['subject_name']

    def _native(v):
        return int(v) if isinstance(v, (np.integer, int)) else str(v)

    # ---- 1. pick the shank to align (every shank annotation of the brain;
    #         its ProbeTrack is fitted on demand if not done yet) -----------
    anns = (AtlasRegistrationAnnotation & sel_key
            & 'annotation_type = "shank"').fetch('annotation_id',
                                                 'annotation_name', as_dict=True)
    if not anns:
        st.info('No shank annotations for this brain. Trace them in the Track tab '
                'first.')
        return
    tname = {int(a['annotation_id']): a['annotation_name'] for a in anns}

    top = st.columns(5)
    tid = int(top[0].selectbox('Probe track', sorted(tname),
                               format_func=lambda i: f"{i} · {tname[i]}",
                               key='ar_align_tid'))
    annotation_name = tname[tid]
    probe_id, shank_str = annotation_name.rsplit('_shank', 1)
    sel_shank = int(shank_str)
    tkey = dict(sel_key, annotation_id=tid)

    if not len(ProbeTrack & tkey):
        with st.spinner(f'Fitting ProbeTrack for {annotation_name}…'):
            try:
                ProbeTrack.populate(tkey, display_progress=False)
            except Exception as exc:
                st.error(f'Could not fit a ProbeTrack for {annotation_name}: {exc}')
                return
    if not len(ProbeTrack & tkey):
        st.warning(f'No ProbeTrack for {annotation_name}.')
        return
    
    track = (ProbeTrack & tkey)
    regions = pd.DataFrame(track.fetch1('regions'))
    # ---- 2. spike-sorting session + parameter set (same probe) ---------
    sortings = _fetch_sortings(subject, probe_id)
    if not sortings:
        st.warning(f'No spike sorting recorded with probe **{probe_id}** for this '
                   f'subject.')
        return
    sess_keys = sorted({(s['session_name'], s['dataset_name'], s['probe_num'])
                        for s in sortings})
    sxi = top[1].selectbox(
        'Spike-sorting session', range(len(sess_keys)),
        format_func=lambda i: f"{sess_keys[i][0]}/{sess_keys[i][1]} · "
                              f"probe{sess_keys[i][2]}", key='ar_align_sess')
    sess = sess_keys[sxi]
    psets = sorted({s['parameter_set_num'] for s in sortings
                    if (s['session_name'], s['dataset_name'],
                        s['probe_num']) == sess})
    pset = top[2].selectbox('parameter_set_num', psets, key='ar_align_pset')
    sort_key = {'subject_name': subject, 'session_name': _native(sess[0]),
                'dataset_name': _native(sess[1]), 'probe_num': _native(sess[2]),
                'parameter_set_num': _native(pset)}
    ck = _ck(sort_key)
    _pid, coords, shanks = _fetch_channel_map(ck)
    if coords is None:
        st.warning('No ProbeConfiguration for that sorting.')
        return
    depth_col = _depth_col(coords, shanks)
    if sel_shank not in np.unique(shanks):
        st.warning(f'Shank {sel_shank} (from the track) is not in this sorting’s '
                   f'channel map (shanks {sorted(np.unique(shanks).tolist())}).')
        return
    avail_crit = _fetch_avail_crit(ck)
    crit_sel = top[3].selectbox('Units', ['all units'] + [f'criteria {c}'
                                                          for c in avail_crit],
                                key='ar_align_crit',
                                help='Filter units by a UnitCountCriteria (SUA).')
    align_id = int(top[4].number_input('alignment_id', value=0, step=1,
                                       min_value=0, key='ar_align_id'))
    config_id = _fetch_config_id(ck)
    st.caption(f'Aligning **{annotation_name}** (shank {sel_shank}) → '
               f'ProbeTrack annotation_id={tid} · probe configuration_id='
               f'{config_id} (channel geometry).')
    ic1, ic2 = st.columns([1, 3])
    insertion_depth = ic1.number_input(
        'Insertion depth (µm)', value=_fetch_insertion_depth(ck),
        step=50.0, key=f'ar_ins_{tid}',
        help='Tip depth from the brain surface. With no reference pairs the map '
             'is atlas = insertion_depth − electrode(from tip).')
    ic2.caption('Electrode depth is measured from the probe tip (0 = tip). '
                'Atlas depth is from the brain surface (0 = surface). The '
                'insertion depth + reference pairs map between them.')

    # electrode depth = the raw channel_coords depth (the probe/sorter measures
    # it from the tip, 0 = tip) — do NOT shift by the deepest recorded channel.
    channel_depths = np.sort(coords[shanks == sel_shank, depth_col])
    e_max = float(channel_depths.max())
    crit_id = None if crit_sel == 'all units' else int(crit_sel.split()[-1])

    snr_df = _fetch_snr(ck, sel_shank)

    # ---- reference pairs: (electrode depth from tip, atlas depth) --------
    ref_key = f'ar_refs_{tkey}_{align_id}'
    if ref_key not in st.session_state:
        prev = (ProbeAlignment & dict(tkey, alignment_id=align_id)).fetch(as_dict=True)
        if prev and prev[0].get('feature_ref') is not None:
            st.session_state[ref_key] = [
                [float(f), float(t)] for f, t in
                zip(np.asarray(prev[0]['feature_ref']).ravel(),
                    np.asarray(prev[0]['track_ref']).ravel())]
        else:
            st.session_state[ref_key] = []
    refs = st.session_state[ref_key]
    feature_ref = [r[0] for r in refs] or None    # electrode depths (from tip)
    track_ref = [r[1] for r in refs] or None       # atlas depths (from surface)

    from atlas_registration import (electrode_to_atlas, atlas_to_electrode,
                                    align_channels_to_regions, get_structure_lookup)
    atlas_of_channels = electrode_to_atlas(channel_depths, insertion_depth,
                                           feature_ref, track_ref)
    diffs = np.diff(atlas_of_channels)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    # The ephys is plotted at surf = insertion_depth − electrode (FIXED w.r.t. the
    # reference pairs), so the drift/features don't jump as you align.  The
    # y-scale is set to the electrode span in this coordinate.
    ch_surf = insertion_depth - channel_depths
    domain = (float(ch_surf.min()), float(ch_surf.max()))

    if not snr_df.empty:
        snr_df = snr_df.copy()
        snr_df['electrode'] = snr_df['height']              # raw, from tip
        snr_df['surf'] = insertion_depth - snr_df['electrode']

    # cached by (sorting, shank, criteria, insertion depth, domain) — editing a
    # reference pair doesn't touch these, so they stay warm across pair edits
    drift = _compute_drift(ck, sel_shank, crit_id, insertion_depth, domain)
    grid, fr_profile = _compute_fr(ck, sel_shank, crit_id, insertion_depth, domain)
    u_grid, u_dens = _compute_unit_density(ck, sel_shank, crit_id, insertion_depth,
                                           domain)
    # aligned region column: warp the atlas boundaries onto the fixed ephys axis
    #   boundary at atlas A -> surf = insertion_depth - atlas_to_electrode(A)
    regions_aligned = regions.copy()
    ae = atlas_to_electrode(regions['entry_um'].to_numpy(), insertion_depth,
                            feature_ref, track_ref)
    ax = atlas_to_electrode(regions['exit_um'].to_numpy(), insertion_depth,
                            feature_ref, track_ref)
    pe, px = insertion_depth - ae, insertion_depth - ax
    regions_aligned['entry_um'] = np.minimum(pe, px)
    regions_aligned['exit_um'] = np.maximum(pe, px)
    with st.spinner('Drawing alignment figure…'):
        fig = _build_figure(regions, regions_aligned, snr_df, grid, fr_profile,
                            u_grid, u_dens, drift, domain, insertion_depth,
                            feature_ref, track_ref)
    # the key must change with the current selection, otherwise st.plotly_chart
    # with on_select keeps showing the previous figure
    # the key must change with EVERYTHING that changes the figure, otherwise the
    # on_select Plotly widget keeps the previous render (reference pairs and the
    # insertion depth both move the panels, so they belong here too)
    ref_sig = ';'.join(f'{a:.1f}:{b:.1f}' for a, b in refs)
    plot_sig = (f"{tid}_{sxi}_{pset}_{sel_shank}_{crit_id}_{align_id}_"
                f"{int(e_max)}_{insertion_depth:.0f}_{ref_sig}")
    main, side = st.columns([5, 1])
    with main:
        ev = st.plotly_chart(fig, on_select='rerun', selection_mode='points',
                             key=f'ar_plot_{plot_sig}', width='stretch')
    with side:
        st.plotly_chart(_channel_figure(coords, shanks, sel_shank, depth_col),
                        key=f'ar_chan_{plot_sig}', width='stretch')

    # A click returns a depth; write it straight into the matching input.  This
    # runs before the number_input widgets below are instantiated, so setting
    # their session_state here is allowed (no st.rerun / pending dance needed).
    clicks = _clicked_depths(ev)
    for cd, suf in (('track', 't'), ('feature', 'f')):
        wkey = f'{ref_key}_{suf}'
        if cd in clicks and abs(clicks[cd] - st.session_state.get(wkey, 0.)) > 1e-6:
            st.session_state[wkey] = round(clicks[cd], 1)

    # ---- reference pair editor -----------------------------------------
    st.write('**Reference pairs** — pin an *electrode depth* (from tip; click the '
             'FR/Drift panels) to an *atlas depth* (from surface; click the Regions '
             'panel), or type them below. Edit the table directly to change or '
             'delete pairs.')

    # A form batches the two inputs so typing/clicking into them does NOT rerun
    # the whole tab — only the submit button commits (and rebuilds the figure).
    with st.form(key=f'{ref_key}_form', clear_on_submit=False):
        a1, a2, a3 = st.columns([2, 2, 1])
        t_in = a1.number_input('atlas depth from surface (µm)', step=20.0,
                               key=f'{ref_key}_t')
        f_in = a2.number_input('electrode depth from tip (µm)', step=20.0,
                               key=f'{ref_key}_f')
        a3.markdown('<div style="height:1.7em"></div>', unsafe_allow_html=True)
        submitted = a3.form_submit_button('Add pair')
    if submitted:
        refs.append([float(f_in), float(t_in)])
        st.session_state[ref_key] = refs
        st.rerun()

    c_clear, c_save = st.columns(2)
    if c_clear.button('Clear', key=f'{ref_key}_clr', disabled=not refs):
        st.session_state[ref_key] = []
        st.rerun()
    if c_save.button('Save alignment', type='primary', disabled=not monotonic,
                     key=f'{ref_key}_save'):
        ProbeAlignment().align(
            tkey, feature_ref=feature_ref, track_ref=track_ref,
            alignment_id=align_id, insertion_depth=insertion_depth,
            channel_depths=channel_depths, replace=True,
            alignment_user=st.session_state.get('user_name') or None)
        st.success(f'Saved alignment id={align_id} for {annotation_name}.')
        st.rerun()

    if refs:
        # editable table: change a value to update a pair, or use the row menu to
        # delete one.  Committing an edit rebuilds the figure.
        df = pd.DataFrame(refs, columns=['electrode_from_tip_um',
                                         'atlas_from_surface_um'])
        edited = st.data_editor(df, num_rows='dynamic', hide_index=True,
                                width='stretch', key=f'{ref_key}_editor')
        new_refs = [[float(r['electrode_from_tip_um']),
                     float(r['atlas_from_surface_um'])]
                    for _, r in edited.iterrows()
                    if pd.notna(r['electrode_from_tip_um'])
                    and pd.notna(r['atlas_from_surface_um'])]
        if new_refs != refs:
            st.session_state[ref_key] = new_refs
            st.rerun()
    st.caption('0 pairs → pure inversion using the insertion depth '
               '(atlas = insertion_depth − electrode) · 1 pair → shifts the '
               'inversion · n pairs → piecewise. Deeper electrode ⇒ larger atlas '
               'depth (the map is decreasing).')
    if not monotonic:
        st.error('Reference pairs give a non-monotonic map — the atlas depth must '
                 'change monotonically with electrode depth.')

    # ---- per-channel region preview + save -----------------------------
    samples = _fetch_samples(_ck(tkey))
    lookup = _fetch_lookup(st.session_state['ar_selected_atlas'])
    with st.spinner('Assigning channels to regions…'):
        chan = align_channels_to_regions(channel_depths, samples,
                                         feature_ref=feature_ref, track_ref=track_ref,
                                         lookup=lookup, insertion_depth=insertion_depth)
    counts = (chan.groupby('acronym').size().reset_index(name='n_channels'))
    counts = counts[counts['acronym'] != ''].sort_values('n_channels', ascending=False)
    st.write('**Channels per region (preview)**')
    st.dataframe(counts, hide_index=True, width='stretch')
