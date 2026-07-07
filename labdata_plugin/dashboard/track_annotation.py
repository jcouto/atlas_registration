import json
import numpy as np
import pandas as pd
import streamlit as st

from .common import orientation_axes_safe

_TARGET_DIM = 420


def _downsample_factor(shape):
    return max(1, int(np.ceil(max(shape) / _TARGET_DIM)))


@st.cache_resource(show_spinner='Loading registered stack…')
def _load_stack_ds(key_json):
    '''Downsampled registered stack, keeping channels as ``(AP, C, DV, ML)``.'''
    from ..pluginschema import AtlasRegistration
    key = json.loads(key_json)
    stack = np.squeeze(np.asarray((AtlasRegistration & key).get_stack()))
    if stack.ndim == 3:
        stack = stack[:, np.newaxis]      # (AP, 1, DV, ML)
    ds = _downsample_factor((stack.shape[0],) + stack.shape[2:])
    return np.ascontiguousarray(stack[::ds, :, ::ds, ::ds]), ds


@st.cache_resource(show_spinner='Loading atlas annotation…')
def _load_annotation_ds(atlas, geometry, ds):
    from atlas_registration import (get_brainglobe_annotation,
                                    get_structure_lookup)
    ann = get_brainglobe_annotation(atlas, geometry)
    return np.ascontiguousarray(ann[::ds, ::ds, ::ds]), get_structure_lookup(atlas)


def _cumulative_depths(track, res):
    steps = np.diff(track.astype(float), axis=0) * res
    return np.concatenate([[0.0], np.cumsum(np.sqrt((steps ** 2).sum(axis=1)))])


@st.cache_data(show_spinner='Fitting shank tracks…')
def _fit_shanks(sel_key_json, atlas, geometry, ds):
    '''Fit every shank annotation of a brain (cached so switching/clicking is
    responsive).  Returns a list of dicts per shank.'''
    from ..pluginschema import AtlasRegistrationAnnotation
    from atlas_registration import (fit_track_line, trim_track_to_labeled,
                                    sample_annotation_along_track,
                                    regions_along_track, get_brainglobe_metadata)
    key = json.loads(sel_key_json)
    meta = get_brainglobe_metadata(atlas)
    base_res = np.asarray(meta['resolution'], dtype=float)
    annotation_ds, lookup = _load_annotation_ds(atlas, geometry, ds)
    recs = sorted((AtlasRegistrationAnnotation & key
                   & 'annotation_type = "shank"').fetch(as_dict=True),
                  key=lambda r: r['annotation_name'])
    out = []
    for rec in recs:
        pts = np.asarray(rec['xyz'], float)
        if len(pts) < 2:
            continue
        fit = fit_track_line(pts / ds, volume_shape=annotation_ds.shape,
                             orientation=meta.get('orientation', 'asr'))
        track = trim_track_to_labeled(fit['track_voxels'], annotation_ds)
        if not len(track):
            continue
        depths = _cumulative_depths(track, base_res * ds)
        regions = regions_along_track(
            sample_annotation_along_track(track, annotation_ds, lookup=lookup,
                                          resolution=base_res * ds), lookup=lookup)
        name = rec['annotation_name']
        probe, shank = name.rsplit('_shank', 1)
        out.append(dict(name=name, probe=probe, shank=shank,
                        annotation_id=int(rec['annotation_id']),
                        angles=fit['angles'], track=track, depths=depths,
                        regions=regions, points=pts / ds))    # raw points (ds voxels)
    return out


@st.cache_resource(show_spinner='Building 3D brain surface…')
def _atlas_surface(atlas, geometry, ds, step_size=3):
    '''Marching-cubes surface of the labeled atlas (downsampled) for the 3D view.
    Verts are in ``(ap, dv, ml)`` ds-voxel coords, matching the fitted tracks.'''
    import skimage.measure as measure
    annotation_ds, _ = _load_annotation_ds(atlas, geometry, ds)
    verts, faces, _, _ = measure.marching_cubes(annotation_ds != 0,
                                                step_size=step_size)
    return verts, faces


def _region_text_color(rgb):
    return 'white' if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 140 else 'black'


def _add_region_col(fig, reg, col, depth_max, rule_depth, labels, show_axis):
    '''Add one shank's region column (rects + acronym labels + hover markers) to
    subplot ``col``.  ``show_axis`` draws the depth (µm) tick axis — used for the
    leftmost column, which all columns share.'''
    import plotly.graph_objects as go
    xref = 'x domain' if col == 1 else f'x{col} domain'
    yref = 'y' if col == 1 else f'y{col}'
    for _, r in reg.iterrows():
        if r['exit_um'] - r['entry_um'] <= 0:
            continue
        rgb = r.get('rgb') or [120, 120, 120]
        fig.add_shape(type='rect', xref=xref, yref=yref, x0=0, x1=1,
                      y0=r['entry_um'], y1=r['exit_um'], line_width=0.3,
                      line_color='white', layer='below',
                      fillcolor=f'rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})')
        if labels and r['exit_um'] - r['entry_um'] >= depth_max * 0.02:
            fig.add_annotation(xref=xref, yref=yref, x=0.05,
                               y=(r['entry_um'] + r['exit_um']) / 2, text=r['acronym'],
                               showarrow=False, xanchor='left',
                               font=dict(size=8, color=_region_text_color(rgb)))
    yy = np.linspace(0, depth_max, 200)
    hov = []
    for y in yy:
        hit = reg[(reg['entry_um'] <= y) & (reg['exit_um'] > y)]
        r0 = hit.iloc[0] if len(hit) else None
        hov.append('' if r0 is None else
                   f"{r0['acronym']} — {r0.get('name', '')}<br>"
                   f"{r0['entry_um']:.0f}–{r0['exit_um']:.0f} µm")
    fig.add_trace(go.Scatter(x=[0.5] * len(yy), y=yy, mode='markers',
                             marker=dict(size=20, color='rgba(0,0,0,0)'),
                             text=hov, hovertemplate='%{text}<extra></extra>',
                             showlegend=False), row=1, col=col)
    if rule_depth is not None:
        fig.add_hline(y=rule_depth, line_dash='dash', line_color='black',
                      line_width=2, row=1, col=col)
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=col)
    if show_axis:
        fig.update_yaxes(range=[depth_max, 0], showticklabels=True, showgrid=False,
                         ticks='outside', title_text='Depth (µm)',
                         title_font=dict(size=11, color='black'),
                         tickfont=dict(size=9, color='black'), color='black',
                         row=1, col=col)
    else:
        fig.update_yaxes(visible=False, range=[depth_max, 0], row=1, col=col)


def _panel_domains(shanks, reg_w=0.0375, view_block=0.60, probe_gap=0.02,
                   reg_view_gap=0.04):
    '''x-domains for N region columns (small gap within a probe, larger gap
    between probes) plus one projection block (gap before it).  Returns
    ``(region_domains, block)``.'''
    n = len(shanks)
    same_probe_gap = probe_gap / 2.0             # half-gap between sibling shanks
    raw = [reg_w] * n + [view_block]
    gaps = [0.0] * (n + 1)                       # gap BEFORE column j
    for j in range(1, n):
        gaps[j] = (probe_gap if shanks[j]['probe'] != shanks[j - 1]['probe']
                   else same_probe_gap)
    gaps[n] = reg_view_gap                        # regions → projection block
    total = sum(raw) + sum(gaps)
    xs, x = [], 0.0
    for j in range(n + 1):
        x += gaps[j]
        xs.append((min(1.0, x / total), min(1.0, (x + raw[j]) / total)))
        x += raw[j]
    return xs[:n], xs[n]


# projection: (label, fixed role, row role, col role, transpose).  The block is
# a 2x2: Coronal top-left, Horizontal (transposed) top-right, 3D bottom-left,
# Sagittal (transposed) bottom-right (under the Horizontal).
_PROJ = [('Coronal', 'ap', 'dv', 'ml', False),
         ('Horizontal', 'dv', 'ap', 'ml', True),
         ('Sagittal', 'ml', 'ap', 'dv', True)]

_TRACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']


def _build_nav_figure(shanks, sel_idx, nav_depth, stack3d, annotation_ds, lookup,
                      ax, ds, overlay=True, surface=None, height=680):
    '''One figure: every shank's region column (grouped by probe, full height) on
    the left; on the top row Coronal, the transposed Horizontal, and Sagittal of
    the selected shank; on the bottom a 3D view of the labeled surface with every
    shank's raw points and fitted track.'''
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from .common import _normalize_with_pct, annotation_rgb
    n = len(shanks)
    sel = shanks[sel_idx]
    track, depths = sel['track'], sel['depths']
    nav_idx = int(np.argmin(np.abs(depths - nav_depth)))
    nav_voxel = track[nav_idx]

    # region columns span both rows; the projection block is a 2x2: Coronal
    # top-left, Horizontal top-right, 3D bottom-left, Sagittal bottom-right.
    # Axis numbers (row-major): region 1..n, coronal n+1, horizontal n+2, then
    # the scene ('scene') and sagittal n+3 on the bottom row.
    specs = [[{'rowspan': 2}] * n + [{}, {}],
             [None] * n + [{'type': 'scene'}, {}]]
    fig = make_subplots(rows=2, cols=n + 2, specs=specs,
                        horizontal_spacing=0.0, vertical_spacing=0.0)
    region_dom, block = _panel_domains(shanks)
    b0, b1 = block
    bmid = 0.5 * (b0 + b1)
    top_y, bot_y = (0.55, 1.0), (0.0, 0.45)
    #        axis -> (x0, x1, y0, y1) ; and (axis, grid row, grid col)
    place = {n + 1: (b0, bmid, *top_y),    # coronal   (top-left)
             n + 2: (bmid, b1, *top_y),    # horizontal (top-right)
             n + 3: (bmid, b1, *bot_y)}    # sagittal   (bottom-right)
    proj_grid = [(n + 1, 1, n + 1), (n + 2, 1, n + 2), (n + 3, 2, n + 2)]

    global_dmax = max(float(max(s['regions']['exit_um'].max(), s['depths'][-1]))
                      for s in shanks)
    titles = {}
    for i, s in enumerate(shanks):
        _add_region_col(fig, s['regions'], i + 1, global_dmax,
                        nav_depth if i == sel_idx else None,
                        labels=(i == sel_idx), show_axis=(i == 0))
        fig.layout[('xaxis' if i == 0 else f'xaxis{i+1}')].domain = list(region_dom[i])
        fig.layout[('yaxis' if i == 0 else f'yaxis{i+1}')].domain = [0.0, 1.0]
        # color the shank title to match its 3D track color
        titles[i + 1] = (('● ' if i == sel_idx else '') + f"{s['probe']}·sh{s['shank']}",
                         _TRACK_COLORS[i % len(_TRACK_COLORS)])

    for (label, f_role, r_role, c_role, transpose), (k, grow, gcol) in \
            zip(_PROJ, proj_grid):
        idx = int(nav_voxel[ax[f_role]])
        sl = np.take(stack3d, min(idx, stack3d.shape[ax[f_role]] - 1), axis=ax[f_role])
        rgb = np.stack([_normalize_with_pct(sl)] * 3, axis=-1).astype(float)
        aidx = min(idx, annotation_ds.shape[ax[f_role]] - 1)
        ann_sl = np.take(annotation_ds, aidx, axis=ax[f_role])
        match = ann_sl.shape == rgb.shape[:2]
        if overlay and match:
            arg = annotation_rgb(ann_sl, lookup).astype(float)
            labeled = (ann_sl != 0)[..., None]
            rgb = np.where(labeled, 0.6 * rgb + 0.4 * arg, rgb)
        # transparency: keep only the labeled brain, hide everything outside it
        # (falls back to an intensity threshold if the shapes don't line up)
        mask = (ann_sl != 0) if match else (rgb.max(axis=-1) > 8)
        # transposed views: swap image axes and the x/y roles
        xr, yr = (c_role, r_role)
        if transpose:
            rgb = np.ascontiguousarray(rgb.transpose(1, 0, 2))
            mask = np.ascontiguousarray(mask.T)
            xr, yr = (r_role, c_role)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb.astype(np.uint8), alpha])
        fig.add_trace(go.Image(z=rgba, hoverinfo='skip'), row=grow, col=gcol)
        near = np.abs(track[:, ax[f_role]] - idx) <= max(2 * ds, 6)
        seg = track[near]
        if len(seg):
            fig.add_trace(go.Scatter(x=seg[:, ax[xr]], y=seg[:, ax[yr]],
                                     mode='lines',
                                     line=dict(color='rgba(0,0,0,0.5)', width=1.5),
                                     hoverinfo='skip', showlegend=False),
                          row=grow, col=gcol)
        fig.add_trace(go.Scatter(
            x=[float(nav_voxel[ax[xr]])], y=[float(nav_voxel[ax[yr]])],
            mode='markers',
            marker=dict(symbol='square-open', size=18,
                        line=dict(color='rgba(255,255,255,0.5)', width=2)),
            hoverinfo='skip', showlegend=False), row=grow, col=gcol)
        fig.update_xaxes(visible=False, row=grow, col=gcol)
        fig.update_yaxes(visible=False, row=grow, col=gcol)
        x0, x1, y0, y1 = place[k]
        fig.layout[f'xaxis{k}'].domain = [x0, x1]
        fig.layout[f'yaxis{k}'].domain = [y0, y1]
        titles[k] = (label, 'black')

    # bottom-left: 3D labeled surface + every shank's raw points and fitted track
    if surface is not None:
        verts, faces = surface
        fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color='lightgray', opacity=0.1, hoverinfo='skip',
                                showscale=False), row=2, col=n + 1)
    for i, s in enumerate(shanks):
        c = _TRACK_COLORS[i % len(_TRACK_COLORS)]
        tr = s['track']
        fig.add_trace(go.Scatter3d(x=tr[:, 0], y=tr[:, 1], z=tr[:, 2], mode='lines',
                                   line=dict(color=c, width=6 if i == sel_idx else 3),
                                   name=s['name'], hoverinfo='name',
                                   showlegend=False), row=2, col=n + 1)
        pts = s.get('points')
        if pts is not None and len(pts):
            fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                       mode='markers', marker=dict(size=3, color=c),
                                       name=s['name'], hoverinfo='name',
                                       showlegend=False), row=2, col=n + 1)
    # default to a sagittal profile: look down the ML axis (z), AP horizontal,
    # dorsal (small DV) up
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                      aspectmode='data',
                      camera=dict(eye=dict(x=0.0, y=0.0, z=-2.2),
                                  up=dict(x=0.0, y=-1.0, z=0.0)))
    fig.layout.scene.domain = dict(x=[b0, bmid], y=list(bot_y))

    # domain-anchored titles (follow the moved 2D axes)
    for k, (text, color) in titles.items():
        fig.add_annotation(xref=('x domain' if k == 1 else f'x{k} domain'), x=0.5,
                           yref=('y domain' if k == 1 else f'y{k} domain'), y=1.0,
                           yanchor='bottom', text=text, showarrow=False,
                           font=dict(size=11, color=color))

    fig.update_layout(height=height, margin=dict(l=48, r=2, t=24, b=2),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      dragmode='zoom', font=dict(color='black'))
    return fig, nav_idx


def _napari_instructions(sel_key):
    '''Show the commands to trace/edit the shank points in napari from a Jupyter
    session (napari needs its own Qt loop, so it can't run inside Streamlit).'''
    key = {k: (v.item() if hasattr(v, 'item') else v) for k, v in sel_key.items()}
    snippet = (
        "from labdata_plugin.pluginschema import AtlasRegistration, ProbeTrack\n\n"
        f"key = {key!r}\n"
        "ar = AtlasRegistration() & key\n"
        "ar.annotate_probe_tracks_napari()      # edit the shank points in napari\n\n"
        "# when done editing in napari, save the points and refit the tracks:\n"
        "reg_key = ar.proj().fetch1()\n"
        "(ProbeTrack & reg_key).delete_quick()\n"
        "ar.save_probe_tracks_napari(update=True)\n"
        "ProbeTrack.populate(reg_key, display_progress=False, suppress_errors=True)")
    with st.expander('✏️ Edit shank tracks in napari (paste into Jupyter)',
                     expanded=False):
        st.caption('napari needs its own Qt event loop, so run these in a Jupyter '
                   'session (use `%gui qt` first). Then click "↻ Reload" below.')
        st.code(snippet, language='python')
        if st.button('↻ Reload after editing', key='ar_napari_reload',
                     help='Clear the cached fits and reload the edited annotations.'):
            _fit_shanks.clear()
            st.rerun()


def _track_annotation_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                          AtlasRegistrationAnnotation, ProbeTrack):
    sel_key = st.session_state.get('ar_selected_key')
    if not sel_key:
        st.info('Select a registration in the Sessions tab first.')
        return
    atlas = st.session_state['ar_selected_atlas']
    geometry = st.session_state['ar_selected_geometry']
    from atlas_registration import get_brainglobe_metadata
    meta = get_brainglobe_metadata(atlas)
    ax = orientation_axes_safe(meta.get('orientation', 'asr'))

    try:
        stack_ds, ds = _load_stack_ds(json.dumps(sel_key, default=str))
    except Exception as exc:
        st.error(f'Could not load the registered stack: {exc}')
        return
    n_channels = stack_ds.shape[1]
    channel = 0
    if n_channels > 1:
        channel = st.selectbox('Stack channel', list(range(n_channels)),
                               key='ar_nav_channel')
    stack3d = stack_ds[:, int(channel)]      # (AP, DV, ML)

    shanks = _fit_shanks(json.dumps(sel_key, default=str), atlas, geometry, ds)
    if not shanks:
        st.info('No shank annotations for this brain yet. Use the commands below '
                'to trace the shank tracks in napari, then reload — they will then '
                'show up here to navigate and read out.')
        _napari_instructions(sel_key)
        return
    annotation_ds, lookup = _load_annotation_ds(atlas, geometry, ds)

    # shank selector (radio) — a native widget, so switching is instant
    labels = [f"{s['probe']}·sh{s['shank']}" for s in shanks]
    if st.session_state.get('ar_nav_radio') not in labels:
        st.session_state['ar_nav_radio'] = labels[0]
    c_radio, c_over = st.columns([5, 1])
    with c_radio:
        picked = st.radio('Probe · shank', labels, horizontal=True,
                          key='ar_nav_radio')
    with c_over:
        overlay = st.toggle('Atlas overlay', value=True, key='ar_nav_overlay')
    sel_idx = labels.index(picked)
    sel = shanks[sel_idx]

    # cursor depth: a native slider (per shank), so moving it is instant
    sel_dmax = max(10.0, float(max(sel['regions']['exit_um'].max(),
                                   sel['depths'][-1])))
    skey = f'ar_navdepth_{sel["name"]}'
    st.session_state[skey] = min(float(st.session_state.get(skey, 0.0)), sel_dmax)
    nav_depth = st.slider('Cursor depth (µm)', 0.0, sel_dmax, step=10.0, key=skey)

    try:
        surface = _atlas_surface(atlas, geometry, ds)
    except Exception as exc:
        surface = None
        st.caption(f'(3D surface unavailable: {exc})')
    fig, nav_idx = _build_nav_figure(shanks, sel_idx, nav_depth, stack3d,
                                     annotation_ds, lookup, ax, ds, overlay=overlay,
                                     surface=surface)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('In-brain length (µm)', f"{sel['depths'][-1]:.0f}")
    m2.metric('AP angle (°)', f"{sel['angles']['ap']:.1f}")
    m3.metric('ML angle (°)', f"{sel['angles']['ml']:.1f}")
    m4.metric('Cursor depth (µm)', f"{sel['depths'][nav_idx]:.0f}")

    st.caption('Pick the probe·shank and drag the cursor-depth slider; hover the '
               'region columns for names.')
    st.plotly_chart(fig, key=f'ar_nav_{sel_idx}', width='stretch')

    st.divider()
    min_len = st.slider('Hide regions thinner than (µm)', 0, 200, 30, step=10,
                        key='ar_minlen')
    show = sel['regions'][sel['regions']['length_um'] >= min_len]
    st.write(f"**{sel['name']}** regions")
    st.dataframe(show[['acronym', 'name', 'entry_um', 'exit_um', 'length_um']],
                 hide_index=True, width='stretch')

    _napari_instructions(sel_key)
    if st.button(f"Refit ProbeTrack ({sel['name']})", type='primary',
                 key='ar_refit'):
        akey = dict(sel_key, annotation_id=sel['annotation_id'])
        (ProbeTrack & akey).delete_quick()
        ProbeTrack.populate(akey, display_progress=False)
        st.success(f"Refitted ProbeTrack for \"{sel['name']}\".")
        st.rerun()
