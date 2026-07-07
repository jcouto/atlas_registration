'''
Align electrophysiology features to a reconstructed probe track.

After a track has been reconstructed in atlas space (see ``probe_tracks``) the
histology already gives a first depth -> region mapping.  This module refines it
by stretching the electrode-depth axis so that electrophysiology features (LFP
power, spike-density boundaries, ...) line up with the anatomical boundaries on
the track.

The user provides *reference-depth pairs*: ``feature_ref[i]`` is a depth on the
electrophysiology axis and ``track_ref[i]`` is the depth of the matching
boundary along the histology track.  From those pairs a monotonic,
piecewise-linear function maps any electrode depth to a track depth:

    - 0 pairs : identity (electrode depth == track depth)
    - 1 pair  : constant offset
    - 2 pairs : offset + a single linear scaling between the two marks
    - n pairs : piecewise-linear scaling between consecutive marks

Depths outside the outermost pair are handled by ``extrapolate``:
``'segment'`` continues the nearest segment's slope, ``'reg'`` uses a global
linear regression through all pairs, ``'nearest'`` clamps (constant offset at
the ends).

All functions take plain numpy arrays; there is no dependency on ``labdata``.
'''
import numpy as np


def feature_to_track(depths, feature_ref=None, track_ref=None,
                     extrapolate='segment'):
    '''
    Map electrode (feature-axis) depths onto track depths using reference pairs.

    Parameters
    ----------
    depths : array-like
        Electrode depths to convert (micron).
    feature_ref, track_ref : array-like, optional
        Matched reference depths: ``feature_ref[i]`` (electrophysiology axis)
        corresponds to ``track_ref[i]`` (histology track).  With fewer than two
        pairs the map is an identity or a pure offset (see module docstring).
    extrapolate : {'segment', 'reg', 'nearest'}
        Behaviour outside the reference range.  ``'segment'`` (default)
        continues the nearest segment's slope; ``'reg'`` applies a global linear
        regression through all pairs; ``'nearest'`` clamps to the end values.

    Returns
    -------
    ndarray of track depths, same shape as ``depths``.
    '''
    depths = np.asarray(depths, dtype=float)
    if feature_ref is None or track_ref is None or len(feature_ref) == 0:
        return depths.copy()

    feature_ref = np.asarray(feature_ref, dtype=float)
    track_ref = np.asarray(track_ref, dtype=float)
    if len(feature_ref) != len(track_ref):
        raise ValueError('feature_ref and track_ref must have the same length.')

    order = np.argsort(feature_ref)
    fx, tx = feature_ref[order], track_ref[order]

    if len(fx) == 1:
        return depths + (tx[0] - fx[0])

    out = np.interp(depths, fx, tx)  # piecewise-linear within the reference range

    below = depths < fx[0]
    above = depths > fx[-1]
    if extrapolate == 'nearest':
        pass  # np.interp already clamps to the end values
    elif extrapolate == 'reg':
        slope, intercept = np.polyfit(fx, tx, 1)
        out[below] = slope * depths[below] + intercept
        out[above] = slope * depths[above] + intercept
    elif extrapolate == 'segment':
        s_lo = (tx[1] - tx[0]) / (fx[1] - fx[0])
        s_hi = (tx[-1] - tx[-2]) / (fx[-1] - fx[-2])
        out[below] = tx[0] + s_lo * (depths[below] - fx[0])
        out[above] = tx[-1] + s_hi * (depths[above] - fx[-1])
    else:
        raise ValueError(f"unknown extrapolate mode {extrapolate!r}.")
    return out


def track_to_feature(track_depths, feature_ref, track_ref,
                     extrapolate='segment'):
    '''Inverse of :func:`feature_to_track` (track depth -> electrode depth).'''
    return feature_to_track(track_depths, feature_ref=track_ref,
                            track_ref=feature_ref, extrapolate=extrapolate)


def electrode_to_atlas(electrode_depths, insertion_depth, feature_ref=None,
                       track_ref=None):
    '''
    Map electrode depth (measured from the probe TIP, 0 = tip) to atlas depth
    (from the brain surface, 0 = surface).

    The baseline is the geometric inversion ``atlas = insertion_depth -
    electrode`` (the tip sits at ``insertion_depth`` from the surface, and moving
    up the shank by ``electrode`` moves ``electrode`` closer to the surface).
    Reference pairs ``(feature_ref[i], track_ref[i]) = (electrode depth, atlas
    depth)`` refine it as a piecewise-linear function; beyond the outermost pair
    the slope is the physical -1 (1 µm up the shank = 1 µm shallower).

      - 0 pairs : pure inversion (uses ``insertion_depth`` only)
      - 1 pair  : inversion shifted to pass through the pair (== adjusting the
                  insertion depth)
      - n pairs : piecewise-linear through the pairs, slope -1 outside
    '''
    e = np.asarray(electrode_depths, dtype=float)
    if feature_ref is None or track_ref is None or len(feature_ref) == 0:
        return insertion_depth - e
    fx = np.asarray(feature_ref, dtype=float)
    tx = np.asarray(track_ref, dtype=float)
    order = np.argsort(fx)
    fx, tx = fx[order], tx[order]
    if len(fx) == 1:
        return tx[0] - (e - fx[0])
    a = np.interp(e, fx, tx)
    below, above = e < fx[0], e > fx[-1]
    a[below] = tx[0] - (e[below] - fx[0])
    a[above] = tx[-1] - (e[above] - fx[-1])
    return a


def atlas_to_electrode(atlas_depths, insertion_depth, feature_ref=None,
                       track_ref=None):
    '''Inverse of :func:`electrode_to_atlas` (atlas depth -> electrode depth).'''
    a = np.asarray(atlas_depths, dtype=float)
    if feature_ref is None or track_ref is None or len(feature_ref) == 0:
        return insertion_depth - a
    fx = np.asarray(feature_ref, dtype=float)
    tx = np.asarray(track_ref, dtype=float)
    order = np.argsort(tx)           # invert: interpolate atlas -> electrode
    ts, fs = tx[order], fx[order]
    if len(ts) == 1:
        return fs[0] - (a - ts[0])
    e = np.interp(a, ts, fs)
    below, above = a < ts[0], a > ts[-1]
    e[below] = fs[0] - (a[below] - ts[0])
    e[above] = fs[-1] - (a[above] - ts[-1])
    return e


def track_depth_to_voxel(track_depths, samples):
    '''
    Interpolate atlas-voxel coordinates at the given track depths.

    ``samples`` is the DataFrame from
    ``probe_tracks.sample_annotation_along_track`` (needs ``depth_um`` and the
    ``x, y, z`` voxel columns).  Returns an (N, 3) float array of voxels.
    '''
    d = samples['depth_um'].to_numpy()
    xyz = np.stack([np.interp(track_depths, d, samples[c].to_numpy())
                    for c in ('x', 'y', 'z')], axis=1)
    return xyz


def align_channels_to_regions(channel_depths, samples, feature_ref=None,
                              track_ref=None, annotation=None, lookup=None,
                              resolution=(10., 10., 10.),
                              extrapolate='segment', insertion_depth=None):
    '''
    Assign every recording channel to an atlas voxel and region.

    Applies the reference-pair depth stretch (:func:`feature_to_track`) to the
    electrode depths, then reads the region at each resulting position on the
    track.  This is the per-channel depth -> region table for a probe.

    Parameters
    ----------
    channel_depths : array-like
        Depth of each channel along the probe (micron), same coordinate as
        ``feature_ref``.
    samples : DataFrame
        ``sample_annotation_along_track`` output (defines the track geometry and,
        if it carries ``region_id``, the fallback regions).
    feature_ref, track_ref : array-like, optional
        Reference-depth pairs (see :func:`feature_to_track`).  Omit for the raw
        histology mapping with no ephys-feature correction.
    annotation : 3D int array, optional
        Atlas annotation, re-read at the aligned voxels for exact region ids.
        If omitted, region ids are taken (nearest) from ``samples``.
    lookup : dict, optional
        ``atlasutils.get_structure_lookup`` output for acronym/name.
    resolution : (3,) array-like
        Voxel size (micron) per axis (kept for API symmetry / future options).
    extrapolate : str
        Passed through to :func:`feature_to_track`.

    Returns
    -------
    DataFrame per channel: ``depth_um`` (electrode), ``track_depth_um``,
    ``x, y, z`` (voxel), ``region_id, acronym, name``.
    '''
    import pandas as pd
    channel_depths = np.asarray(channel_depths, dtype=float)
    if insertion_depth is not None:
        # channel_depths are electrode depths from the tip; map to atlas depth
        track_depths = electrode_to_atlas(channel_depths, insertion_depth,
                                          feature_ref, track_ref)
    else:
        track_depths = feature_to_track(channel_depths, feature_ref, track_ref,
                                        extrapolate=extrapolate)
    xyz = track_depth_to_voxel(track_depths, samples)

    if annotation is not None:
        vox = np.clip(np.rint(xyz).astype(int), 0,
                      np.asarray(annotation.shape) - 1)
        rid = annotation[vox[:, 0], vox[:, 1], vox[:, 2]].astype(int)
    else:
        d = samples['depth_um'].to_numpy()
        rids = samples['region_id'].to_numpy()
        nearest = np.searchsorted(d, track_depths).clip(0, len(d) - 1)
        rid = rids[nearest].astype(int)

    if lookup is not None:
        acr = np.array([lookup['id_to_acronym'].get(int(i), '') for i in rid])
        nm = np.array([lookup['id_to_name'].get(int(i), '') for i in rid])
    else:
        acr = np.array([''] * len(rid))
        nm = np.array([''] * len(rid))

    return pd.DataFrame(dict(depth_um=channel_depths,
                             track_depth_um=track_depths,
                             x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                             region_id=rid, acronym=acr, name=nm))


# ---------------------------------------------------------------------------
# optional ephys feature images (array-in / array-out, for the dashboard)
# ---------------------------------------------------------------------------
def spike_depth_image(spike_depths, spike_times=None, weights=None,
                      depth_bin=20., n_time_bins=None, time_bin=None):
    '''
    2D (depth x time) density of spikes, a common feature panel for alignment.

    Parameters
    ----------
    spike_depths : array-like
        Depth of each spike along the probe (micron).
    spike_times : array-like, optional
        Spike times (s).  If omitted a single depth histogram is returned.
    weights : array-like, optional
        Per-spike weight (e.g. amplitude); defaults to counts.
    depth_bin : float
        Depth bin size (micron).
    n_time_bins, time_bin : optional
        Time binning; ``time_bin`` (s) takes precedence over ``n_time_bins``.

    Returns
    -------
    (image, depth_edges, time_edges).  When ``spike_times`` is None the image is
    1D (per-depth counts) and ``time_edges`` is None.
    '''
    spike_depths = np.asarray(spike_depths, dtype=float)
    dmin, dmax = np.nanmin(spike_depths), np.nanmax(spike_depths)
    depth_edges = np.arange(dmin, dmax + depth_bin, depth_bin)
    if spike_times is None:
        img, _ = np.histogram(spike_depths, bins=depth_edges, weights=weights)
        return img, depth_edges, None
    spike_times = np.asarray(spike_times, dtype=float)
    if time_bin is not None:
        time_edges = np.arange(spike_times.min(), spike_times.max() + time_bin,
                               time_bin)
    else:
        time_edges = np.linspace(spike_times.min(), spike_times.max(),
                                 (n_time_bins or 100) + 1)
    img, _, _ = np.histogram2d(spike_depths, spike_times,
                               bins=[depth_edges, time_edges], weights=weights)
    return img, depth_edges, time_edges


def lfp_power_by_depth(lfp, channel_depths, fs, freq_band=(0., 300.)):
    '''
    Mean LFP power in a frequency band per channel, sorted by depth.

    Parameters
    ----------
    lfp : (n_channels, n_samples) array
        LFP traces.
    channel_depths : array-like
        Depth (micron) of each channel/row of ``lfp``.
    fs : float
        Sampling rate (Hz).
    freq_band : (low, high)
        Frequency band (Hz) to integrate power over.

    Returns
    -------
    (power, depths_sorted): band power per channel and the matching depths,
    both sorted by increasing depth.  Requires scipy.
    '''
    from scipy.signal import welch
    lfp = np.asarray(lfp, dtype=float)
    channel_depths = np.asarray(channel_depths, dtype=float)
    f, psd = welch(lfp, fs=fs, axis=1)
    band = (f >= freq_band[0]) & (f <= freq_band[1])
    power = psd[:, band].mean(axis=1)
    order = np.argsort(channel_depths)
    return power[order], channel_depths[order]
