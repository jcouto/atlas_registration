'''
Core, standalone functions to reconstruct a probe track in atlas space.

The workflow:

    1. the user marks points along the probe track in the registered atlas volume
    2. ``fit_track_line`` fits a straight trajectory to those points
    3. ``sample_annotation_along_track`` looks up the atlas annotation at each
       point along the trajectory
    4. ``regions_along_track`` collapses that into the ordered list of regions
       the probe traversed, with entry/exit depth and length
    5. ``assign_channels_to_regions`` counts how many recording channels fall in
       each region.

'''
import numpy as np


# ---------------------------------------------------------------------------
# orientation helpers
# ---------------------------------------------------------------------------
# brainglobe orientation codes (e.g. "asr") describe the ORIGIN of each axis:
#   a/p -> anterior-posterior (AP) axis
#   s/i -> superior-inferior  (DV / depth) axis
#   l/r -> left-right         (ML) axis
_AXIS_ROLE = {'a': 'ap', 'p': 'ap',
              's': 'dv', 'i': 'dv',
              'l': 'ml', 'r': 'ml'}


def orientation_axes(orientation='asr'):
    '''Map a brainglobe orientation code to axis indices.

    Returns a dict ``{'ap': i, 'dv': j, 'ml': k}`` giving which volume axis is
    the antero-posterior, dorso-ventral and medio-lateral one.
    '''
    roles = [_AXIS_ROLE[c] for c in orientation.lower()]
    return {role: roles.index(role) for role in ('ap', 'dv', 'ml')}


# ---------------------------------------------------------------------------
# trajectory fitting
# ---------------------------------------------------------------------------
def fit_track_line(points, volume_shape=None, extent=4000, spacing=2,
                   n_line=10000, orientation='asr'):
    '''
    Fit a straight trajectory (total-least-squares / SVD) to marked points.

    Parameters
    ----------
    points : (N, 3) array
        Voxel coordinates marked along the probe track, in atlas-volume index
        order (axis0, axis1, axis2).
    volume_shape : (3,) array-like, optional
        Shape of the atlas volume; the returned voxels are clipped to it.
    extent : float
        Half-length (in voxels) over which to extend the fitted line before
        clipping to the volume.
    spacing : float
        Spacing (in voxels) of the coarse line used to find the entry/exit.
    n_line : int
        Number of samples returned along the trajectory (Bresenham).
    orientation : str
        brainglobe orientation code, used to report insertion angles.

    Returns
    -------
    dict with keys:
        ``track_voxels``  : (M, 3) int  ordered voxels along the trajectory
        ``direction``     : (3,) float  unit direction (points[0] -> points[-1] sense)
        ``centroid``      : (3,) float  mean of the input points
        ``entry``         : (3,) float  first (most dorsal) voxel in the volume
        ``exit``          : (3,) float  last  (most ventral) voxel in the volume
        ``angles``        : dict        {'ap': deg, 'ml': deg} from the DV axis
    '''
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be an (N, 3) array of voxel coordinates.')
    if len(points) < 2:
        raise ValueError('need at least 2 points to fit a trajectory.')

    ax = orientation_axes(orientation)
    centroid = points.mean(axis=0)
    _, _, vv = np.linalg.svd(points - centroid)
    direction = vv[0]
    # Orient the trajectory dorsal -> ventral (increasing DV axis) so that the
    # entry (depth 0) is always the most dorsal end / cortical surface,
    # regardless of the order the points were marked.  This keeps the region
    # read-out the right way up (surface on top).  Fall back to the marked sense
    # only for (unusual) tracks that run almost flat in DV.
    if abs(direction[ax['dv']]) > 1e-3:
        if direction[ax['dv']] < 0:
            direction = -direction
    elif np.dot(points[-1] - points[0], direction) < 0:
        direction = -direction

    tt = np.arange(-extent, extent, spacing)[:, np.newaxis]
    linepoints = direction * tt + centroid
    inside = (linepoints > 0).sum(axis=1) == 3
    if volume_shape is not None:
        inside &= (linepoints < np.asarray(volume_shape)).sum(axis=1) == 3
    linepoints = linepoints[inside]
    if not len(linepoints):
        raise ValueError('fitted trajectory falls outside the volume.')

    entry, exit_ = linepoints[0], linepoints[-1]
    track = bresenhamlines(entry.reshape(1, -1), exit_.reshape(1, -1),
                           n_line).squeeze()
    keep = (track > 0).sum(axis=1) == 3
    if volume_shape is not None:
        keep &= (track < np.asarray(volume_shape)).sum(axis=1) == 3
    track = track[keep].astype(int)

    # insertion angles relative to the dorso-ventral (depth) axis
    d = direction / (np.linalg.norm(direction) + 1e-12)
    dv = d[ax['dv']]
    angles = dict(
        ap=float(np.rad2deg(np.arctan2(abs(d[ax['ap']]), abs(dv)))),
        ml=float(np.rad2deg(np.arctan2(abs(d[ax['ml']]), abs(dv)))))

    return dict(track_voxels=track,
                direction=direction,
                centroid=centroid,
                entry=entry,
                exit=exit_,
                angles=angles)


def get_line_trajectory(data, volume_shape=None):
    '''
    Backwards-compatible wrapper: return only the ordered voxels of the fitted
    straight trajectory.  See ``fit_track_line`` for the full result.
    '''
    return fit_track_line(data, volume_shape=volume_shape)['track_voxels']


def trim_track_to_labeled(track_voxels, annotation, background=0):
    '''
    Clip a trajectory to the labeled brain.

    ``fit_track_line`` extends the line to the volume's bounding box, so its ends
    usually sit in unlabeled space (annotation == ``background``) above/below the
    brain.  This keeps only the span between the first and last labeled voxel, so
    that track depth 0 is the brain surface.

    Returns the trimmed ``(M, 3)`` int voxel array (unchanged if nothing is
    labeled along the track).
    '''
    track_voxels = np.asarray(track_voxels, dtype=int)
    shape = np.asarray(annotation.shape)
    inside = np.all((track_voxels >= 0) & (track_voxels < shape), axis=1)
    ids = np.zeros(len(track_voxels), dtype=annotation.dtype)
    tv = track_voxels[inside]
    ids[inside] = annotation[tv[:, 0], tv[:, 1], tv[:, 2]]
    labeled = np.where(ids != background)[0]
    if not len(labeled):
        return track_voxels
    return track_voxels[labeled[0]:labeled[-1] + 1]


# ---------------------------------------------------------------------------
# sampling the annotation along a track
# ---------------------------------------------------------------------------
def _track_depths_um(track_voxels, resolution):
    '''Cumulative distance (micron) from the track start for each voxel.'''
    resolution = np.asarray(resolution, dtype=float)
    steps = np.diff(track_voxels.astype(float), axis=0) * resolution
    seglen = np.sqrt((steps ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seglen)])


def sample_annotation_along_track(track_voxels, annotation, lookup=None,
                                  resolution=(10., 10., 10.)):
    '''
    Look up the atlas annotation at every voxel along a trajectory.

    Parameters
    ----------
    track_voxels : (M, 3) int array
        Ordered voxels along the trajectory (from ``fit_track_line``).
    annotation : 3D int array
        The atlas annotation volume (structure id per voxel).
    lookup : dict, optional
        Output of ``atlasutils.get_structure_lookup`` to resolve acronym/name.
    resolution : (3,) array-like
        Voxel size (micron) per axis, used to compute depth.

    Returns
    -------
    pandas.DataFrame with columns ``depth_um, region_id, acronym, name,
    x, y, z`` (one row per voxel along the track).
    '''
    import pandas as pd
    track_voxels = np.asarray(track_voxels, dtype=int)
    shape = np.asarray(annotation.shape)
    inside = np.all((track_voxels >= 0) & (track_voxels < shape), axis=1)
    track_voxels = track_voxels[inside]

    ids = annotation[track_voxels[:, 0], track_voxels[:, 1], track_voxels[:, 2]]
    depth = _track_depths_um(track_voxels, resolution)

    if lookup is not None:
        acr = np.array([lookup['id_to_acronym'].get(int(i), '') for i in ids])
        nm = np.array([lookup['id_to_name'].get(int(i), '') for i in ids])
    else:
        acr = np.array([''] * len(ids))
        nm = np.array([''] * len(ids))

    return pd.DataFrame(dict(depth_um=depth,
                             region_id=ids.astype(int),
                             acronym=acr,
                             name=nm,
                             x=track_voxels[:, 0],
                             y=track_voxels[:, 1],
                             z=track_voxels[:, 2]))


def roll_up_region_ids(region_ids, lookup, level=None, target_acronyms=None):
    '''
    Map fine structure ids to a coarser hierarchy level.

    ``level`` (int) keeps the ancestor at that depth of the Allen
    ``structure_id_path`` (0 = root).  ``target_acronyms`` (iterable) instead
    rolls each id up to the first ancestor whose acronym is in the set (useful
    to summarise a track by a chosen list of regions).  If neither is given the
    ids are returned unchanged.
    '''
    id_to_path = lookup['id_to_path']
    acronym_to_id = lookup['acronym_to_id']
    out = []
    if target_acronyms is not None:
        targets = {acronym_to_id[a] for a in target_acronyms if a in acronym_to_id}
        for rid in region_ids:
            path = id_to_path.get(int(rid), [int(rid)])
            match = next((p for p in reversed(path) if p in targets), int(rid))
            out.append(match)
    elif level is not None:
        for rid in region_ids:
            path = id_to_path.get(int(rid), [int(rid)])
            out.append(path[min(level, len(path) - 1)])
    else:
        out = [int(r) for r in region_ids]
    return np.array(out, dtype=int)


def regions_along_track(samples, lookup=None):
    '''
    Collapse per-voxel samples into the ordered list of regions traversed.

    Parameters
    ----------
    samples : DataFrame
        Output of ``sample_annotation_along_track``.  The ``region_id`` column
        may already be rolled up to a coarser level.
    lookup : dict, optional
        To (re)fill acronym/name/rgb columns from ``region_id``.

    Returns
    -------
    DataFrame with one row per contiguous region segment and columns
    ``region_id, acronym, name, entry_um, exit_um, length_um, rgb``.
    '''
    import pandas as pd
    if not len(samples):
        return pd.DataFrame(columns=['region_id', 'acronym', 'name',
                                     'entry_um', 'exit_um', 'length_um', 'rgb'])
    ids = samples['region_id'].to_numpy()
    depth = samples['depth_um'].to_numpy()
    # boundaries where the region id changes
    change = np.concatenate([[0], np.where(np.diff(ids) != 0)[0] + 1, [len(ids)]])
    rows = []
    for a, b in zip(change[:-1], change[1:]):
        rid = int(ids[a])
        entry = float(depth[a])
        # exit is the depth of the next boundary (or the last sample depth)
        exit_ = float(depth[b]) if b < len(depth) else float(depth[-1])
        if lookup is not None:
            acr = lookup['id_to_acronym'].get(rid, '')
            nm = lookup['id_to_name'].get(rid, '')
            rgb = lookup['id_to_rgb'].get(rid, [0, 0, 0])
        else:
            acr = samples['acronym'].iloc[a] if 'acronym' in samples else ''
            nm = samples['name'].iloc[a] if 'name' in samples else ''
            rgb = [0, 0, 0]
        rows.append(dict(region_id=rid, acronym=acr, name=nm,
                         entry_um=entry, exit_um=exit_,
                         length_um=exit_ - entry, rgb=rgb))
    return pd.DataFrame(rows)


def assign_channels_to_regions(track_regions, channel_depths):
    '''
    Assign recording channels to regions and count channels per region.

    Parameters
    ----------
    track_regions : DataFrame
        Output of ``regions_along_track`` (``entry_um``/``exit_um`` segments).
    channel_depths : array-like
        Depth of each channel (micron) expressed in the SAME track-depth
        coordinate as ``track_regions`` (0 = track entry).  Use
        ``probe_alignment`` to convert electrode depths into this coordinate.

    Returns
    -------
    (channel_regions, region_counts):
        channel_regions : DataFrame  per channel: ``depth_um, region_id,
                          acronym, name``
        region_counts   : DataFrame  ``regions_along_track`` with an added
                          ``n_channels`` column.
    '''
    import pandas as pd
    channel_depths = np.asarray(channel_depths, dtype=float)
    entry = track_regions['entry_um'].to_numpy()
    exit_ = track_regions['exit_um'].to_numpy()

    idx = np.full(len(channel_depths), -1, dtype=int)
    for i, (lo, hi) in enumerate(zip(entry, exit_)):
        idx[(channel_depths >= lo) & (channel_depths < hi)] = i
    # channels at or beyond the last exit fall in the last segment
    idx[channel_depths >= exit_[-1]] = len(exit_) - 1 if len(exit_) else -1

    rid = np.array([track_regions['region_id'].iloc[i] if i >= 0 else -1
                    for i in idx])
    acr = np.array([track_regions['acronym'].iloc[i] if i >= 0 else ''
                    for i in idx])
    nm = np.array([track_regions['name'].iloc[i] if i >= 0 else ''
                   for i in idx])
    channel_regions = pd.DataFrame(dict(depth_um=channel_depths,
                                        region_id=rid, acronym=acr, name=nm))

    counts = channel_regions.groupby('region_id').size()
    region_counts = track_regions.copy()
    region_counts['n_channels'] = [int(counts.get(r, 0))
                                   for r in region_counts['region_id']]
    return channel_regions, region_counts


# ---------------------------------------------------------------------------
# Bresenham line (adapted from
# https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/)
# ---------------------------------------------------------------------------
def _bresenhamline_nslope(slope):
    '''Normalize slope for Bresenham's line algorithm.'''
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def bresenhamlines(start, end, max_iter):
    '''
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)
    '''
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat
    return np.array(np.rint(bline), dtype=start.dtype)
