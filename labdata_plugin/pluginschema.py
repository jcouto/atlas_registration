from labdata.schema import *

username = prefs['database']['database.user']
atlas_schema = None
if 'atlas_registration_schema' in prefs.keys(): # to be able to override to another name
    atlas_schema = prefs['atlas_registration_schema']

if atlas_schema is None:
    atlas_schema = get_user_schema()
else:
    if 'root' in atlas_schema:
        raise(ValueError('[atlas_registration] "atlas_registration_schema" must be specified in the preference file to run as root.'))
    atlas_schema = dj.schema(atlas_schema)

__all__ = ['AtlasRegistrationAnnotation',
           'AtlasRegistrationParams',
           'AtlasRegistration',
           'ProbeTrack',
           'ProbeAlignment',
           'UnitAlignment']


def _atlas_annotation_and_lookup(atlas):
    '''Load the atlas annotation volume, metadata and structure lookup once.'''
    from atlas_registration import (get_brainglobe_annotation,
                                    get_brainglobe_metadata,
                                    get_structure_lookup)
    meta = get_brainglobe_metadata(atlas)
    lookup = get_structure_lookup(atlas)
    return meta, lookup


from functools import lru_cache


@lru_cache(maxsize=8)
def _cached_surface(atlas, geometry, step_size=2, downsample=4):
    '''Marching-cubes brain surface (verts, faces), verts scaled back to full-res
    atlas voxels so they overlay the stored track/channel/unit voxels.'''
    from atlas_registration import get_brainglobe_annotation, plotting
    ann = get_brainglobe_annotation(atlas, geometry)
    ann = ann[::downsample, ::downsample, ::downsample]
    verts, faces = plotting.atlas_surface(ann, step_size=step_size)
    return verts * downsample, faces

@atlas_schema
class AtlasRegistrationParams(dj.Manual):
    definition   = '''
    -> FixedBrainTransform
    atlas_reg_id = 0                       : smallint
    ---
    atlas                                  : varchar(36)
    orientation = "asl"                    : varchar(6)
    brain_geometry = "left"                : varchar(24)
    number_of_resolutions = 4              : int  
    number_of_resolutions_second = 6       : int
    final_grid_spacing = 15.0              : float
    number_of_histogram_bins = 32          : int 
    maximum_number_of_interactions = 2500  : int
    number_of_spatial_samples = 4000       : int
    stack_gaussian_smoothing = NULL        : int
    '''

@atlas_schema
class AtlasRegistration(dj.Computed):
    default_container = "labdata-atlasreg"
    shank_names = None
    shank_layers = None
    definition = '''
    -> AtlasRegistrationParams
    ---
    elastix_transforms                    : longblob
    -> [nullable] AnalysisFile
    '''
    def make(self,key):
        par = (AtlasRegistrationParams() & key).fetch1()
        stack = FixedBrainTransform().transform(key)
        from atlas_registration import elastix_register_brain
        registered, transforms = elastix_register_brain(
            stack[:,0],
            atlas=par['atlas'],
            brain_geometry = par['brain_geometry'],
            number_of_resolutions = par['number_of_resolutions'],
            number_of_resolutions_second = par['number_of_resolutions_second'],
            final_grid_spacing = par['final_grid_spacing'],
            number_of_histogram_bins = par['number_of_histogram_bins'],
            maximum_number_of_interactions = par['maximum_number_of_interactions'],
            number_of_spatial_samples = par['number_of_spatial_samples'],
            stack_gaussian_smoothing = par['stack_gaussian_smoothing'])
        from atlas_registration import elastix_apply_transform
        # elastix can not run in parallel (need to have )
        na = [elastix_apply_transform(s, transforms) for s in stack.transpose(1,0,2,3)]
        na = np.stack(na).transpose(1,0,2,3)
        # save file with the result and upload to the analysis bucket.
        folder_path = (((Path(prefs['local_paths'][0])
                         /schema_project/key['subject_name']))
                       /key['session_name'])/f'brain_transform_{key["transform_id"]}'
        filepath = folder_path/f'atlas_reg_{key["atlas_reg_id"]}.ome.tif'
        folder_path.mkdir(exist_ok=True, parents=True)
        from tifffile import imwrite  # saving in tiff so it is easier to read
        imwrite(filepath, na, 
                imagej = True,
                metadata={'axes': 'ZCYX'}, 
                compression ='zlib',
                compressionargs = {'level': 6})
        added = AnalysisFile().upload_files([filepath],
                                            dict(subject_name = key['subject_name'],
                                                 session_name = key['session_name'],
                                                 dataset_name = f'brain_transform_{key["transform_id"]}'))[0]
        self.insert1(dict(key,
                          elastix_transforms = transforms,
                          **added))
    def get_stack(self):
        files = (AnalysisFile() & self).get()
        from tifffile import imread
        stacks = [imread(f) for f in files]
        if len(stacks) == 1:
            return stacks[0]
        else: 
            return stacks
        
    def get_reference(self):
        atlas = (self*AtlasRegistrationParams).fetch1('atlas')
        geometry = (self*AtlasRegistrationParams).fetch1('brain_geometry')
        from atlas_registration import get_brainglobe_annotation
        return get_brainglobe_annotation(atlas,geometry)

    def surface(self, step_size=2, downsample=4):
        '''Brain surface ``(verts, faces)`` for 3D plots (cached), in full-res
        atlas voxels so it overlays stored track/channel/unit voxels.'''
        p = (self * AtlasRegistrationParams).fetch1('atlas', 'brain_geometry')
        return _cached_surface(p[0], p[1], step_size, downsample)
    
    def napari_open(self,color = False, **kwargs):
        if color:
            kwargs['channel_axis'] = 1
        stack = self.get_stack()
        from labdata.stacks import napari_open
        napari_open(stack,**kwargs)

    def get1(self):
        '''Get the shank names and points'''
        if len(self) == 0:
            raise(ValueError('No brain to annotate.'))
        if len(self) > 1:
            raise(ValueError('Select only one brain.'))
        key = self.proj().fetch1()
        unique_probes = (Probe() & 
                         (EphysRecording.ProbeSetting() &
                          (Subject & key))).fetch(as_dict = True)
        shank_names = []
        for p in unique_probes:
            for i in range(p['probe_n_shanks']):
                shank_names.append(f"{p['probe_id']}_shank{i}")
        self.shank_names = shank_names
        self.shank_points = []
        for i,shank in enumerate(self.shank_names):
            dd = (AtlasRegistrationAnnotation() & self & dict(annotation_name = shank)).fetch(as_dict = True)
            if len(dd):
                self.shank_points.append(dd[0]['xyz'])
            else:
                self.shank_points.append([])
        return self # returns because it is a get method..
    
    def annotate_probe_tracks_napari(self):
        ''' Annotate probe tracks for electrophysiology.'''
        self.get1()
        stack = self.get_stack()
        
        self.shank_layers = []
        import pylab as plt
        colormap = plt.colormaps['tab10']
        colors = [plt.matplotlib.colors.to_hex(c) for c in colormap(range(10))]
        
        import napari
        viewer = napari.Viewer()
        im = viewer.add_image(stack,channel_axis = 1)
        
        for i,(shank,points) in enumerate(zip(self.shank_names,self.shank_points)):
            par = dict(name = shank,
                       ndim=3,
                       size=5,
                       opacity=1,
                       face_color=colors[np.mod(i,len(colors))])
            if len(points) > 0:
                par['data'] = points
            self.shank_layers.append(viewer.add_points(**par))
        viewer.show()
        return self
    
    def save_probe_tracks_napari(self, update=False):
        ''' Save the probe tracks in AtlasRegistrationAnnotation.

        With ``update=True`` existing annotations are **updated in place**
        (``update1``) rather than replaced — a REPLACE would delete the annotation
        row and fail the ProbeTrack foreign key.  Refit the tracks afterwards
        (``refit_tracks`` / ``ProbeTrack.populate``).
        '''
        key = (self).proj().fetch1()
        toadd = []
        for i, (name, layer) in enumerate(zip(self.shank_names, self.shank_layers)):
            toadd.append(dict(key, annotation_id=i, annotation_name=name,
                              annotation_type='shank', xyz=layer.data))
        pk = AtlasRegistrationAnnotation.primary_key
        n_new = n_upd = 0
        for row in toadd:
            exists = len(AtlasRegistrationAnnotation & {k: row[k] for k in pk})
            if exists and update:
                AtlasRegistrationAnnotation.update1(row)   # updates xyz, no FK delete
                n_upd += 1
            elif not exists:
                AtlasRegistrationAnnotation.insert1(row)
                n_new += 1
            # exists and not update -> leave it as is
        print(f"Saved {len(toadd)} shank annotations ({n_new} new, {n_upd} updated).")

    def refit_tracks(self, restore_alignments=True, safemode=True,
                     display_progress=True):
        '''Save landmarks (from napari if connected), then delete + refit the
        ProbeTracks of this brain and (optionally) restore the ProbeAlignments.

        Steps: back up the alignments' reference-pair config (if
        ``restore_alignments``); delete the ProbeTracks (cascading ProbeAlignment
        + UnitAlignment); if a napari session is connected, save its edited
        landmarks (done *after* the delete so the annotations have no dependents to
        block the replace); refit the tracks; then replay the alignments.

        Parameters
        ----------
        restore_alignments : bool
            Replay the previous ProbeAlignments onto the refitted tracks (default).
            Set ``False`` to drop them and start the alignments from scratch.
        safemode : bool
            If ``True`` (default) DataJoint asks for confirmation before the delete.

        It does **not** recompute ``UnitAlignment`` — check the refitted tracks and
        alignments, then run ``UnitAlignment.populate()`` yourself.
        '''
        import datajoint as dj
        reg_key = self.proj().fetch1()          # a single brain

        # 1) back up the alignments before deleting the tracks cascades them away
        aligns = []
        if restore_alignments:
            keep = ('annotation_id', 'alignment_id', 'feature_ref', 'track_ref',
                    'insertion_depth', 'extrapolate', 'alignment_user', 'confidence')
            aligns = [{k: a[k] for k in keep if k in a}
                      for a in (ProbeAlignment & reg_key).fetch(as_dict=True)]
            print(f'Backed up {len(aligns)} alignment(s).')

        # 2) delete the ProbeTracks (cascades ProbeAlignment + UnitAlignment).
        #    safemode=True asks the user first; False deletes without prompting.
        _sm = dj.config['safemode']
        dj.config['safemode'] = safemode
        try:
            (ProbeTrack & reg_key).delete()
        finally:
            dj.config['safemode'] = _sm

        # 3) persist the napari-edited landmarks if a session is connected
        if self.shank_layers is not None:
            self.save_probe_tracks_napari(update=True)

        # 4) refit the tracks from the current landmarks
        ProbeTrack.populate(reg_key, display_progress=display_progress,
                            suppress_errors=True)

        # 5) replay the alignments onto the refitted tracks
        ok = 0
        for a in aligns:
            tkey = dict(reg_key, annotation_id=a['annotation_id'])
            if not len(ProbeTrack & tkey):
                print(f"  skip alignment (no track for annotation_id={a['annotation_id']}).")
                continue
            ProbeAlignment().align(
                tkey, feature_ref=a.get('feature_ref'), track_ref=a.get('track_ref'),
                alignment_id=int(a['alignment_id']),
                insertion_depth=(None if a.get('insertion_depth') is None
                                 else float(a['insertion_depth'])),
                replace=True, alignment_user=a.get('alignment_user'),
                confidence=a.get('confidence'))
            ok += 1
        msg = f'Refitted {len(ProbeTrack & reg_key)} track(s)'
        if restore_alignments:
            msg += f'; restored {ok}/{len(aligns)} alignment(s)'
        print(msg + '.\nCheck them, then run UnitAlignment.populate().')
        return self

@atlas_schema
class AtlasRegistrationAnnotation(dj.Manual):
    definition = '''
    -> AtlasRegistration
    annotation_id : int
    ---
    annotation_name : varchar(36)
    annotation_type : varchar(36)
    xyz : blob
    '''


@atlas_schema
class BrainRegion(dj.Manual):
    '''Atlas region metadata (acronym, name, colour) keyed by ``(atlas,
    region_id)``.  The per-region/channel/unit tables store only ``region_id``
    (+ ``acronym`` for convenience); join here for the full ``name``/``rgb``.
    Filled on demand from ``atlas_registration.get_structure_lookup``.'''
    definition = '''
    atlas      : varchar(64)
    region_id  : int          # atlas structure id
    ---
    acronym    : varchar(32)
    name       : varchar(255)
    rgb        : blob          # [r, g, b] display color
    '''

    @classmethod
    def ensure(cls, atlas):
        '''Populate all regions of ``atlas`` (idempotent).'''
        import numpy as np
        from atlas_registration import get_structure_lookup
        lk = get_structure_lookup(atlas)
        rows = [dict(atlas=atlas, region_id=int(rid),
                     acronym=str(lk['id_to_acronym'].get(rid, '')),
                     name=str(lk['id_to_name'].get(rid, '')),
                     rgb=np.asarray(lk['id_to_rgb'].get(rid, [0, 0, 0]), dtype=int))
                for rid in lk['id_to_acronym']]
        cls().insert(rows, skip_duplicates=True)

    @classmethod
    def rgb_map(cls, atlas):
        '''``{region_id: [r, g, b]}`` for an atlas (for colouring plots).'''
        import numpy as np
        cls.ensure(atlas)
        rid, rgb = (cls() & dict(atlas=atlas)).fetch('region_id', 'rgb')
        return {int(i): np.asarray(c, int).tolist() for i, c in zip(rid, rgb)}


@atlas_schema
class ProbeTrack(dj.Computed):
    '''
    Straight trajectory fitted to a shank annotation and the atlas regions it
    traverses.  One row per ``AtlasRegistrationAnnotation`` of ``annotation_type
    = 'shank'``.  The heavy lifting is done by the standalone
    ``atlas_registration`` core (``fit_track_line`` / ``regions_along_track``);
    this table only caches the read-out and the trajectory geometry.
    '''
    definition = '''
    -> AtlasRegistrationAnnotation
    ---
    probe_id         : varchar(32)  # from the annotation name (joins to sortings)
    shank            : int          # shank index on the probe
    entry_voxel      : blob       # first (most dorsal) voxel inside the volume
    exit_voxel       : blob       # last (most ventral) voxel inside the volume
    direction        : blob       # unit direction vector of the trajectory
    ap_angle         : float      # insertion angle from the DV axis in the AP plane (deg)
    ml_angle         : float      # insertion angle from the DV axis in the ML plane (deg)
    track_length_um  : float      # length of the trajectory inside the volume (micron)
    '''

    class Region(dj.Part):
        '''One row per contiguous atlas region the track traverses, ordered from
        the brain surface (``region_index = 0``) inward.  Join ``BrainRegion`` on
        ``region_id`` for ``name``/``rgb``.'''
        definition = '''
        -> ProbeTrack
        region_index    : int        # order along the track (0 = at the surface)
        ---
        region_id       : int        # atlas structure id (join BrainRegion)
        region_acronym  : varchar(32)
        entry_um        : float      # region entry depth from the surface (micron)
        exit_um         : float      # region exit depth from the surface (micron)
        length_um       : float
        '''

    @property
    def key_source(self):
        # only fit shank annotations
        return AtlasRegistrationAnnotation & 'annotation_type = "shank"'

    def _fit(self, key):
        '''Fit the trajectory and sample the annotation for a single key.

        Returns ``(fit, samples, regions, annotation, meta, lookup)``.
        '''
        import numpy as np
        from atlas_registration import (fit_track_line, trim_track_to_labeled,
                                        sample_annotation_along_track,
                                        regions_along_track)
        ann = (AtlasRegistrationAnnotation & key).fetch1()
        xyz = np.asarray(ann['xyz'], dtype=float)
        atlas = (AtlasRegistrationParams & key).fetch1('atlas')
        annotation = (AtlasRegistration & key).get_reference()
        meta, lookup = _atlas_annotation_and_lookup(atlas)
        resolution = np.asarray(meta['resolution'], dtype=float)
        fit = fit_track_line(xyz, volume_shape=annotation.shape,
                             orientation=meta['orientation'])
        # trim to the labeled brain so depth 0 is the brain surface
        track = trim_track_to_labeled(fit['track_voxels'], annotation)
        fit['track_voxels'] = track
        if len(track):
            fit['entry'], fit['exit'] = track[0].astype(float), track[-1].astype(float)
        samples = sample_annotation_along_track(
            track, annotation, lookup=lookup, resolution=resolution)
        regions = regions_along_track(samples, lookup=lookup)
        return fit, samples, regions, annotation, meta, lookup

    def make(self, key):
        fit, samples, regions, *_ = self._fit(key)
        atlas = (AtlasRegistrationParams & key).fetch1('atlas')
        BrainRegion.ensure(atlas)
        name = (AtlasRegistrationAnnotation & key).fetch1('annotation_name')
        probe_id, shank = name.rsplit('_shank', 1)
        self.insert1(dict(
            key, probe_id=probe_id, shank=int(shank),
            entry_voxel=fit['entry'],
            exit_voxel=fit['exit'],
            direction=fit['direction'],
            ap_angle=fit['angles']['ap'],
            ml_angle=fit['angles']['ml'],
            track_length_um=float(samples['depth_um'].iloc[-1]) if len(samples) else 0.))
        self.Region.insert(
            dict(key, region_index=int(i),
                 region_id=int(r['region_id']),
                 region_acronym=str(r.get('acronym', '')),
                 entry_um=float(r['entry_um']), exit_um=float(r['exit_um']),
                 length_um=float(r['length_um']))
            for i, r in regions.reset_index(drop=True).iterrows())

    def get_samples(self):
        '''Recompute the per-voxel ``sample_annotation_along_track`` DataFrame
        for a single selected track (needed to align channels).'''
        key = self.fetch1('KEY')
        _, samples, _, _, _, _ = self._fit(key)
        return samples

    def get_regions(self):
        '''Region read-out (``Region`` part joined to ``BrainRegion`` for name/
        rgb) as a DataFrame ordered from the surface inward, for a single track.'''
        import pandas as pd
        key = self.fetch1('KEY')
        df = pd.DataFrame((self.Region & key).fetch(as_dict=True))
        if not len(df):
            return df
        atlas = (AtlasRegistrationParams & key).fetch1('atlas')
        br = pd.DataFrame((BrainRegion & dict(atlas=atlas)).fetch(
            'region_id', 'acronym', 'name', 'rgb', as_dict=True))
        df = df.merge(br, on='region_id', how='left')
        return df.sort_values('region_index').reset_index(drop=True)

    def track_line(self):
        '''The fitted straight track as ``[entry_voxel, exit_voxel]`` (full-res
        atlas voxels), for a single shank.'''
        import numpy as np
        e, x = self.fetch1('entry_voxel', 'exit_voxel')
        return np.stack([np.asarray(e, float), np.asarray(x, float)])

    def plot_track_3d(self, fig=None, brain=True, color='red'):
        '''3D plot of this shank's track (optionally on the brain surface).'''
        from atlas_registration import plotting
        surface = (AtlasRegistration & self).surface() if (brain and fig is None) else None
        if fig is None:
            fig = plotting.figure_3d(surface=surface)
        plotting.add_line_3d(fig, self.track_line(), color=color,
                             name=(AtlasRegistrationAnnotation & self).fetch1('annotation_name'))
        return fig

    def plot_regions(self, ax=None):
        '''2D coloured region-vs-depth column for this shank (matplotlib).'''
        from atlas_registration import plotting
        df = self.get_regions()
        atlas = (AtlasRegistrationParams & self).fetch1('atlas')
        colors = plotting.region_colors(df['region_id'].to_numpy(),
                                        BrainRegion.rgb_map(atlas))
        ax = plotting.plot_region_column(df['entry_um'].to_numpy(),
                                         df['exit_um'].to_numpy(),
                                         acronyms=df['acronym'].to_numpy(),
                                         colors=colors, ax=ax)
        ax.set_title((AtlasRegistrationAnnotation & self).fetch1('annotation_name'),
                     fontsize=9)
        return ax

    def plot_track_slice(self, ax=None, channel=0, margin_um=200, step=1.0,
                         order=1, downsample=1, cmap='gray', track_color='red',
                         regions=False, regions_alpha=0.3):
        '''2D slice of the **registered stack** in the plane of the probe's shank
        track(s), with the tracks overlaid (0 = surface at top).  Restrict ``self``
        to one probe to see all its shanks in one image::

            (ProbeTrack & reg_key & 'probe_id="..."').plot_track_slice()

        ``downsample`` (>1) reads a coarser stack to save memory; ``channel`` picks
        the stack colour channel.  ``regions=True`` overlays the atlas regions
        (sampled on the same plane) at ``regions_alpha`` transparency.
        '''
        import numpy as np
        import matplotlib.pyplot as plt
        from atlas_registration import (probe_plane, oblique_slice,
                                        get_brainglobe_metadata, orientation_axes)
        keys, ev, xv, pid = self.fetch('KEY', 'entry_voxel', 'exit_voxel', 'probe_id')
        entries = np.array([np.asarray(e, float) for e in ev])
        exits = np.array([np.asarray(x, float) for x in xv])
        regkey = {k: v for k, v in keys[0].items() if k != 'annotation_id'}
        atlas = (AtlasRegistrationParams & regkey).fetch1('atlas')
        meta = get_brainglobe_metadata(atlas)
        axo = orientation_axes(meta['orientation'])
        res = float(np.mean(meta['resolution']))
        origin, u, v = probe_plane(entries, exits, dv_axis=axo['dv'])

        stack = np.squeeze((AtlasRegistration & regkey).get_stack())
        vol = stack[:, int(channel)] if stack.ndim == 4 else stack     # (AP, DV, ML)
        ds = int(downsample)
        if ds > 1:
            vol = np.ascontiguousarray(vol[::ds, ::ds, ::ds])
            entries, exits, origin, res = entries / ds, exits / ds, origin / ds, res * ds

        # sample the plane over the volume extent (project its 8 corners onto
        # u/v) so it spans the whole brain in this plane, then crop to the tissue.
        sh = np.asarray(vol.shape, float) - 1
        corners = np.array([[a, b, c] for a in (0, sh[0])
                            for b in (0, sh[1]) for c in (0, sh[2])])
        cu = (corners - origin) @ u
        cv = (corners - origin) @ v
        u_extent, v_extent = (cu.min(), cu.max()), (cv.min(), cv.max())
        img, us, vs = oblique_slice(vol, origin, u, v, u_extent, v_extent,
                                    step=step, order=order)
        # crop to the visible brain (nonzero stack) + a small margin
        r0, r1, c0, c1 = 0, img.shape[0], 0, img.shape[1]
        mask = img > img.max() * 0.05
        if mask.any():
            pad = int(round(margin_um / res))
            ri, ci = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
            r0, r1 = max(int(ri[0]) - pad, 0), min(int(ri[-1]) + pad + 1, img.shape[0])
            c0, c1 = max(int(ci[0]) - pad, 0), min(int(ci[-1]) + pad + 1, img.shape[1])
        img, us, vs = img[r0:r1, c0:c1], us[r0:r1], vs[c0:c1]
        extent = [vs[0] * res, vs[-1] * res, us[-1] * res, us[0] * res]

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 6))
        ax.imshow(img, cmap=cmap, aspect='equal', extent=extent)
        if regions:      # overlay the atlas regions sampled on the same plane
            ann = np.squeeze((AtlasRegistration & regkey).get_reference())
            if ds > 1:
                ann = ann[::ds, ::ds, ::ds]
            aimg = oblique_slice(ann, origin, u, v, u_extent, v_extent,
                                 step=step, order=0)[0][r0:r1, c0:c1].astype(int)
            rgb_map = BrainRegion.rgb_map(atlas)
            ids = np.unique(aimg)
            palette = np.array([rgb_map.get(int(i), [0, 0, 0]) for i in ids], float) / 255.0
            ax.imshow(palette[np.searchsorted(ids, aimg)], aspect='equal', extent=extent,
                      alpha=regions_alpha)# * (aimg != 0))
        for e, x in zip(entries, exits):     # overlay each shank track (on top)
            ax.plot([((e - origin) @ v) * res, ((x - origin) @ v) * res],
                    [((e - origin) @ u) * res, ((x - origin) @ u) * res],
                    color=track_color, lw=1.5)
        ax.set_xlim(vs[0] * res, vs[-1] * res)          # keep the crop (don't let the
        ax.set_ylim(us[-1] * res, us[0] * res)          # track line re-expand the axes)
        ax.set_xlabel('across shanks (µm)')
        ax.set_ylabel('depth along track (µm)')
        ax.set_title(', '.join(sorted(set(pid))), fontsize=9)
        return ax


@atlas_schema
class ProbeAlignment(dj.Manual):
    '''
    Alignment of electrode depths to a ``ProbeTrack`` using reference-depth
    pairs, and the resulting per-channel region assignment.

    ``feature_ref``/``track_ref`` are the matched reference depths the user
    marked (electrophysiology axis vs. histology track); ``channel_locations``
    is the per-channel depth -> region table computed by
    ``atlas_registration.align_channels_to_regions``.  With no reference pairs
    the alignment is the raw histology mapping.
    '''
    definition = '''
    -> ProbeTrack
    alignment_id = 0        : smallint
    ---
    feature_ref = NULL      : blob       # electrode reference depths from the tip (micron)
    track_ref = NULL        : blob       # matching atlas depths from the surface (micron)
    insertion_depth = NULL  : float      # tip depth from the surface (micron); ties the two frames
    extrapolate = "segment" : varchar(16)  # (legacy) extrapolation mode
    alignment_user = NULL   : varchar(64)
    confidence = NULL       : varchar(24)
    alignment_ts = CURRENT_TIMESTAMP : timestamp
    '''

    class Channel(dj.Part):
        '''One row per recording channel of the aligned shank.  ``channel_index``
        matches ``UnitMetrics.channel_index`` (row into the ProbeConfiguration
        channel arrays), so joining gives each unit its atlas region, e.g.::

            UnitMetrics * ProbeAlignment.Channel & alignment_key
        '''
        definition = '''
        -> ProbeAlignment
        channel_index        : int       # index into the ProbeConfiguration channels
        ---
        electrode_depth_um   : float     # channel depth from the probe tip
        track_depth_um       : float     # aligned atlas depth from the surface
        voxel_x              : float
        voxel_y              : float
        voxel_z              : float
        region_id            : int       # atlas structure id (join BrainRegion for name)
        region_acronym       : varchar(32)
        '''

    def _shank_channels(self, key):
        '''``(channel_index, depth)`` of the annotated shank's channels, sorted
        by depth.  ``channel_index`` is the row into the ProbeConfiguration
        channel arrays (matches ``UnitMetrics.channel_index``).'''
        import numpy as np
        annotation_name = (AtlasRegistrationAnnotation & key).fetch1('annotation_name')
        probe_id, shank = annotation_name.rsplit('_shank', 1)
        shank = int(shank)
        conf = (ProbeConfiguration
                & (EphysRecording.ProbeSetting() & (Subject & key))
                & dict(probe_id=probe_id)).fetch(as_dict=True)
        if not conf:
            raise ValueError(f'No ProbeConfiguration for probe {probe_id}.')
        conf = conf[0]
        coords = np.asarray(conf['channel_coords'], dtype=float)
        shanks = np.asarray(conf['channel_shank']).ravel()
        idx = np.where(shanks == shank)[0]          # global channel indices
        sel = coords[idx]
        # depth axis: the column with the larger span along the shank
        depth_axis = int(np.argmax(sel.max(0) - sel.min(0)))
        depths = sel[:, depth_axis]
        order = np.argsort(depths)                   # sort along the shank
        return idx[order], depths[order]

    def channel_depths(self, key=None):
        '''Depth (micron) of each channel of the annotated shank, sorted along
        the shank (see :meth:`_shank_channels`).'''
        if key is None:
            key = self.fetch1('KEY')
        return self._shank_channels(key)[1]

    def align(self, key, feature_ref=None, track_ref=None, alignment_id=0,
              insertion_depth=None, channel_depths=None, channel_index=None,
              replace=False, **provenance):
        '''
        Compute and insert an alignment for a ``ProbeTrack`` key.

        Parameters
        ----------
        key : dict
            A ``ProbeTrack`` key (single shank annotation).
        feature_ref : array-like, optional
            Electrode reference depths measured from the probe tip (micron).
        track_ref : array-like, optional
            The matching atlas depths from the brain surface (micron).
        alignment_id : int
        insertion_depth : float
            Tip depth from the surface (micron).  With no reference pairs the
            mapping is the pure inversion ``atlas = insertion_depth - electrode``.
        channel_depths : array-like, optional
            Electrode depths from the tip; defaults to the probe configuration.
        channel_index : array-like, optional
            The channel index of each ``channel_depths`` entry (row into the
            ProbeConfiguration channel arrays; matches ``UnitMetrics.channel_index``).
            Pass it alongside ``channel_depths`` to avoid re-deriving the channels
            from the config.  Defaults to the probe configuration.
        provenance : keys
            e.g. ``alignment_user``, ``confidence``.
        '''
        import numpy as np
        from atlas_registration import align_channels_to_regions
        track = (ProbeTrack & key)
        tkey = track.fetch1('KEY')
        _, samples, _, annotation, meta, lookup = track._fit(tkey)
        # channel indices + depths of the shank, sorted along the probe.  The
        # index labels each row so the Channel part joins to UnitMetrics.  Only
        # re-derive from the config if the caller didn't supply them.
        if channel_index is None or channel_depths is None:
            ch_index, ch_depths = self._shank_channels(tkey)
            if channel_depths is None:
                channel_depths = ch_depths      # raw depths from the tip (0 = tip)
            if channel_index is None:
                channel_index = ch_index
        channel_depths = np.asarray(channel_depths, dtype=float)
        channel_index = np.asarray(channel_index).astype(int).ravel()
        if len(channel_depths) != len(channel_index):
            raise ValueError(
                f'channel_depths ({len(channel_depths)}) does not match '
                f'channel_index ({len(channel_index)}).')
        if insertion_depth is None:
            insertion_depth = float(np.max(channel_depths))
        chan = align_channels_to_regions(
            channel_depths, samples,
            feature_ref=feature_ref, track_ref=track_ref,
            annotation=annotation, lookup=lookup,
            resolution=np.asarray(meta['resolution'], dtype=float),
            insertion_depth=float(insertion_depth))

        BrainRegion.ensure((AtlasRegistrationParams & tkey).fetch1('atlas'))
        akey = dict(tkey, alignment_id=int(alignment_id))
        master = dict(akey,
                      feature_ref=None if feature_ref is None else np.asarray(feature_ref, float),
                      track_ref=None if track_ref is None else np.asarray(track_ref, float),
                      insertion_depth=float(insertion_depth), **provenance)
        parts = [dict(akey, channel_index=int(ci),
                      electrode_depth_um=float(r['depth_um']),
                      track_depth_um=float(r['track_depth_um']),
                      voxel_x=float(r['x']), voxel_y=float(r['y']), voxel_z=float(r['z']),
                      region_id=int(r['region_id']), region_acronym=str(r['acronym']))
                 for ci, (_, r) in zip(channel_index, chan.iterrows())]
        if replace:
            (self.Channel & akey).delete_quick()
            (self & akey).delete_quick()
        with self.connection.transaction:
            self.insert1(master)
            self.Channel.insert(parts)
        return chan

    def get_channels(self):
        '''Per-channel region assignment (from the ``Channel`` part) as a
        DataFrame ordered by depth, for a single alignment.'''
        import pandas as pd
        df = pd.DataFrame((self.Channel & self.fetch1('KEY')).fetch(as_dict=True))
        if len(df):
            df = df.sort_values('electrode_depth_um').reset_index(drop=True)
        return df

    def plot_channels_3d(self, fig=None, brain=True, size=4):
        '''3D plot of this shank's channels coloured by atlas region.'''
        import numpy as np
        from atlas_registration import plotting
        key = self.fetch1('KEY')
        atlas = (AtlasRegistrationParams & key).fetch1('atlas')
        vx, vy, vz, rid, acr = (self.Channel & key).fetch(
            'voxel_x', 'voxel_y', 'voxel_z', 'region_id', 'region_acronym')
        xyz = np.stack([vx, vy, vz], axis=1).astype(float)
        colors = plotting.region_colors([int(r) for r in rid], BrainRegion.rgb_map(atlas))
        if fig is None:
            surf = (AtlasRegistration & key).surface() if brain else None
            fig = plotting.figure_3d(surface=surf)
        plotting.add_points_3d(fig, xyz, colors=colors, size=size, text=acr,
                               name='channels')
        return fig


@atlas_schema
class UnitAlignment(dj.Computed):
    '''Per-unit atlas region for a spike sorting, from a ``ProbeAlignment``.

    Each unit is assigned by its physical depth along the shank (micron from the
    tip, because ``channel_index`` is probe configuration dependent) evaluated
    through the alignment's electrode->atlas mapping.  Keyed by the ephys
    ``SpikeSorting``, so it is computed *per session* (within-session drift is
    already baked into the sorted unit positions; ``drift_offset_um`` is a hook
    for cross-session drift correction).  The histology session/dataset of the
    ``ProbeAlignment`` are renamed (``areg_session``/``areg_dataset``) so they
    don't collide with the ephys ones.

    Join to units cleanly on the full ephys key + ``unit_id``::

        UnitMetrics * UnitAlignment.Unit & align_key
    '''
    definition = '''
    -> SpikeSorting
    -> ProbeAlignment.proj(areg_session='session_name', areg_dataset='dataset_name')
    ---
    drift_offset_um = 0 : float   # depth offset applied to the unit depths (drift hook)
    '''

    class Unit(dj.Part):
        '''One row per unit of the aligned shank.  ``-> UnitMetrics`` so it joins
        the ephys unit table directly; join ``BrainRegion`` for ``name``.'''
        definition = '''
        -> UnitAlignment
        -> UnitMetrics
        ---
        electrode_depth_um : float   # unit depth from the tip (+ drift_offset)
        track_depth_um     : float   # aligned atlas depth from the surface
        voxel_x            : float
        voxel_y            : float
        voxel_z            : float
        region_id          : int     # atlas structure id (join BrainRegion for name)
        region_acronym     : varchar(32)
        '''

        # ---- plotting operates on a *query* of units (restrict to select) ----
        def _registration_key(self):
            k = self.fetch('KEY', limit=1)[0]
            return dict(subject_name=k['subject_name'], session_name=k['areg_session'],
                        dataset_name=k['areg_dataset'], transform_id=k['transform_id'],
                        atlas_reg_id=k['atlas_reg_id'])

        def get_units(self):
            '''The queried units joined to their metrics and region metadata:
            ``voxel_x/y/z``, depths, ``region_id``/``region_acronym``, ``name``,
            ``rgb`` and every ``UnitMetrics`` column.'''
            import pandas as pd
            rkey = self._registration_key()
            atlas = (AtlasRegistrationParams & rkey).fetch1('atlas')
            df = pd.DataFrame((UnitMetrics * self).fetch(as_dict=True))
            if len(df):
                br = pd.DataFrame((BrainRegion & dict(atlas=atlas)).fetch(
                    'region_id', 'name', 'rgb', as_dict=True))
                df = df.merge(br, on='region_id', how='left')
            df.attrs.update(atlas=atlas, rkey=rkey)
            return df

        def _shank_lines(self):
            '''One ``[entry, tip]`` line per shank present in the query, truncated
            to the alignment's insertion depth when available (else the full track).'''
            import numpy as np
            rkey = self._registration_key()
            pairs = sorted({(int(r['annotation_id']), int(r['alignment_id']))
                            for r in self.fetch('annotation_id', 'alignment_id',
                                                as_dict=True)})
            lines = []
            for aid, alid in pairs:
                tk = ProbeTrack & dict(rkey, annotation_id=aid)
                if not len(tk):
                    continue
                e, x, length = tk.fetch1('entry_voxel', 'exit_voxel', 'track_length_um')
                e, x = np.asarray(e, float), np.asarray(x, float)
                ins = (ProbeAlignment & dict(rkey, annotation_id=aid, alignment_id=alid)
                       ).fetch1('insertion_depth')
                tip = (e + min(float(ins) / (float(length) or 1.0), 1.0) * (x - e)
                       if ins is not None else x)
                lines.append(np.stack([e, tip]))
            return lines

        def plot_units_3d(self, color_by='region', brain=True, track_fit=True,
                          fig=None, size=3, cmap='Viridis', clim=None,
                          track_color=None, track_alpha=1.0, track_linewidth=4):
            '''3D plot of the **queried** units on the brain surface, coloured by
            region (default) or a ``UnitMetrics`` metric.  ``track_fit`` toggles
            the fitted probe track (truncated to the insertion depth); style it with
            ``track_color`` (default: per-shank palette), ``track_alpha`` and
            ``track_linewidth``.  ``cmap``/``clim`` set the colormap and range for a
            metric.  Pass ``fig`` to overlay.'''
            from atlas_registration import plotting
            df = self.get_units()
            atlas, rkey = df.attrs['atlas'], df.attrs['rkey']
            xyz = df[['voxel_x', 'voxel_y', 'voxel_z']].to_numpy(float)
            if fig is None:
                surf = (AtlasRegistration & rkey).surface() if brain else None
                fig = plotting.figure_3d(surface=surf)
            if track_fit:
                for i, line in enumerate(self._shank_lines()):
                    plotting.add_line_3d(
                        fig, line, width=track_linewidth, opacity=track_alpha,
                        color=track_color or plotting._TRACK_COLORS[i % len(plotting._TRACK_COLORS)])
            if color_by == 'region':
                colors = plotting.region_colors(df['region_id'].to_numpy(),
                                                BrainRegion.rgb_map(atlas))
                plotting.add_points_3d(fig, xyz, colors=colors, size=size,
                                       text=df['region_acronym'], name='units')
            else:
                plotting.add_points_3d(fig, xyz, values=df[color_by].to_numpy(float),
                                       cmap=cmap, clim=clim, size=size,
                                       colorbar_title=color_by,
                                       text=df['region_acronym'], name='units')
            return fig

        def plot_units_depth(self, color_by='region', ax=None):
            '''2D scatter of the queried units by atlas depth (region or metric).'''
            from atlas_registration import plotting
            df = self.get_units()
            depth = df['track_depth_um'].to_numpy(float)
            if color_by == 'region':
                colors = plotting.region_colors(df['region_id'].to_numpy(),
                                                BrainRegion.rgb_map(df.attrs['atlas']))
                return plotting.plot_units_depth(depth, colors=colors, ax=ax)
            return plotting.plot_units_depth(depth, values=df[color_by].to_numpy(float),
                                             ax=ax, colorbar_title=color_by)

        def plot_region_counts(self, ax=None):
            '''2D bar chart of the number of queried units per region.'''
            from atlas_registration import plotting
            df = self.get_units()
            g = (df[df['region_acronym'] != ''].groupby('region_acronym')
                 .agg(n=('unit_id', 'size'), region_id=('region_id', 'first'))
                 .reset_index())
            colors = plotting.region_colors(g['region_id'].to_numpy(),
                                            BrainRegion.rgb_map(df.attrs['atlas']))
            return plotting.plot_region_bar(g['region_acronym'].to_numpy(),
                                            g['n'].to_numpy(), colors=colors, ax=ax)

    @property
    def key_source(self):
        # pair each sorting with each ProbeAlignment of the SAME subject + probe
        # (the histology session is renamed so only subject_name + probe_id match).
        # dj.U projects the matched pairs down to exactly this table's key,
        # dropping probe_id / configuration_id brought in by the ProbeConfiguration.
        sortings = (SpikeSorting * EphysRecording.ProbeSetting
                    * ProbeConfiguration).proj('probe_id')
        aligns = (ProbeAlignment * ProbeTrack).proj(
            'probe_id', 'shank',
            areg_session='session_name', areg_dataset='dataset_name')
        return dj.U(*self.primary_key) & (sortings * aligns)

    def make(self, key):
        import numpy as np
        import pandas as pd
        from atlas_registration import (electrode_to_atlas, get_brainglobe_metadata,
                                        orientation_axes)
        # rebuild the two parent keys from the (renamed) combined key
        akey = dict(subject_name=key['subject_name'],
                    session_name=key['areg_session'],
                    dataset_name=key['areg_dataset'],
                    transform_id=key['transform_id'],
                    atlas_reg_id=key['atlas_reg_id'],
                    annotation_id=key['annotation_id'],
                    alignment_id=key['alignment_id'])
        skey = dict(subject_name=key['subject_name'],
                    session_name=key['session_name'],
                    dataset_name=key['dataset_name'],
                    probe_num=key['probe_num'],
                    parameter_set_num=key['parameter_set_num'])
        tkey = {k: v for k, v in akey.items() if k != 'alignment_id'}
        mkey = {k: key[k] for k in self.primary_key}
        atlas = (AtlasRegistrationParams & tkey).fetch1('atlas')
        ax = orientation_axes(get_brainglobe_metadata(atlas)['orientation'])

        # everything is read from stored tables — NO atlas annotation load.
        trow = (ProbeTrack & tkey).fetch1()
        entry = np.asarray(trow['entry_voxel'], float)
        exit_ = np.asarray(trow['exit_voxel'], float)
        length = float(trow['track_length_um']) or 1.0
        shank = int(trow['shank'])
        reg = pd.DataFrame((ProbeTrack.Region & tkey).fetch(
            'entry_um', 'exit_um', 'region_id', 'region_acronym', as_dict=True))
        al = (ProbeAlignment & akey).fetch1()
        drift = 0.0

        # units on this shank (fall back to nearest lateral shank centre if null)
        um = (UnitMetrics & skey).fetch('unit_id', 'shank', 'position', as_dict=True)
        conf = (ProbeConfiguration & (EphysRecording.ProbeSetting & skey)).fetch(
            as_dict=True)[0]
        coords = np.asarray(conf['channel_coords'], float)
        cshank = np.asarray(conf['channel_shank']).ravel()
        # depth axis = the larger *within-shank* span (NOT the global span: on a
        # multi-shank probe the across-shank lateral extent can exceed the depth).
        ws = np.array([coords[cshank == s].max(0) - coords[cshank == s].min(0)
                       for s in np.unique(cshank) if int((cshank == s).sum()) > 1])
        depth_axis = int(np.argmax(np.median(ws, 0) if len(ws)
                                   else coords.max(0) - coords.min(0)))
        lat_axis = 1 - depth_axis
        centres = {int(s): float(np.median(coords[cshank == s, lat_axis]))
                   for s in np.unique(cshank)}
        uid, udepth, ulateral = [], [], []
        for u in um:
            if u['position'] is None:
                continue
            pos = np.asarray(u['position'], float).ravel()
            if pos.size <= max(depth_axis, lat_axis) or not np.isfinite(pos).all():
                continue          # malformed / missing position — can't place it
            ush = u['shank']
            if ush is None:
                ush = min(centres, key=lambda s: abs(pos[lat_axis] - centres[s]))
            if int(ush) != shank:
                continue
            uid.append(int(u['unit_id']))
            udepth.append(float(pos[depth_axis]))
            ulateral.append(float(pos[lat_axis] - centres[shank]))   # off shank centre

        self.insert1(dict(mkey, drift_offset_um=float(drift)))
        if not uid:
            return
        udepth = np.asarray(udepth) + drift
        # electrode depth -> atlas depth from the surface
        track_depth = electrode_to_atlas(udepth, float(al['insertion_depth']),
                                         al['feature_ref'], al['track_ref'])
        # voxel on the shank centreline: linear interp along the track (entry -> exit)
        frac = np.clip(track_depth / length, 0.0, 1.0)[:, None]
        vox = entry[None, :] + frac * (exit_ - entry)[None, :]
        # lateral shift: offset each unit off the centreline by its distance from the
        # shank centre, perpendicular to the track in the horizontal plane
        # (cross(track_dir, DV)); fall back to the ML axis for a near-vertical track.
        direction = exit_ - entry
        voxel_size = length / (np.linalg.norm(direction) or 1.0)       # µm per voxel
        d = direction / (np.linalg.norm(direction) or 1.0)
        dv = np.zeros(3); dv[ax['dv']] = 1.0
        lat_dir = np.cross(d, dv)
        n = np.linalg.norm(lat_dir)
        lat_dir = lat_dir / n if n > 1e-6 else np.eye(3)[ax['ml']]
        vox = vox + lat_dir[None, :] * (np.asarray(ulateral)[:, None] / voxel_size)
        # region: the Region interval (in atlas depth) that contains each unit
        ent = reg['entry_um'].to_numpy() if len(reg) else np.array([])
        ex = reg['exit_um'].to_numpy() if len(reg) else np.array([])

        def _region_at(d):
            if not len(reg):
                return 0, ''
            hit = np.where((ent <= d) & (ex > d))[0]
            k = int(hit[0]) if len(hit) else (len(reg) - 1 if d >= ex[-1] else 0)
            return int(reg['region_id'].iloc[k]), str(reg['region_acronym'].iloc[k])

        rows = []
        for j in range(len(uid)):
            rid, acr = _region_at(track_depth[j])
            rows.append(dict(mkey, unit_id=uid[j], electrode_depth_um=float(udepth[j]),
                             track_depth_um=float(track_depth[j]),
                             voxel_x=float(vox[j, 0]), voxel_y=float(vox[j, 1]),
                             voxel_z=float(vox[j, 2]), region_id=rid, region_acronym=acr))
        self.Unit.insert(rows)

    def _registration_key(self):
        '''The AtlasRegistration key (undo the areg_* rename) for a single row.'''
        k = self.fetch1('KEY')
        return dict(subject_name=k['subject_name'], session_name=k['areg_session'],
                    dataset_name=k['areg_dataset'], transform_id=k['transform_id'],
                    atlas_reg_id=k['atlas_reg_id'])

    # convenience: operate on all of this alignment's units.  For a subset,
    # restrict the part directly, e.g. (UnitAlignment.Unit & al & 'region_acronym="CA1"')
    def get_units(self):
        '''This alignment's units + metrics + region metadata (delegates to Unit).'''
        return (self.Unit & self).get_units()

    def plot_units_3d(self, **kwargs):
        '''3D plot of this alignment's units (delegates to Unit.plot_units_3d).'''
        return (self.Unit & self).plot_units_3d(**kwargs)

    def plot_units_depth(self, **kwargs):
        return (self.Unit & self).plot_units_depth(**kwargs)

    def plot_region_counts(self, **kwargs):
        return (self.Unit & self).plot_region_counts(**kwargs)


# Quick-reference used by the dashboard's schema tab.
SCHEMA_REFERENCE = {
    'AtlasRegistrationParams': (
        'Manual — elastix registration parameters for a FixedBrainTransform; '
        'one row per `atlas_reg_id` (atlas, geometry, resolutions, grid spacing).',
        {}, {}),
    'AtlasRegistration': (
        'Computed — runs elastix to register the brain to the atlas and stores '
        'the transforms + registered stack. Run with `.populate()`.',
        {'get_stack()': 'load the registered stack from the AnalysisFile',
         'get_reference()': 'the atlas annotation volume for this row',
         'napari_open(color=False)': 'open the registered stack in napari',
         'annotate_probe_tracks_napari()': 'click shank tracks in napari',
         'save_probe_tracks_napari(update=False)': 'save napari points to AtlasRegistrationAnnotation'},
        {}),
    'AtlasRegistrationAnnotation': (
        'Manual — hand-marked points (`xyz`) along a shank track in atlas-volume '
        'voxels; `annotation_name` = "<probe_id>_shank<i>".',
        {}, {}),
    'BrainRegion': (
        'Manual lookup — atlas region metadata keyed by (atlas, region_id); the '
        'Region/Channel/Unit parts store region_id (+ acronym), join here for name/rgb.',
        {'ensure(atlas)': 'populate all regions of an atlas (idempotent)'}, {}),
    'ProbeTrack': (
        'Computed — straight trajectory fitted to a shank annotation and the '
        'atlas regions traversed (probe_id, shank, entry/exit voxel, angles, length).',
        {'get_samples()': 'per-voxel depth->region DataFrame (single row)',
         'get_regions()': 'region read-out (Region joined to BrainRegion) as a DataFrame'},
        {'Region': 'one row per contiguous region traversed (region_index 0 = '
                   'surface): region_id, region_acronym, entry_um, exit_um, length_um'}),
    'ProbeAlignment': (
        'Manual — reference-depth alignment of electrode depths to a ProbeTrack; '
        'the per-channel region assignment lives in the Channel part.',
        {'align(key, feature_ref, track_ref, alignment_id=0, ...)':
            'compute + insert an alignment (master + Channel part)',
         'channel_depths(key=None)': 'shank channel depths from ProbeConfiguration',
         'get_channels()': 'per-channel regions (from Channel) as a DataFrame'},
        {'Channel': 'one row per channel keyed by channel_index (histology-config '
                    'specific): electrode_depth_um, track_depth_um, voxel_x/y/z, '
                    'region_id, region_acronym'}),
    'UnitAlignment': (
        'Computed — per-unit atlas region for a SpikeSorting, by unit depth '
        '(config-independent) through a ProbeAlignment; per session. Join units: '
        '`UnitMetrics * UnitAlignment.Unit & align_key`.',
        {}, {'Unit': 'one row per unit (-> UnitMetrics): electrode_depth_um, '
                     'track_depth_um, voxel_x/y/z, region_id, region_acronym'}),
}
