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
           'ProbeAlignment']


def _atlas_annotation_and_lookup(atlas):
    '''Load the atlas annotation volume, metadata and structure lookup once.'''
    from atlas_registration import (get_brainglobe_annotation,
                                    get_brainglobe_metadata,
                                    get_structure_lookup)
    meta = get_brainglobe_metadata(atlas)
    lookup = get_structure_lookup(atlas)
    return meta, lookup

@atlas_schema
class AtlasRegistrationParams(dj.Manual):
    definition = '''
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
    
    def save_probe_tracks_napari(self,update = False):
        ''' Save the probe tracks in AtlasRegistrationAnnotation'''
        key = (self).proj().fetch1()
        toadd = []
        for i,(name,layer) in enumerate(zip(self.shank_names,self.shank_layers)):
            points = layer.data
            toadd.append(dict(key,annotation_id = i,
                              annotation_name = name,
                              annotation_type = 'shank',
                              xyz = points))
        print(f"Inserting {len(toadd)} shank annotations.")
        # TODO: Ask if the user wants to update if they are already there.
        AtlasRegistrationAnnotation.insert(toadd, replace=update)

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
    entry_voxel      : blob       # first (most dorsal) voxel inside the volume
    exit_voxel       : blob       # last (most ventral) voxel inside the volume
    direction        : blob       # unit direction vector of the trajectory
    ap_angle         : float      # insertion angle from the DV axis in the AP plane (deg)
    ml_angle         : float      # insertion angle from the DV axis in the ML plane (deg)
    track_length_um  : float      # length of the trajectory inside the volume (micron)
    regions          : longblob   # regions_along_track() records (list of dicts)
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
        self.insert1(dict(
            key,
            entry_voxel=fit['entry'],
            exit_voxel=fit['exit'],
            direction=fit['direction'],
            ap_angle=fit['angles']['ap'],
            ml_angle=fit['angles']['ml'],
            track_length_um=float(samples['depth_um'].iloc[-1]) if len(samples) else 0.,
            regions=regions.to_dict('records')))

    def get_samples(self):
        '''Recompute the per-voxel ``sample_annotation_along_track`` DataFrame
        for a single selected track (needed to align channels).'''
        key = self.fetch1('KEY')
        _, samples, _, _, _, _ = self._fit(key)
        return samples

    def regions_dataframe(self):
        '''Return the cached region read-out as a DataFrame for a single row.'''
        import pandas as pd
        return pd.DataFrame(self.fetch1('regions'))


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
    channel_locations       : longblob   # per-channel align_channels_to_regions() records
    alignment_user = NULL   : varchar(64)
    confidence = NULL       : varchar(24)
    alignment_ts = CURRENT_TIMESTAMP : timestamp
    '''

    def channel_depths(self, key=None):
        '''Depth (micron) of each channel of the annotated shank, from the
        probe configuration ``channel_coords`` (the depth axis is the larger-
        range coordinate).  Returns depths sorted along the shank.'''
        import numpy as np
        if key is None:
            key = self.fetch1('KEY')
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
        sel = coords[shanks == shank]
        # depth axis: the column with the larger span along the shank
        depth_axis = int(np.argmax(sel.max(0) - sel.min(0)))
        return np.sort(sel[:, depth_axis])

    def align(self, key, feature_ref=None, track_ref=None, alignment_id=0,
              insertion_depth=None, channel_depths=None, replace=False,
              **provenance):
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
        provenance : keys
            e.g. ``alignment_user``, ``confidence``.
        '''
        import numpy as np
        from atlas_registration import align_channels_to_regions
        track = (ProbeTrack & key)
        _, samples, _, annotation, meta, lookup = track._fit(track.fetch1('KEY'))
        if channel_depths is None:
            # raw channel depths (the probe measures them from the tip, 0 = tip)
            channel_depths = self.channel_depths(track.fetch1('KEY'))
        if insertion_depth is None:
            insertion_depth = float(np.max(channel_depths))
        chan = align_channels_to_regions(
            channel_depths, samples,
            feature_ref=feature_ref, track_ref=track_ref,
            annotation=annotation, lookup=lookup,
            resolution=np.asarray(meta['resolution'], dtype=float),
            insertion_depth=float(insertion_depth))
        row = dict((ProbeTrack & key).fetch1('KEY'),
                   alignment_id=int(alignment_id),
                   feature_ref=None if feature_ref is None else np.asarray(feature_ref, float),
                   track_ref=None if track_ref is None else np.asarray(track_ref, float),
                   insertion_depth=float(insertion_depth),
                   channel_locations=chan.to_dict('records'),
                   **provenance)
        self.insert1(row, replace=replace)
        return chan

    def channel_locations_dataframe(self):
        '''Return the stored per-channel region table as a DataFrame.'''
        import pandas as pd
        return pd.DataFrame(self.fetch1('channel_locations'))


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
    'ProbeTrack': (
        'Computed — straight trajectory fitted to a shank annotation and the '
        'atlas regions traversed (entry/exit, angles, length, region read-out).',
        {'get_samples()': 'per-voxel depth->region DataFrame (single row)',
         'regions_dataframe()': 'cached region read-out as a DataFrame'},
        {}),
    'ProbeAlignment': (
        'Manual — reference-depth alignment of electrode depths to a ProbeTrack '
        'and the per-channel region assignment (`channel_locations`).',
        {'align(key, feature_ref, track_ref, alignment_id=0, ...)':
            'compute + insert an alignment',
         'channel_depths(key=None)': 'shank channel depths from ProbeConfiguration',
         'channel_locations_dataframe()': 'stored per-channel regions as a DataFrame'},
        {}),
}
