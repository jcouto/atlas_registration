from labdata.schema import *

username = prefs['database']['database.user']
atlas_schema = f'{username}_atlas_registration'
if 'atlas_registration_schema' in prefs.keys(): # to be able to override to another name
    atlas_schema = prefs['atlas_registration_schema']
if 'root' in atlas_schema:
    raise(ValueError('[atlas_registration] "atlas_registration_schema" must be specified in the preference file to run as root.'))

atlas_schema = dj.schema(atlas_schema)

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
                         /key['subject_name']))
                       /key['session_name'])/f'brain_transform_{key["transform_id"]}'
        filepath = folder_path/f'atlas_reg_{key["atlas_reg_id"]}.ome.tif'
        folder_path.mkdir(exist_ok=True)
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

    def napari_open(self,color = False, **kwargs):
        if color:
            kwargs['channel_axis'] = 1
        stack = self.get_stack()
        from labdata.stacks import napari_open
        napari_open(stack,**kwargs)

    def annotate_probe_tracks(self):
        ''' Annotate probe tracks for electrophysiology.'''
        if len(self) == 0:
            raise(ValueError('No brain to annotate.'))
        if len(self) > 1:
            raise(ValueError('Select only one brain.'))
        key = self.proj().fetch1()
        stack = self.get_stack()
        unique_probes = (Probe() & 
                         (EphysRecording.ProbeSetting() &
                          (Subject & key))).fetch(as_dict = True)
        shank_names = []
        for p in unique_probes:
            for i in range(p['probe_n_shanks']):
                shank_names.append(f"{p['probe_id']}_shank{i}")
        self.shank_names = shank_names
        self.shank_layers = []
        import pylab as plt
        colormap = plt.colormaps['tab10']
        colors = [plt.matplotlib.colors.to_hex(c) for c in colormap(range(10))]
        import napari

        viewer = napari.Viewer()
        im = viewer.add_image(stack,channel_axis = 1)
        for i,shank in enumerate(self.shank_names):
            dd = (AtlasRegistrationAnnotation() & self & dict(annotation_name = shank)).fetch(as_dict = True)
            par = dict(name = shank,
                       ndim=3,
                       size=5,
                       opacity=1,
                       face_color=colors[np.mod(i,len(colors))])
            if len(dd) > 0:
                par['data'] = dd[0]['xyz']
            self.shank_layers.append(viewer.add_points(**par))
        viewer.show()
        return self
    
    def save_probe_tracks(self):
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
        AtlasRegistrationAnnotation.insert(toadd)

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

    