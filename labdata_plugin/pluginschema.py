from labdata.schema import *

username = prefs['database']['database.user']
atlas_schema = f'{username}_atlas_registration'
if 'atlas_registration_schema' in prefs.keys(): # to be able to override to another name
    atlas_schema = prefs['atlas_registration_schema']

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
    number_of_resolutions = 6              : int  
    number_of_resolutions_second = 8       : int
    final_grid_spacing = 25.0              : float
    number_of_histogram_bins = 32          : int 
    maximum_number_of_interactions = 2500  : int
    number_of_spatial_samples = 5000       : int
    stack_gaussian_smoothing = NULL        : int
    '''

@atlas_schema
class AtlasRegistration(dj.Computed):
    definition = '''
    -> AtlasRegistrationParams
    ---
    elastix_transforms                    : longblob
    -> [nullable] AnalysisFile
    '''
    def register(self,key):
        par = (BrainRegParams() & key).fetch1()
        stack = FixedBrainTransform().transform(key)
        from ..atlas_registration import exastix_register_brain
        registered, transforms = elastix_register_brain(stack[:,0],
                                                        atlas='kim_dev_mouse_idisco_10um',
                                                        brain_geometry = key['brain_geometry'],
                                                        number_of_resolutions = key['number_of_resolutions'],
                                                        number_of_resolutions_second = key['number_of_resolutions_second'],
                                                        final_grid_spacing = key['final_grid_spacing'],
                                                        number_of_histogram_bins = key['number_of_histogram_bins'],
                                                        maximum_number_of_interactions = key['maximum_number_of_interactions'],
                                                        number_of_spatial_samples = key['number_of_spatial_samples'],
                                                        stack_gaussian_smoothing = key['stack_gaussian_smoothing'])
        