
from tifffile import imread
from pathlib import Path
import pandas as pd

def get_brainglobe_annotation(atlas, brain_geometry ='both'):
    atlas_folder = list((Path.home()/'.brainglobe').glob(atlas+'*'))
    if not len(atlas_folder):
        raise(ValueError(f'Could not find {atlas} in the .brainglobe folder'))
    mat = imread(atlas_folder[0]/'annotation.tiff')
    if brain_geometry == 'left':
        mat  = mat[:,:,:mat.shape[2]//2]
    elif brain_geometry == 'right':
        mat  = mat[:,:,mat.shape[2]//2:]
    return mat

def get_brainglobe_atlas(atlas,brain_geometry ='both'):
    atlas_folder = list((Path.home()/'.brainglobe').glob(atlas+'*'))
    if not len(atlas_folder):
        raise(ValueError(f'Could not find {atlas} in the .brainglobe folder'))
    mat = imread(atlas_folder[0]/'reference.tiff')
    if brain_geometry == 'left':
        mat  = mat[:,:,:mat.shape[2]//2]
    elif brain_geometry == 'right':
        mat  = mat[:,:,mat.shape[2]//2:]
    return mat

def get_brainglobe_structure_data(atlas):
    atlas_folder = list((Path.home()/'.brainglobe').glob(atlas+'*'))
    if not len(atlas_folder):
        raise(ValueError(f'Could not find {atlas} in the .brainglobe folder'))
    import json
    with open(atlas_folder[0]/'structures.json','r') as fd:
        structures = json.load(fd)    
    return pd.DataFrame(structures)