
from tifffile import imread
from pathlib import Path
import pandas as pd


def _find_brainglobe_folder(atlas):
    '''Return the path to the brainglobe atlas folder that starts with ``atlas``.'''
    atlas_folder = list((Path.home()/'.brainglobe').glob(atlas+'*'))
    if not len(atlas_folder):
        raise(ValueError(f'Could not find {atlas} in the .brainglobe folder'))
    return atlas_folder[0]


def get_brainglobe_metadata(atlas):
    '''
    Read the brainglobe ``metadata.json`` for an atlas.

    Returns a dict with keys like ``resolution`` (list of 3 floats in micron,
    one per axis), ``orientation`` (e.g. "asr"), ``shape`` and ``symmetric``.
    '''
    import json
    atlas_folder = _find_brainglobe_folder(atlas)
    with open(atlas_folder/'metadata.json','r') as fd:
        return json.load(fd)


def get_brainglobe_resolution(atlas):
    '''Voxel size (micron) as a numpy array of length 3, one per axis.'''
    import numpy as np
    return np.array(get_brainglobe_metadata(atlas)['resolution'], dtype=float)


def get_structure_lookup(atlas):
    '''
    Build fast lookup maps from the atlas ``structures.json``.

    Returns a dict with:
      - ``id_to_acronym``      : {structure_id: acronym}
      - ``id_to_name``         : {structure_id: name}
      - ``id_to_rgb``          : {structure_id: [r, g, b]}
      - ``id_to_path``         : {structure_id: [ancestor_id, ..., structure_id]}
      - ``acronym_to_id``      : {acronym: structure_id}
      - ``structures``         : the source DataFrame

    ``id_to_path`` (the atlas ``structure_id_path``) lets a fine annotation be
    rolled up to a coarser hierarchy level for read-outs.
    '''
    structures = get_brainglobe_structure_data(atlas)
    id_to_acronym = dict(zip(structures['id'], structures['acronym']))
    id_to_name = dict(zip(structures['id'], structures['name']))
    id_to_rgb = {i: list(c) for i, c in
                 zip(structures['id'], structures['rgb_triplet'])}
    id_to_path = dict(zip(structures['id'], structures['structure_id_path']))
    acronym_to_id = dict(zip(structures['acronym'], structures['id']))
    # id 0 is "outside the brain" / unlabelled; make it resolvable.
    id_to_acronym.setdefault(0, 'root')
    id_to_name.setdefault(0, 'outside brain')
    id_to_rgb.setdefault(0, [0, 0, 0])
    id_to_path.setdefault(0, [0])
    return dict(id_to_acronym=id_to_acronym,
                id_to_name=id_to_name,
                id_to_rgb=id_to_rgb,
                id_to_path=id_to_path,
                acronym_to_id=acronym_to_id,
                structures=structures)


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