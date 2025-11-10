import numpy as np

def get_line_trajectory(data,volume_shape = None):
    '''
    Get the linear approximation to a trajectory of a probe given labelings in the atlas space.
    TODO: Make that is uses the spline approximation for when it is inside the labeling.
    '''
    data_mean = data.mean(axis = 0)
    _,_,vv = np.linalg.svd(data-data_mean) #
    linepoints = vv[0]*np.arange(-2000,2000,2)[:, np.newaxis] + data_mean
    linepoints = linepoints[(linepoints>0).sum(axis = 1)==3]
    if not volume_shape is None:
        linepoints = linepoints[(linepoints<volume_shape).sum(axis = 1)==3]
    print(linepoints[0].reshape(1,-1), linepoints[-1].reshape(1,-1))
    lineidx = bresenhamlines(linepoints[0].reshape(1,-1), linepoints[-1].reshape(1,-1),10000).squeeze()
    lineidx = lineidx[(lineidx>0).sum(axis = 1)==3]
    if not volume_shape is None:
        lineidx = lineidx[(lineidx<volume_shape).sum(axis = 1)==3]
    return lineidx.astype(int)


import numpy as np
def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.
    This is adapted from https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope

def bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension) 
    This is adapted from https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
    TODO: Make it work for following an arbitrary trajectory
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)
    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T
    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat
    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)