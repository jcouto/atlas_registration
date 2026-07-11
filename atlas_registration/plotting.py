
import numpy as np

def interact_check_rotate(x,cmap = 'gray_r',clim = None,
                    steps = 4,
                    include_atlas_label_indication = True, # write names areas in the oriented atlas
                    **kwargs):
    ''' 
    Function to manually approximate the alignment of the 
    raw stack to the reference atlas.
    '''
    import pylab as plt

    res = dict(angles = [0,0,0],
               flip_x = False,
               flip_y = False,
               points_view0 = [],
               points_view1 = [],
               points_view2 = [])
    
    fig = plt.figure(facecolor = 'w')
    ax = []
    ims = []
    
    iframe = [40,40,40]    
    xviews = [x.transpose(2,0,1),x.transpose(1,2,0),x]
    iframe = [i.shape[0]//2 for i in xviews]
    points = [[] for i in range(3)]
    points_plots = []
    fits = []
    angles_text = []
    if clim is None:
        clim = [np.percentile(x,10),np.percentile(x,99.5)]
        print(clim)
    for i in range(3):
        ax.append(fig.add_subplot(1,3,1+i))
    
        ims.append(ax[-1].imshow(xviews[i][iframe[i]],
                             clim=clim,
                             cmap = cmap,
                             **kwargs))
        points_plots.append(ax[-1].plot(np.nan,np.nan,'yo--',alpha=0.4)[0])
        fits.append(ax[-1].plot(np.nan,np.nan,'r-',alpha=0.4)[0])
        angles_text.append(ax[-1].text(0,-1,'',color = 'y'))
        plt.axis('off')
    if include_atlas_label_indication:
        ax[2].text(0,x.shape[1]//2,'cortex',color='orange',rotation=90,ha='center',va = 'center')
        ax[2].text(x.shape[0]//2,0,'cerebellum',color='orange',ha='center',va = 'center',fontsize = 9)

    def check_angles():
        for i in range(3):
            if len(points[i]):
                p = np.stack(points[i])
                res[f'points_view{i}'] = p
                if len(points[i])>1:
                    ft = np.polyfit(*p.T,1)
                    xx = [np.min(p[:,0]),np.max(p[:,0])]
                    yy = np.polyval(ft,xx)
                    res[f'xy_view{i}'] = np.stack([xx,yy])
                    x = xx[1]-xx[0]
                    y = yy[1]-yy[0]
                    if i == 0: # cos
                        res['angles'][i] = np.rad2deg(np.arcsin(y/np.sqrt(x**2 + y**2)))
                        if res['angles'][i] > 90:
                            res['angles'][i] = res['angles'][i] - 180
                    elif i == 1: # sin
                        res['angles'][i] = np.rad2deg(np.arccos(y/np.sqrt(x**2 + y**2)))
                        if res['angles'][i] > 90:
                            res['angles'][i] = res['angles'][i]- 180
                    elif i == 2:
                        print('Not implemented')

    def on_scroll(event):
        if event.inaxes is None:
            return
        increment = steps if event.button == 'up' else -steps
        idx = ax.index(event.inaxes)
        iframe[idx] += increment
        iframe[idx] = np.clip(iframe[idx],0,len(xviews[idx])-1)
        
        ims[idx].set_data(xviews[idx][iframe[idx]])
        ims[idx].figure.canvas.draw()
        
    def on_click(event):
        if event.inaxes is None:
            return
        idx = ax.index(event.inaxes)
        if idx == 2: # then check if you have to flip the axis
            if event.button == 1:
                res['flip_x'] = not res['flip_x']
                xviews[2] = xviews[2][:,:,::-1]
            elif event.button == 3:
                res['flip_y'] = not res['flip_y']
                xviews[2] = xviews[2][:,::-1,:]
            ims[idx].set_data(xviews[idx][iframe[idx]])
        else:
            if event.button == 1:
                points[idx].append([event.xdata,event.ydata])
                p = np.stack(points[idx])
                points_plots[idx].set_xdata(p[:,0])
                points_plots[idx].set_ydata(p[:,1])
                if len(points[idx])>1:
                    #compute the angles
                    check_angles()
                    fits[idx].set_xdata(res[f'xy_view{idx}'][0])
                    fits[idx].set_ydata(res[f'xy_view{idx}'][1])
                    angles_text[idx].set_text('{0:2.1f}deg'.format(res['angles'][idx]))
            if event.button == 3:
                points[idx].pop()
                if not len(points[idx]):
                    points_plots[idx].set_xdata(np.nan)   
                    points_plots[idx].set_ydata(np.nan)   
                else:
                    p = np.stack(points[idx])
                    points_plots[idx].set_xdata(p[:,0])
                    points_plots[idx].set_ydata(p[:,1])
        ims[idx].figure.canvas.draw()
        
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_release_event', on_click)
    plt.axis('off')
    plt.show()
    return res


def region_colors(region_ids, rgb_map, default=(120, 120, 120), scale=1 / 255.0):
    '''Map region ids -> Nx3 colours (floats in 0..1 by default) via ``rgb_map``
    ({region_id: [r, g, b]}, 0..255).  ``scale=1`` keeps 0..255.'''
    out = np.empty((len(region_ids), 3), float)
    for i, rid in enumerate(region_ids):
        out[i] = rgb_map.get(int(rid), default)
    return out * scale


def _rgb_str(c, scale255=True):
    c = np.asarray(c, float)
    if scale255 and c.max() <= 1.0:
        c = c * 255.0
    return f'rgb({int(c[0])},{int(c[1])},{int(c[2])})'

# 3D plotting functions and helpers
def atlas_surface(annotation, step_size=3, level=0):
    '''Marching-cubes surface ``(verts, faces)`` of ``annotation != level``.
    Verts are in the annotation's voxel space (downsample the annotation first
    for a lighter mesh; scale the verts back if you overlay full-res points).'''
    import skimage.measure as measure
    verts, faces, _, _ = measure.marching_cubes(np.asarray(annotation) != level,
                                                step_size=step_size)
    return verts, faces


def figure_3d(surface=None, camera='sagittal', height=650,
              mesh_color='lightgray', mesh_opacity=0.1):
    '''Base plotly 3D figure with an optional gray brain ``surface`` (verts,
    faces), axes hidden, data aspect, and a camera preset.'''
    import plotly.graph_objects as go
    fig = go.Figure()
    if surface is not None:
        verts, faces = surface
        v = np.asarray(verts, float)
        f = np.asarray(faces)
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=mesh_color, opacity=mesh_opacity,
                                hoverinfo='skip', showscale=False))
    cams = {'sagittal': dict(eye=dict(x=0, y=0, z=-2.2), up=dict(x=0, y=-1, z=0)),
            'coronal': dict(eye=dict(x=-2.2, y=0, z=0), up=dict(x=0, y=-1, z=0)),
            'horizontal': dict(eye=dict(x=0, y=-2.2, z=0), up=dict(x=-1, y=0, z=0))}
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                      aspectmode='data', camera=cams.get(camera))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=24, b=0),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig


def add_line_3d(fig, xyz, color='red', width=4, opacity=1.0, name=None):
    '''Add a 3D poly-line (e.g. a probe track) to ``fig``.'''
    import plotly.graph_objects as go
    xyz = np.asarray(xyz, float)
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
                               line=dict(color=color, width=width), opacity=opacity,
                               name=name, hoverinfo='skip' if name is None else 'name',
                               showlegend=name is not None))
    return fig


def add_points_3d(fig, xyz, colors=None, values=None, color=None,
                  cmap='Viridis', clim=None, size=3, name=None, text=None,
                  showscale=True, colorbar_title=None):
    '''Add 3D points to ``fig``, coloured by a single ``color``, by explicit
    per-point ``colors`` (Nx3 rgb), or by a scalar ``values`` array with colormap
    ``cmap`` and colour limits ``clim=(vmin, vmax)`` (colorbar).'''
    import plotly.graph_objects as go
    xyz = np.asarray(xyz, float)
    if color is not None:
        marker = dict(size=size, color=color)
    elif colors is not None:
        marker = dict(size=size, color=[_rgb_str(c) for c in np.asarray(colors)])
    elif values is not None:
        marker = dict(size=size, color=np.asarray(values, float), colorscale=cmap,
                      showscale=showscale, colorbar=dict(title=colorbar_title))
        if clim is not None:
            marker['cmin'], marker['cmax'] = float(clim[0]), float(clim[1])
    else:
        marker = dict(size=size)
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
                               marker=marker, name=name, text=text,
                               hovertemplate=(None if text is None else '%{text}<extra></extra>'),
                               showlegend=name is not None))
    return fig


def plot_points_3d(xyz, colors=None, values=None, surface=None, lines=None,
                   size=3, camera='sagittal', height=650, text=None,
                   colorbar_title=None, name='units'):
    '''Convenience: brain ``surface`` + optional ``lines`` (list of Nx3 tracks)
    + points coloured by region (``colors``) or a metric (``values``).'''
    fig = figure_3d(surface=surface, camera=camera, height=height)
    for i, ln in enumerate(lines or []):
        add_line_3d(fig, ln, color=_TRACK_COLORS[i % len(_TRACK_COLORS)], name=None)
    add_points_3d(fig, xyz, colors=colors, values=values, size=size, text=text,
                  name=name, colorbar_title=colorbar_title)
    return fig


_TRACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']


# 2D plotting functions
def plot_region_column(entries, exits, acronyms=None, colors=None, ax=None,
                       depth_domain=None, width=1.0, label_min_frac=0.02,
                       fontsize=8, x0=0.0, ylabel='depth from surface (µm)'):
    '''Colored region-vs-depth column (0 = surface at top).

    ``entries``/``exits`` are the region boundary depths (micron); ``acronyms``
    are optional labels; ``colors`` is an Nx3 array (0..1, one per region).
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(1.4, 6))
    entries = np.asarray(entries, float)
    exits = np.asarray(exits, float)
    n = len(entries)
    colors = np.full((n, 3), 0.78) if colors is None else np.asarray(colors, float)
    lo = depth_domain[0] if depth_domain else 0.0
    hi = depth_domain[1] if depth_domain else (float(exits.max()) if n else 1.0)
    span = max(hi - lo, 1.0)
    for i in range(n):
        h = exits[i] - entries[i]
        if h <= 0:
            continue
        ax.add_patch(Rectangle((x0, entries[i]), width, h, facecolor=colors[i],
                               edgecolor='white', linewidth=0.3))
        if acronyms is not None and h >= span * label_min_frac:
            lum = 0.299 * colors[i][0] + 0.587 * colors[i][1] + 0.114 * colors[i][2]
            ax.text(x0 + 0.05 * width, (entries[i] + exits[i]) / 2, str(acronyms[i]),
                    va='center', ha='left', fontsize=fontsize,
                    color='white' if lum < 0.55 else 'black')
    ax.set_xlim(x0, x0 + width)
    ax.set_ylim(hi, lo)                     # 0 (surface) on top
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def plot_units_depth(depths, colors=None, values=None, ax=None, x=None, cmap='viridis',
                     size=14, colorbar_title=None, ylabel='depth from surface (µm)'):
    '''Scatter units along depth (y), coloured by region (``colors`` Nx3, 0..1) or
    a scalar ``values``; ``x`` is an x position per unit (default small jitter).'''
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(2.2, 6))
    depths = np.asarray(depths, float)
    if x is None:
        x = np.random.default_rng(0).uniform(-1, 1, len(depths))
    if colors is not None:
        ax.scatter(x, depths, s=size, c=np.asarray(colors), edgecolors='none')
    else:
        sc = ax.scatter(x, depths, s=size, c=np.asarray(values, float), cmap=cmap,
                        edgecolors='none')
        cb = ax.figure.colorbar(sc, ax=ax)
        if colorbar_title:
            cb.set_label(colorbar_title)
    ax.set_ylim(depths.max() if len(depths) else 1, 0)
    ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def plot_region_bar(labels, counts, colors=None, ax=None, xlabel='n'):
    '''Horizontal bar chart of a per-region quantity (e.g. unit counts).'''
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, max(2, 0.25 * len(labels))))
    order = np.argsort(counts)
    labels = np.asarray(labels)[order]
    counts = np.asarray(counts)[order]
    cols = None if colors is None else np.asarray(colors)[order]
    ax.barh(range(len(labels)), counts, color=cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax
