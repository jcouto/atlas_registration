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
