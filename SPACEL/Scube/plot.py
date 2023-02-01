import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np

def plot_3d(
    loc,
    val=None, 
    color=None, 
    figsize=(8,8), 
    return_fig=False, 
    elev=None, 
    azim=None, 
    xlim=None, 
    ylim=None, 
    zlim=None, 
    frameon=True, 
    save_path=None, 
    save_dpi=150, 
    show=True,
    *args,
    **kwargs
):
    """Plot all slices stacked in 3D
    
    Plot all slices stacked with a given number of subplots rows and columns. Spots/cells colored by spatial domain.

    Args:
        loc: An array of the coordinates of each spots/cells in all slices, which the first three columns are X-axis, Y-axis, Z-axis coordinates.
        val: The colors of each spots/cells given in loc use to plot. 
        color: The colors of each spots/cells given in loc use to plot. 
        figsize: Size of the figure.
        return_fig: Whether to return the figure. 
        elev: The elevation angle in the vertical plane in degrees. If ``None`` then the initial value as specified in the ``Axes3D`` constructor is used.
        azim: The azimuth angle in the horizontal plane in degrees. If ``None`` then the initial value as specified in the ``Axes3D`` constructor is used. 
        xlim: A tuple given the left and right xlims in X-axis. 
        ylim: A tuple given the left and right xlims in Y-axis. 
        zlim: A tuple given the left and right xlims in Z-axis. 
        frameon: Whether to hide the coordinate axes.  
        save_path: A string representing the path directory where the figure saved. 
        save_dpi: The resolution in dots per inch.
        show: Whether to show the figure. 
    
    Returns:
        A ``matplotlib.figure.Figure`` object
    """
    if 'marker' not in kwargs.keys():
        kwargs['marker'] = 'o'
    if 's' not in kwargs.keys():
        kwargs['s'] = 5
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'Spectral_r'
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    if (elev is not None) and (azim is not None):
        ax.view_init(elev, azim)  # 设定视角
    if color is None:
        ax.scatter(loc[:,0], loc[:,1], loc[:,2],c=val,*args,**kwargs)
    else:
        ax.scatter(loc[:,0], loc[:,1], loc[:,2],c=color,*args,**kwargs)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    if not frameon:
        plt.axis('off')
    if save_path is not None:
        print(save_path)
        plt.savefig(save_path,dpi=save_dpi,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    if return_fig:
        return fig

def plot_single_slice(adata, spatial_key, cluster_key, frameon=False, i=1, j=1, n=1, s=1):
    ind = np.sort(adata.obs[cluster_key].unique().copy())
    color = adata.uns[cluster_key+'_colors'].copy()
    dic = dict(zip(ind, color))
    col = adata.obs[cluster_key].replace(dic)
    ax = plt.subplot(i,j,n)
    if type(adata.obsm[spatial_key]) == pd.core.frame.DataFrame:
        plt.scatter(adata.obsm[spatial_key].iloc[:,0], adata.obsm[spatial_key].iloc[:,1], c=col, s=s, rasterized=True)
    if type(adata.obsm[spatial_key]) == np.ndarray:
        plt.scatter(adata.obsm[spatial_key][:,0], adata.obsm[spatial_key][:,1], c=col, s=s, rasterized=True)
    plt.axis('equal')
    if not frameon:
        plt.axis('off')

def plot_stacked_slices(
    ad_list, 
    spatial_key, 
    cluster_key, 
    legend=True, 
    frameon=False, 
    colors=None, 
    i=1, 
    j=1, 
    s=1
):
    """Plot all slices stacked
    
    Plot all slices stacked in one figure with a given number of subplots rows and columns. Spots/cells colored by spatial domain.

    Args:
        ad_list: A list of ``AnnData`` objects containing all slices.
        spatial_key: A string representing one key of ``obsm`` in AnnData object of all slices, containing the coordinates used to plot.
        cluster_key: A string representing one column of ``obs`` in AnnData object of all slices, containing the spatial domain information used to plot.
        legend: Whether to display the legend.
        frameon: Whether to hide the coordinate axes. 
        colors: A list of colors for each spatial domain to plot. If ``None``, it will default to ``tab10`` or ``tab20`` accoording to the number of spatial domains.
        i: Number of rows of the subplots.
        j: Number of columns of the subplots. If i=j=1, it will be a single figure.
        s: Size of points.
    
    Returns:
        ``None``
    """
    
    clusters = []
    colored_num = 0
    for ad in ad_list:
        if f'{cluster_key}_colors' in ad.uns.keys():
            colored_num += 1
        clusters.extend(ad.obs[cluster_key].cat.categories)
    clusters = np.unique(clusters)
    if colored_num < len(ad_list):
        if colors is None:
            if len(clusters) > 10:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab20',n_colors=len(clusters))]
            else:
                colors = [matplotlib.colors.to_hex(c) for c in sns.color_palette('tab10',n_colors=len(clusters))]
        color_map = pd.DataFrame(colors,index=clusters,columns=['color'])
        for ad in ad_list:
            ad.uns[f'{cluster_key}_color_map'] = color_map
            ad.uns[f'{cluster_key}_colors'] = [color_map.loc[c,'color'] for c in ad.obs[cluster_key].cat.categories]
    else:
        color_map = pd.DataFrame(index=clusters,columns=['color'])
        for ad in ad_list:
            color_map.loc[ad.obs[cluster_key].cat.categories,'color'] = ad.uns[f'{cluster_key}_colors']
            ad.uns[f'{cluster_key}_color_map'] = color_map
    if legend:
        legend_elements = [ Line2D([], [], marker='.', markersize=10, color=color_map.loc[i,'color'], linestyle='None', label=i) for i in color_map.index]
        for k in range(len(ad_list)):
            plot_single_slice(ad_list[k], spatial_key, cluster_key, frameon, i, j, n=np.min([k+1,i*j]), s=s)
        plt.legend(handles=legend_elements, loc='right')
        