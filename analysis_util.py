import matplotlib.pyplot as plt
import geopandas as gpd

def read_base_map(shapefile):
    global _base_map
    _base_map = gpd.read_file(shapefile)
    return _base_map

def map_plot(domain, vals, title, ax=None, **kwargs):
    """
    Makes a map by loading the given set of values for all the nodes into a
    copy of the domain GeoPandas DataFrame, then plotting. A few stylistic
    improvements are also made, and some customization is possible.
    
    Parameters
    ----------
    domain: `gpd.GeoDataFrame`
        The shapefile for the domain containing all nodes to plot.
    vals: `np.Array`
        The node values to plot on the map.
    title: str
        The (axis) title to give the plot.
    ax: `matplotlib.axes.Axes`
        The axes to plot on, default is to generate a new plot
    
    Returns
    -------
    `geopandas.GeoDataFrame`
        The assembled copy of the model domain DataFrame with a column labeled `values`
        which contains the `vals` data passed in.
    `matplotlib.figure.Figure` or None
        If a new figure was created it is returned.
    """
    copy = domain.copy()
    copy['values'] = vals
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    else:
        fig = None
    if 'legend' not in kwargs:
        kwargs['legend'] = True
    p = copy.plot('values', ax=ax, markersize=1, zorder=2, **kwargs)
    # Save the axes boundaries so we can restore them after plotting the full model
    # boundary
    xmin, xmax, ymin, ymax = p.axis()
    _base_map.plot(ax=ax, facecolor='#ccc', edgecolors='black', zorder=1)
    ax.set(title=title, ybound=(ymin, ymax), xbound=(xmin, xmax), xticklabels=(), yticklabels=())
    return copy, fig
