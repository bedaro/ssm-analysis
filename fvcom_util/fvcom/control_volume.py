from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import matplotlib.patheffects as pe
import geopandas as gpd
from adjustText import adjust_text
import networkx as nx

from .grid import FvcomGrid
from .transect import Transect

text_outline = pe.withStroke(linewidth=3, foreground='white', alpha=0.6)

@dataclass(frozen=True, kw_only=True)
class ControlVolume:
    grid: FvcomGrid
    nodes: set
    calc: bool = False

    def __post_init__(self):
        if self.calc:
            object.__setattr__(self, "_tces", self._calc_tces())

    @staticmethod
    def from_transects(transects, calc=False):
        # Check that all transects were made from the same Grid
        grids = [t.grid for t in transects]
        g = groupby(grids)
        assert next(g, True) and not next(g, False), "Grids don't match"

        grid = grids[0]

        border_nodes = [t.get_nodes() for t in transects]
        # Get the grid's node adjacency, then remove all connections
        # between upstream and downstream nodes on all transects. If
        # caclulations were done correctly, this will break up the graph
        # into separate components, one of which will be our control
        # volume
        adj_dict = grid.node_neis()
        for (upstream_nodes, downstream_nodes) in border_nodes:
            for un in upstream_nodes:
                adj_dict[un] -= downstream_nodes
            for dn in downstream_nodes:
                adj_dict[dn] -= upstream_nodes

        g = nx.Graph(adj_dict)

        # Use the Graph to find which component is our control volume,
        # based on all sections having nodes within it
        if len(transects) == 1:
            # Just pick an upstream node
            node = list(upstream_nodes)[0]
        else:
            # Start with the first transect, index 0
            # Pick a candidate node from each side of the transect
            candidates = [next(iter(bns)) for bns in border_nodes[0]]
            node = None
            # We need to test every other section to handle sections that
            # end at islands. It's then possible for mirror sides of two
            # sections to connect to each other.
            for nodes in border_nodes[1:]:
                tests = [next(iter(bns)) for bns in nodes]
                paths = np.array(
                        [[nx.has_path(g, c, t) for t in tests[:2]] for c in candidates[:2]])
                assert (not (paths[0,0] and paths[0,1])
                    and not (paths[1,0] and paths[1,1])), 'Loop detected, invalid CV'
                if not paths[0,0] and not paths[0,1]:
                    node = candidates[1]
                    break
            else:
                node = candidates[0]
        nodes = nx.node_connected_component(g, node)

        return TransectControlVolume(grid=grid, nodes=nodes,
                transects=transects, calc=calc)

    @property
    def nodes_list(self):
        """The nodes as a list"""
        nlist = list(self.nodes)
        nlist.sort()
        return nlist

    @property
    def tces(self):
        if self.calc:
            return self._tces.copy()
        else:
            return self._calc_tces()

    def _calc_tces(self):
        return self.grid.tces_gdf().loc[self.nodes_list]

    @property
    def area(self):
        """The total CV area."""
        return self.tces['geometry'].area.sum()

    def plot(self, data=None, label=None, base='union',
            helpers=[], **kwargs):
        cv_tces = self.tces
        if data is not None:
            cv_tces['data'] = data
            col = 'data'
            if 'legend' not in kwargs:
                kwargs['legend'] = True
        else:
            col = None
        ax = cv_tces.plot(col, zorder=2, **kwargs)
        if ax is None:
            ax = kwargs['ax']
        xmin, xmax, ymin, ymax = ax.axis()
        if base == 'elements':
            grid_base = self.grid.elements_gdf()
            grid_base.plot(ax=ax, facecolor='#ccc', edgecolors='#aaa', zorder=1)
        elif base == 'union':
            grid_els = self.grid.elements_gdf()
            grid_base = gpd.GeoDataFrame({'geometry': [grid_els['geometry'].unary_union]}, crs=grid_els.crs)
            grid_base.plot(ax=ax, facecolor='#ccc', edgecolors='k',
                    zorder=1)

        for h in helpers:
            h.reset()
            h.set_cv(self)
            h(ax)

        texts = []
        avoid_x = []
        avoid_y = []
        if label is not None:
            pt = cv_tces['geometry'].unary_union.representative_point()
            ax.annotate(label, (pt.x, pt.y), ha='center', va='center',
                    path_effects=[text_outline]
            )
            avoid_x.append(pt.x)
            avoid_y.append(pt.y)
        ax.set(ybound=(ymin, ymax), xbound=(xmin, xmax))
        for h in helpers:
            texts.extend(h.texts)
            avoid_x.extend(h.avoid_x)
            avoid_y.extend(h.avoid_y)
        if len(texts):
            adjust_text(texts, avoid_x, avoid_y)
        return ax

    def __sub__(self, ns: set):
        return ControlVolume(grid=self.grid, nodes=self.nodes - ns,
                calc=self.calc)

    def __add__(self, ns: set):
        return ControlVolume(grid=self.grid, nodes=self.nodes + ns,
                calc=self.calc)

@dataclass(frozen=True, kw_only=True)
class TransectControlVolume(ControlVolume):
    transects: list[Transect]

    def transect_directions(self):
        """List of bools; True if upstream from transect is in CV"""
        cv_node = next(iter(self.nodes))
        directions = []
        for t in self.transects:
            up, down = t.get_nodes()
            directions.append(next(iter(up)) in self.nodes)
        return directions

    def __sub__(self, ns: set):
        return TransectControlVolume(grid=self.grid,
                nodes=self.nodes - ns, transects=self.transects,
                calc=self.calc)

    def __add__(self, ns: set):
        return TransectControlVolume(grid=self.grid,
                nodes=self.nodes + ns, transects=self.transects,
                calc=self.calc)

    def plot(self, transect_labels=None, helpers=[], **kwargs):
        if transect_labels is not None:
            helpers.append(TransectHelper(self.cv.transects, transect_labels))
        return super().plot(helpers=helpers, **kwargs)

class PlotHelper:
    cv = None
    avoid_x = []
    avoid_y = []
    texts = []

    def set_cv(self, cv):
        self.cv = cv

    def reset(self):
        self.avoid_x = []
        self.avoid_y = []
        self.texts = []

    def __call__(self, ax):
        """Perform any additional plotting before axis bounds are adjusted"""
        raise NotImplementedError("Subclass must override")

class TransectHelper(PlotHelper):
    def __init__(self, transects, transect_labels):
        self.transects = transects
        self.transect_labels = transect_labels

    def __call__(self, ax):
        for t,label in zip(self.transects, self.transect_labels):
            ls = t.to_geom()
            gs = gpd.GeoSeries(ls)
            gs.plot(ax=ax, color='k', zorder=3)
            self.texts.append(ax.annotate(label,
                xy=(ls.centroid.x, ls.centroid.y), ha='center',
                va='center', zorder=4,
                path_effects=[text_outline]
            ))
            self.avoid_x.extend(ls.coords.xy[0])
            self.avoid_y.extend(ls.coords.xy[1])

class StationHelper(PlotHelper):
    sites_df = None

    def __init__(self, sites: dict):
        self.sites = sites

    def __call__(self, ax):
        if self.sites_df is None:
            self.sites_df = self.cv.grid.nodes_gdf().loc[
                    self.sites.keys()].copy()
            self.sites_df['name'] = ''
            for n,s in self.sites.items():
                self.sites_df.loc[n, 'name'] = s
        self.sites_df.plot(ax=ax, color='k', marker='^', alpha=0.6, zorder=3)
        for n,row in self.sites_df.iterrows():
            self.texts.append(ax.annotate(row['name'],
                xy=(row['geometry'].x, row['geometry'].y), ha='center',
                va='center', zorder=4, fontstyle='italic',
                path_effects=[text_outline]
            ))
            self.avoid_x.append(row['geometry'].x)
            self.avoid_y.append(row['geometry'].y)
