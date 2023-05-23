from dataclasses import dataclass, field
from itertools import groupby

import matplotlib.patheffects as pe
import geopandas as gpd
from adjustText import adjust_text
import networkx as nx

from .grid import FvcomGrid
from .transect import Transect

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
        # based on all sections having nodes within in
        if len(transects) == 1:
            # Just pick an upstream node
            node = list(upstream_nodes)[0]
        else:
            # Start with the first transect, index 0
            # Pick a candidate note from each side of the transect
            candidates = [next(iter(bns)) for bns in border_nodes[0]]
            node = None
            # We need to test every other section to handle sections that
            # end at islands. It's then possible for mirror sides of two
            # sections to connect to each other.
            for nodes in border_nodes[1:]:
                tests = [next(iter(bns)) for bns in nodes]
                if(not nx.has_path(g, candidates[0], tests[0]) and
                        not nx.has_path(g, candidates[0], tests[1])):
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
            callback=None, **kwargs):
        cv_tces = self.tces
        if data is not None:
            cv_tces['data'] = data
            col = 'data'
            if 'legend' not in kwargs:
                kwargs['legend'] = True
        else:
            col = None
        ax = cv_tces.plot(col, zorder=2, **kwargs)
        xmin, xmax, ymin, ymax = ax.axis()
        if base == 'elements':
            grid_base = self.grid.elements_gdf()
            grid_base.plot(ax=ax, facecolor='#ccc', edgecolors='#aaa', zorder=1)
        elif base == 'union':
            grid_els = self.grid.elements_gdf()
            grid_base = gpd.GeoDataFrame({'geometry': [grid_els['geometry'].unary_union]}, crs=grid_els.crs)
            grid_base.plot(ax=ax, facecolor='#ccc', edgecolors='k',
                    zorder=1)

        if callback is not None:
            callback(self, ax)

        if label is not None:
            pt = cv_tces['geometry'].unary_union.representative_point()
            ax.annotate(label, (pt.x, pt.y), ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=3,
                        foreground='white', alpha=0.6)]
            )
        ax.set(ybound=(ymin, ymax), xbound=(xmin, xmax))
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

    def plot(self, transect_labels=None, callback=None, **kwargs):
        if transect_labels is not None:
            texts = []
            avoid_x = []
            avoid_y = []
            def on_plot(self, ax):
                for t,label in zip(self.transects, transect_labels):
                    ls = t.to_geom()
                    gs = gpd.GeoSeries(ls)
                    gs.plot(ax=ax, color='k', zorder=3)
                    texts.append(ax.annotate(label,
                        xy=(ls.centroid.x, ls.centroid.y), ha='center',
                        va='center', zorder=4,
                        path_effects=[pe.withStroke(linewidth=3,
                            foreground='white', alpha=0.6)]
                    ))
                    avoid_x.extend(ls.coords.xy[0])
                    avoid_y.extend(ls.coords.xy[1])
                if callback is not None:
                    callback(self, ax)
            cb = on_plot
        else:
            cb = callback
        ax = super().plot(callback=cb, **kwargs)
        if transect_labels is not None:
            adjust_text(texts, avoid_x, avoid_y)
