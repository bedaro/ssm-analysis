from dataclasses import dataclass, field
from multiprocessing import Pool
from functools import partial
# Temporary workaround for recent geopandas
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import pandas as pd
import geopandas as gpd
import py2dm
from shapely.geometry import Point,Polygon

@dataclass(frozen=True)
class FvcomGrid:
    """Class to represent an unstructured FVCOM grid

    TODO:
    - support classification of boundary condition nodes
    - add methods to generate grid and bathymetry input files
    """
    ncoord: np.array # 2xM or 3xM array of node coordinates
    nv: np.array     # 3xN array of node vertices for each element
    calc: bool = field(default=False) # Whether or not to calculate extras

    def __post_init__(self):
        # Ensure that array shapes are correct
        assert self.nv.shape[0] == 3, f'FvcomGrid nv shape {self.nv.shape} is not (3, n)'
        assert self.ncoord.shape[0] in (2,3), f'FvcomGrid ncoord shape {self.ncoord.shape} is not (2, m) or (3, m)'
        # Using lru_cache won't work because then the class isn't thread-safe,
        # and the cache won't be available across threads. So the only way
        # to accommodate caching on a frozen dataclass is to hack it.
        if self.calc:
            object.__setattr__(self, "_nbe", self._el_neis())
            object.__setattr__(self, "_elcoord", self._calc_elcoord())

    @staticmethod
    def from_output(dataset, calc=True):
        """Read the grid defined in an output NetCDF file"""
        return FvcomGrid(np.array([dataset['x'][:], dataset['y'][:],
            dataset['h'][:]]), dataset['nv'][:].astype(int), calc=calc)

    @staticmethod
    def from_gridfile(gridfile, depfile=None, calc=True):
        """Read the grid defined in a FVCOM grid input file

        Based on subroutine GETDIM (getdim.F) from FVCOM 2.7"""
        # We don't care about the fifth column so ignore it
        j, nvtmp1, nvtmp2, nvtmp3 = np.genfromtxt(gridfile, usecols=(0, 1, 2, 3),
                unpack=True, dtype=np.int32)
        # The number of nodes is the last value of j
        m = j[-1]
        # The number of elements is the last value of j before the count resets
        # (causing the following value to be smaller)
        n = j[np.nonzero(j[:-1] > j[1:])[0][0]]
        # Lookup arrays for vertex nodes 1, 2, and 3 of each element
        nv = np.array([nvtmp1[:n], nvtmp2[:n], nvtmp3[:n]])
        # Node coordinate array
        xy = [nvtmp1[n:], nvtmp2[n:]]
        if depfile is not None:
            nodex, nodey, nodez = np.loadtxt(depfile, unpack=True)
            xy.append(nodez)
        ncoord = np.array(xy)

        return FvcomGrid(ncoord, nv, calc=calc)

    @staticmethod
    def from_mesh(meshfile, calc=True):
        """Read the grid defined in a 2DM mesh file"""
        with py2dm.Reader(meshfile) as mesh:
            ncoord = np.zeros((3, mesh.num_nodes))
            nv = np.zeros((3, mesh.num_elements), dtype=np.int64)

            for node in mesh.iter_nodes():
                ncoord[:,node.id-1] = node.pos
            for el in mesh.iter_elements():
                nv[:,el.id-1] = el.nodes

        return FvcomGrid(ncoord, nv, calc=calc)

    @property
    def m(self) -> int:
        return self.ncoord.shape[1]

    @property
    def n(self) -> int:
        return self.nv.shape[1]

    @property
    def nbe(self) -> np.array:
        if not self.calc:
            raise ValueError("nbe needs to be calculated at init. Pass calc=True)")
        return self._nbe

    @property
    def elcoord(self) -> np.array:
        """Return the coordinates (X,Y,H) of the element centers"""
        if not self.calc:
            raise ValueError("elcoord needs to be calculated at init. Pass calc=True)")
        return self._elcoord

    def el_dist(self, el1, el2):
        """Calculate the distance between the two given elements"""
        xc,yc = tuple(self.elcoord[(0,1),:])
        return np.sqrt((xc[el2-1]-xc[el1-1])**2+(yc[el2-1]-yc[el1-1])**2)

    def n_dist(self, n1, n2):
        """Calculate the distance between the two given nodes"""
        return np.sqrt(((self.ncoord[:,n1-1]-self.ncoord[:,n2-1])**2).sum())

    def nodes_gdf(self, crs='epsg:32610'):
        """Create a GeoDataFrame of all the nodes"""
        nodes = np.empty(self.m, dtype=object)
        depths = np.zeros(self.m)
        for i,n in enumerate(self.ncoord.swapaxes(1, 0)):
            nodes[i] = Point(n[0], n[1])
            if len(n) > 2:
                depths[i] = n[2]
        return gpd.GeoDataFrame({"geometry": nodes, "depth": depths},
                crs=crs, index=pd.RangeIndex(1, self.m+1))

    def elements_gdf(self, crs='epsg:32610'):
        """Create a GeoDataFrame of all the elements"""
        els = []
        for ns in self.nv.swapaxes(1, 0):
            els.append(Polygon([self.ncoord[:,n-1] for n in ns]))
        return gpd.GeoDataFrame({"geometry": els}, crs=crs,
                index=pd.RangeIndex(1, self.n+1))

    def _azimuth(p1, p2):
        return np.arctan2(p2.y - p1.y, p2.x - p1.x)

    def _build_shape(self, node_id):
        node_pt = Point(self.ncoord[0,node_id-1], self.ncoord[1,node_id-1])

        # Start by gathering the centroids of all the elements
        elements = (self.nv == node_id).nonzero()[1] + 1
        el_centroids = [Point(x, y) for x, y in self.elcoord[0:2, elements - 1].T]

        # Get the list of list of nodes that are part of those elements
        neighbors = self.nv[:,elements - 1]

        # Get all the midpoints between this node and its neighbors.
        # If there are any neighbor nodes that are only part of one
        # element, then this is an edge node and the segment between this
        # node and that neighbor constitutes an edge
        nei_nodes, nei_counts = np.unique(neighbors[neighbors != node_id],
                return_counts=True)
        midpoints = [Point(x, y) for x, y in (
            (self.ncoord[0:2, nei_nodes - 1].T +
            self.ncoord[0:2, node_id - 1]) / 2)]
        points = el_centroids + midpoints
        edge_mids = [midpoints[i] for i in (nei_counts == 1).nonzero()[0]]

        # Order the points around the node with a sort by angle
        points.sort(key=lambda x: FvcomGrid._azimuth(node_pt, x))
        # If there are midpoints added (edge node), add the node point
        # itself to the list in between the two
        if len(edge_mids):
            # There are guaranteed to be exactly two midpoints if the grid
            # is structured properly
            try:
                m1 = edge_mids[0]
                m2 = edge_mids[1]
            except IndexError as e:
                raise IndexError(f'Node {node_id} has {len(edge_mids)} edge?!')
            i1 = points.index(m1)
            i2 = points.index(m2)
            # If they are consecutive, insert between
            if abs(i1 - i2) == 1:
                points.insert(max(i1, i2), node_pt)
            else:
                # The only other possibility is that one is at the end of
                # the list and the other at the beginning, so just append
                # the node_pt to the end
                points.append(node_pt)

        shape = Polygon([(p.x, p.y) for p in points])
        return shape

    def tces_gdf(self, crs='epsg:32610'):
        """Create a GeoDataFrame of all the node tracer control elements"""

        node_ids = np.arange(1, self.m+1)
        with Pool() as pool:
            node_shapes = pool.map(self._build_shape, node_ids)
        #node_shapes = [self._build_shape(n) for n in node_ids]

        tces = gpd.GeoDataFrame({
            "geometry": node_shapes
        }, index=node_ids, crs=crs)
        return tces

    def _meaner(b, a):
        return a[:, b].mean(axis=1)

    def _calc_elcoord(self):
        # Calculating the centroid values takes a long time without making it
        # multithreaded
        with Pool() as p:
            elcoord = np.array(p.map(
                partial(FvcomGrid._meaner, a=self.ncoord), self.nv.T - 1))
        return elcoord.T

    def _el_neis(self):
        """Computes NBE (indices of element neighbors) just like in FVCOM."""
        nbe = np.zeros((3, self.n), int)
        node_els = [[] for x in range(self.m)]
        for i,ns in enumerate(self.nv.T - 1):
            for n in ns:
                node_els[n].append(i)
        for i,nl in enumerate(node_els):
            node_els[i] = np.array(nl)
        for i,(n1,n2,n3) in enumerate(self.nv.T - 1):
            pairs = ((n2,n3),(n1,n3),(n1,n2))
            for pi,(p1,p2) in enumerate(pairs):
                overlap = np.intersect1d(node_els[p1], node_els[p2])
                # There can't be more than two elements returned, so find the
                # element that is not equal to i (else zero)
                nbe[pi,i] = np.where(overlap != i, overlap + 1, 0).max()
        return nbe

    def node_neis(self):
        """Calculates the neighbors of all nodes as an adjacency dict"""
        adj_dict = {}
        for i in range(self.m):
            adj_dict[i+1] = set()
        for n1,n2,n3 in self.nv.T:
            adj_dict[n1] |= set((n2, n3))
            adj_dict[n2] |= set((n1, n3))
            adj_dict[n3] |= set((n1, n2))
        return adj_dict

    def el_neis(self):
        """Calculates the neighbors of all elements as an adjacency adj_dict

        Simple transformation of nbe can also work, but nbe contains
        zeros for boundary elements. This would confuse a graph system
        like NetworkX."""

        ele_adj_dict = dict()
        for i,els in enumerate(self.nbe.T,1):
            ele_adj_dict[i] = els[els > 0]
        return ele_adj_dict

    def to_mesh(self, meshfile):
        with py2dm.Writer(meshfile) as mesh:
            for i,node in enumerate(self.ncoord.T):
                mesh.node(i+1, *node)
            for i,el in enumerate(self.nv.T):
                mesh.element('E3T', i+1, *el)

# Simple grid generators for testing purposes
def uniform_triangular(sz=3):
    """Create a simple triangular grid of equilateral triangles"""
    ncoord = []
    for i in range(sz):
        y = np.sqrt(3)/2 * i
        x0 = i / 2
        for j in range(sz-i):
            ncoord.append([x0 + j, y])
    ncoord = np.array(ncoord).T

    nv = []
    for i in range(0,2*sz-3,2):
        # handle even row
        # First node on this row
        v0 = sz * i // 2 + 1 - (
            ((i // 2 - 1) ** 2 + i // 2 - 1) // 2 if i // 2 - 1 > 0 else 0
        )
        # First node on next row
        v1 = v0 + sz - i // 2
        for j in range(sz - i//2 - 1):
            nv.append([v0 + j, v0 + j + 1, v1 + j])
        # handle odd row
        for j in range(sz - i//2 - 2):
            nv.append([v1 + j, v0 + j + 1, v1 + j + 1])
    nv = np.array(nv).T

    return FvcomGrid(ncoord, nv, calc=True)
