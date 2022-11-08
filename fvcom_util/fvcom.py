from dataclasses import dataclass, field
from multiprocessing import Pool
from functools import partial

import numpy as np
import geopandas as gpd

@dataclass(frozen=True)
class FvcomGrid:
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
        return FvcomGrid(np.array([dataset['x'][:], dataset['y'][:],
            dataset['h'][:]]), dataset['nv'][:].astype(int), calc=calc)

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

    def elements_gdf(self):
        """Create a GeoDataFrame of all the elements

        FIXME make the CRS an argument or class property"""
        els = []
        for ns in self.nv.swapaxes(1, 0):
            els.append(Polygon([self.ncoord[:,n-1] for n in ns]))
        return gpd.GeoDataFrame({"geometry": els}, crs='epsg:32610')

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
