from dataclasses import dataclass, field

import numpy as np
import networkx as nx

from .grid import FvcomGrid

@dataclass(frozen=True)
class Transect:
    """Class to represent a transect of elements across a grid"""

    grid: FvcomGrid
    eles: np.array

    def __post_init__(self):
        # TODO ensure elements form a continuous transect
        self._calc_midpoints()
        self._calc_normals()

    def _calc_midpoints(self):
        midpoints = np.zeros((self.grid.ncoord.shape[0], len(self.eles)+1))
        # First, handle the first and last elements which have an edge on
        # the model boundary. Find those boundary locations
        for b in range(0, len(self.eles)+1):
            nodes = self._el_node_pair(b)
            midpoints[:,b] = self.grid.ncoord[:, nodes - 1].mean(axis=1)
        object.__setattr__(self, "midpoints", midpoints)

    def _calc_normals(self):
        ele_xys = self.ele_xys
        # Angle normal to the line from the centroid to the previous
        # midpoint
        dxy1 = self.midpoints[0:2,:-1] - ele_xys
        th1 = np.arctan2(-dxy1[1], -dxy1[0])
        th1 = (np.where(th1 < 0, th1 + 2 * np.pi, th1) - np.pi / 2) % (2 * np.pi)
        # Angle normal to line from the centroid to the next midpoint
        dxy2 = self.midpoints[0:2, 1:] - ele_xys
        th2 = np.arctan2(dxy2[1], dxy2[0])
        th2 = (np.where(th2 < 0, th2 + 2 * np.pi, th2) - np.pi / 2) % (2 * np.pi)
        # Convert vectors to x/y coordinates
        ns1 = np.zeros((len(self.eles), 2))
        ns2 = np.zeros_like(ns1)
        ns1[:,0] = np.cos(th1)
        ns1[:,1] = np.sin(th1)
        ns2[:,0] = np.cos(th2)
        ns2[:,1] = np.sin(th2)
        object.__setattr__(self, "ns1", ns1)
        object.__setattr__(self, "ns2", ns2)

    def _el_boundary_nodes(self, ele):
        # The nodes listed in grid.nv in the same locations as the
        # neighbor elements (from nbe) are the boundary nodes
        return self.grid.nv[self.grid.nbe[:,ele-1].nonzero()[0], ele-1]

    def _el_node_pair(self, i):
        if i == 0:
            return self._el_boundary_nodes(self.eles[i])
        elif i == len(self.eles):
            return self._el_boundary_nodes(self.eles[-1])
        e1 = self.eles[i-1]
        e2 = self.eles[i]
        # These are the common nodes between the neighbor elements
        which_prev_nei = (self.grid.nbe[:, e1 - 1] == e2).nonzero()[0]
        non_nei_idxs = (np.arange(3) != which_prev_nei).nonzero()[0]
        common_nodes = self.grid.nv[non_nei_idxs, e1 - 1]
        return common_nodes

    @property
    def size(self):
        return self.eles.size

    @staticmethod
    def shortest(grid, waypoint_els):
        # Remove any zeros from the nbe's
        ele_adj_dict = grid.el_neis()
        G = nx.Graph(ele_adj_dict)
        assert hasattr(waypoint_els, "__len__")
        scalar_arg = not hasattr(waypoint_els[0], "__len__")
        if scalar_arg:
            waypoint_els = [waypoint_els]
        trs = []
        # TODO parallelize me. Move G creation inside
        for i,wps in enumerate(waypoint_els):
            assert len(wps) >= 2, f'Not enough waypoints defined for section {i}'

            # Find all the elements by taking the shortest path in the graph
            # through the given waypoints
            eles = []
            for w,x in zip(wps[:-1], wps[1:]):
                eles.extend(nx.shortest_path(G, source=w, target=x,
                    weight=lambda p1, p2, atts: grid.el_dist(p1, p2)))
            trs.append(Transect(grid, np.array(eles)))
        return trs[0] if scalar_arg else trs

    def get_n_bisectors(self):
        """Get unit vectors for each element by bisecting their segments"""
        bisectors = self.ns1 + self.ns2
        return (bisectors.T / np.sqrt(
            bisectors[:,0] ** 2 + bisectors[:,1] ** 2)).T

    def get_nodes(self):
        """Get all border nodes as a two-tuple of upstream/downstream sets"""
        upstream_nodes = set()
        downstream_nodes = set()
        ns = self.get_n_bisectors()
        for i,(ele,mid,n) in enumerate(zip(self.eles, self.midpoints[0:2, :-1].T, ns)):
            nodes = self._el_node_pair(i)
            # Construct a vector from the element centroid to the shared/edge
            # midpoint
            center = self.grid.elcoord[:2, ele-1]
            a = mid - center
            # Project the midpoint coordinates onto the a/n coordinate system
            # centered on the element centroid.
            A = np.array([a, n]).T
            Ainv = np.linalg.inv(A)
            nodes_recenter = self.grid.ncoord[:2, nodes-1] - np.broadcast_to(center, (2,2)).T
            nodes_xform = Ainv @ nodes_recenter
            # The normal vector "n" points in the up-estuary direction. So the
            # node coordinate transformed into the (a,n) vector space will have a
            # positive n component if the node is on the upstream side.
            upstream_nodes |= set(nodes[np.nonzero(nodes_xform[1] > 0)[0]])
            downstream_nodes |= set(nodes[np.nonzero(nodes_xform[1] < 0)[0]])
        # Find the other edge node of the last element
        nodes = self._el_boundary_nodes(self.eles[-1])
        # One of the edge nodes has been added to either upstream_nodes or
        # downstream_nodes. Add the other one to the other set
        if nodes[0] in upstream_nodes:
            downstream_nodes.add(nodes[1])
        elif nodes[0] in downstream_nodes:
            upstream_nodes.add(nodes[1])
        elif nodes[1] in upstream_nodes:
            downstream_nodes.add(nodes[0])
        elif nodes[1] in downstream_nodes:
            upstream_nodes.add(nodes[0])
        return (upstream_nodes, downstream_nodes)

    @property
    def ele_xys(self):
        return self.grid.elcoord[0:2,self.eles-1]

    @property
    def a(self):
        """The horizontal dimension of each flow section
        
        Shape is (len(eles),)"""
        return (
            np.sqrt(((self.midpoints[0:2,1:] - self.ele_xys) ** 2).sum(axis=0)) + 
            np.sqrt(((self.midpoints[0:2,:-1] - self.ele_xys) ** 2).sum(axis=0))
        )

    def center_dists(self):
        """Get running total distance along transect for each ele center

        Shape is (len(eles),)"""
        # Start with the legs from the previous midpoint to the centroid
        center_dists = np.sqrt(((self.midpoints[0:2,:-1] - self.ele_xys) ** 2).sum(axis=0))
        # Now add the legs from the previous centroid to the previous
        # midpoint (for all elements after the first one)
        center_dists[1:] += np.sqrt(((self.midpoints[0:2,1:-1] - self.ele_xys[:,:-1]) ** 2).sum(axis=0))
        # Create a running total
        return np.cumsum(center_dists)
