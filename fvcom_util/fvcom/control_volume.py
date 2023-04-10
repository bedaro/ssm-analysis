from dataclasses import dataclass, field
from itertools import groupby

import networkx as nx

from .transect import Transect

@dataclass(frozen=True)
class ControlVolume:
    transects: list[Transect]

    def __post_init__(self):
        # Check that all transects were made from the same Grid
        grids = [t.grid for t in self.transects]
        g = groupby(grids)
        assert next(g, True) and not next(g, False), "Grids don't match"
        self._calc_nodes()

    @property
    def grid(self):
        """The grid"""
        return self.transects[0].grid

    @property
    def nodes(self):
        """The nodes property."""
        return self._nodes

    def _calc_nodes(self):
        border_nodes = [t.get_nodes() for t in self.transects]
        # Get the grid's node adjacency, then remove all connections between
        # upstream and downstream nodes on all transects. If caclulations were
        # done correctly, this will break up the graph into separate
        # components, one of which will be our control volume
        adj_dict = self.grid.node_neis()
        for (upstream_nodes, downstream_nodes) in border_nodes:
            for un in upstream_nodes:
                adj_dict[un] -= downstream_nodes
            for dn in downstream_nodes:
                adj_dict[dn] -= upstream_nodes

        g = nx.Graph(adj_dict)

        # Use the Graph to find which component is our control volume, based
        # on all sections having nodes within in
        if len(self.transects) == 1:
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
        object.__setattr__(self, "_nodes",
                nx.node_connected_component(g, node))

    def transect_directions(self):
        """List of bools; True if upstream from transect is in CV"""
        cv_node = next(iter(self.nodes))
        directions = []
        for t in self.transects:
            up, down = t.get_nodes()
            directions.append(next(iter(up)) in self._nodes)
        return directions
