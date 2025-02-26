from dataclasses import dataclass
import geopandas as gpd

from fvcom import FvcomGrid

@dataclass(frozen=True)
class Thalweg:
    """Representation of an along-channel thalweg"""

    grid: FvcomGrid
    nodes: list

    def __post_init__(self):
        # TODO ensure all passed nodes are in adjacent sequence
        pass

    def to_gdf(self):
        """Get a GeoDataFrame of nodes including running distance"""
        gdf = self.grid.nodes_gdf().loc[self.nodes].copy()
        # Calculate the distance between adjacent nodes in the thalweg
        gdf['distance'] = gdf['geometry'][1:].distance(gdf['geometry'][:-1], align=False)
        # Set the first one to zero since it's currently NaN
        gdf.loc[gdf.index == gdf.index[0], 'distance'] = 0
        # Transform into a running total in km
        gdf['distance'] = gdf['distance'].cumsum() / 1000

        return gdf
