from dataclasses import dataclass

import numpy as np

from fvcom.grid import FvcomGrid

# TODO this class design would need a lot of work to be generalized
@dataclass(frozen=True)
class DepthCoordinate:
    z: np.array
    grid: FvcomGrid

    @staticmethod
    def from_asym_sigma(kb, grid, p_sigma=1):
        """Generate a P-Sigma coordinate with thickest layers at the bottom"""

        if kb % 2 != 1:
            raise ValueError("kb must be an odd integer")

        k = np.arange(kb)
        z = -np.power(k/(kb-1), p_sigma)

        return DepthCoordinate(z, grid)

    @staticmethod
    def from_output(dataset, grid=None):
        if grid is None:
            grid = FvcomGrid.from_output(dataset, calc=False)
        dataset['siglev'].set_auto_mask(False)
        return DepthCoordinate(dataset['siglev'][:], grid)

    @property
    def kb(self):
        """Size of the vertical coordinate. Number of depth layers is one less"""
        return len(self.z)

    @property
    def dz(self):
        """Delta between vertical levels"""
        return self.z[:-1] - self.z[1:]

    @property
    def zz(self):
        """The intra-vertical levels. Corresponds to 'siglay' in an output"""
        zz = np.zeros(self.kb)
        zz[:-1] = .5 * (self.z[:-1] + self.z[1:])
        zz[-1] = 2 * zz[-2] - zz[-3]
        return zz

    @property
    def dzz(self):
        """Delta between intra-vertical levels"""
        zz = self.zz
        return zz[:-2] - zz[1:-1]
