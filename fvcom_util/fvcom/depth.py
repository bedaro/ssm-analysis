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

    def to_nc(self, ds):
        """Output depth coordinate data to NetCDF. Pass Dataset opened in write mode"""
        siglayDim = ds.createDimension('siglay', self.kb - 1)
        siglevDim = ds.createDimension('siglev', self.kb)

        siglayVar = ds.createVariable('siglay', 'f4', (siglayDim,))
        siglayVar.long_name = 'Sigma Layers'
        siglayVar.standard_name = 'ocean_sigma_coordinate'
        siglayVar.positive = 'up'
        siglayVar.valid_min = -1
        siglayVar.valid_max = 0
        siglayVar.formula_terms = 'siglay:siglay eta:zeta depth:depth'
        siglayVar[:] = self.zz[:-1]

        siglevVar = ds.createVariable('siglev', 'f4', (siglevDim,))
        siglevVar.long_name = 'Sigma Levels'
        siglevVar.standard_name = 'ocean_sigma_coordinate'
        siglevVar.positive = 'up'
        siglevVar.valid_min = -1
        siglevVar.valid_max = 0
        siglevVar.formula_terms = 'siglev:siglev eta:zeta depth:depth'
        siglevVar[:] = self.z
