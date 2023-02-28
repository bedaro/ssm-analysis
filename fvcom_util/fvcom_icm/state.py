from dataclasses import dataclass

import numpy as np
from scipy.io import FortranFile

from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate

@dataclass
class FvcomicmState:
    grid: FvcomGrid
    dcoord: DepthCoordinate
    title: list # Can't specify type until Python 3.10 I think
    c2: np.array # grid.m x kbm1 x ncp
    sed: np.array = None
    bbm: np.array = None
    sav: np.array = None
    dfeedm1s: np.array = None
    sfeedm1s: np.array = None

    @staticmethod
    def from_restart_file(restart_file, grid, dcoord, ncp=34, sediment=False,
            balg=False, sav=False, dfeeder=False, sfeeder=False,
            prectype=np.float64):
        with FortranFile(restart_file, 'r') as f:
            title_raw = bytes(f.read_ints('i1')).decode()
            title = [str.rstrip(title_raw[72*i:72*(i+1)]) for i in range(6)]
            
            c2 = f.read_reals(prectype).reshape(
                    grid.m, dcoord.kb-1, ncp, order='F')

            if sediment:
                sed = f.read_reals(prectype)
                if balg:
                    bbm = f.read_reals(prectype)
                else:
                    bbm = None
            else:
                sed = None
                bbm = None
            if sav:
                savdata = f.read_reals(prectype)
            else:
                savdata = None
            if dfeeder:
                dfeedm1s = f.read_reals(prectype)
            else:
                dfeedm1s = None
            if sfeeder:
                sfeedm1s = f.read_reals(prectype)
            else:
                sfeedm1s = None

        return FvcomicmState(grid, dcoord, title, c2, sed, bbm, savdata,
                dfeedm1s, sfeedm1s)

    @staticmethod
    def from_icfile(icfile, grid, dcoord, ncp=34, sediment=False,
            balg=False, sav=False, dfeeder=False, sfeeder=False):
        with open(icfile) as f:
            title = f.readline()
            # Read C2
            c2 = np.zeros((grid.m, dcoord.kb-1, ncp))
            for i in range(grid.m):
                for k in range(dcoord.kb-1):
                    j = 0
                    while j < ncp:
                        chunk = [float(c) for c in f.readline().split()]
                        l = len(chunk)
                        c2[i,k,j:j+l] = chunk
                        j += l
            # TODO handle sediment, etc
        return FvcomicmState(grid, dcoord, [title], c2)

    @property
    def ncp(self):
        return self.c2.shape[2]

    def to_icfile(self, out_file):
        max_per_line = 10
        with open(out_file,'w') as f:
            print(self.title[0], file=f)
            for i in range(self.grid.m):
                for k in range(self.dcoord.kb-1):
                    j = 0
                    while j < self.ncp:
                        data = self.c2[i,k,j:min(j+10, self.ncp)]
                        print(' '.join(['{:12.4f}']*len(data)).format(*data),
                                file=f)
                        j += len(data)

            if self.sed is not None:
                # TODO
                if self.bbm is not None:
                    # TODO
                    pass

            if self.sav is not None:
                # TODO
                pass

            if self.dfeedm1s is not None:
                pass

            if self.sfeedm1s is not None:
                pass
