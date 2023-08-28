#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
from multiprocessing import Pool
from functools import partial
import os.path

from netCDF4 import Dataset
import numpy as np

from fvcom.grid import FvcomGrid

logger = logging.getLogger(__name__)

def find_pinvs(grid):
    pinvs = np.zeros((grid.n, 2, 3))
    for ele,(coord,neis) in enumerate(zip(grid.elcoord[0:2].T, grid.nbe.T - 1)):
        neis2 = neis[neis > -1]
        dxy = (grid.elcoord[0:2, neis2].T -
                coord.T)
        for edge in (neis == -1).nonzero()[0]:
            # Find the verticies that bound this edge
            edge_vs = grid.nv[
                    [i for i in range(3) if i != edge],
                    ele]
            # Calculate the midpoint between them
            mid = grid.ncoord[0:2, edge_vs - 1].mean(axis=1)
            # Add it to dxy
            dxy = np.concatenate((dxy, [mid]))
        pinvs[ele] = np.linalg.pinv(dxy)
    return pinvs

def ele_velfield(grid, ele, u, v, pinvs):
    """Get the necessary ele_vort arguments for this element"""
    neis = grid.nbe.T[ele] - 1
    neis2 = neis[neis > -1]
    u0 = u[:,:,ele]
    v0 = v[:,:,ele]
    nei_u = u[:,:,neis2]
    nei_v = v[:,:,neis2]
    du = (nei_u.T - u0.T).T
    dv = (nei_v.T - v0.T).T

    return (ele,neis,pinvs[ele],du,dv,u0,v0)

def ele_vort(ele,neis,pinv,du,dv,u0,v0):
    for edge in (neis == -1).nonzero()[0]:
        # Add a zero velocity condition (corresponding to the edge)
        du = np.concatenate((du, np.expand_dims(-u0, 2)), axis=2)
        dv = np.concatenate((dv, np.expand_dims(-v0, 2)), axis=2)
    du_reshape = du.T.reshape(du.shape[2], (du.shape[0]*du.shape[1])).data
    dv_reshape = dv.T.reshape(dv.shape[2], (dv.shape[0]*dv.shape[1])).data
    # Get the a/b least squares constants
    abu_reshape = pinv @ du_reshape
    abv_reshape = pinv @ dv_reshape
    abu = abu_reshape.reshape(2, du.shape[1], du.shape[0]).T
    abv = abv_reshape.reshape(2, dv.shape[1], dv.shape[0]).T

    return abv[:,:,0] - abu[:,:,1]

def main():
    parser = ArgumentParser(description="Add vorticity to FVCOM output files")
    parser.add_argument("incdf", nargs="+",
            help="each netCDF file")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the calculation")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with Dataset(args.incdf[0]) as ds:
        grid = FvcomGrid.from_output(ds)

    pinvs = find_pinvs(grid)

    DIMS = ('time','siglay','nele','three')
    for f in args.incdf:
        logger.info(f'Processing {os.path.basename(f)}')
        ds = Dataset(f, 'a')
        if 'vort' not in ds.variables:
            var = ds.createVariable('vort','f4',DIMS)
            var.long_name = 'Vorticity'
            var.units = 's-1'
            var.grid = 'fvcom_grid'
            var.type = 'data'
        vorts = np.zeros([ds.dimensions[d].size for d in DIMS])
        u = ds['u'][:]
        v = ds['v'][:]
        with Pool() as p:
            data = [ele_velfield(grid, ele, u, v, pinvs) for ele in range(grid.n)]
            vorts = np.moveaxis(p.starmap(ele_vort, data), 0, 2)
        ds['vort'][:,:,:,2] = vorts
        ds.close()

if __name__ == "__main__": main()
