#!/usr/bin/env python3

from argparse import ArgumentParser
import os.path
import glob

from netCDF4 import Dataset

from fvcom.grid import FvcomGrid
from fvcom.transect import Transect
import extract_sections as es

def main():
    parser = ArgumentParser(description="Generate plots of sections")
    parser.add_argument("incdf", help="One model output NetCDF file")
    parser.add_argument("outdir", help="The run name/directory")
    parser.add_argument("sections", nargs="*", help="Section name(s) (default all)")
    args = parser.parse_args()

    with Dataset(args.incdf) as indata:
        grid = FvcomGrid.from_output(indata)
        transects = {}
        if args.sections:
            files = [es.EXTRACTION_OUT_PATH.format(outdir=args.outdir, name=s) for s in args.sections]
        else:
            files = glob.glob(es.EXTRACTION_OUT_PATH.format(outdir=args.outdir, name="*"))
        for f in files:
            # get basename and remove .nc extension
            name = os.path.basename(f)[:-3]
            with Dataset(f) as ds:
                transect = Transect(grid, ds['ele'][:])
            transects[name] = transect

        es.make_plots(indata, transects, args.outdir)

if __name__ == "__main__": main()
