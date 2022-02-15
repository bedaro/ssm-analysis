#!/usr/bin/env python3

import time
import os
import tempfile
import shutil
import logging
from argparse import ArgumentParser, Namespace
from netCDF4 import Dataset, MFDataset
import geopandas as gpd
import numpy as np

domain_nodes_shp = "gis/ssm domain nodes.shp"

def get_node_ids(shp):
    domain_nodes = gpd.read_file(shp)
    return domain_nodes['node_id'].sort_values().to_numpy()

def init_output(output_cdf, times_ct, nodes):
    do_output = Dataset(output_cdf, "w")
    timeDim = do_output.createDimension('time', times_ct)
    nodeDim = do_output.createDimension('node', len(nodes))
    nodeVar = do_output.createVariable('node', "i4", ('node'))
    do_output['node'][:] = nodes
    timeVar = do_output.createVariable('time', "f4", ('time'))
    return do_output

def append_output(output_cdf):
    return Dataset(output_cdf, 'a')

def init_output_var(output, var):
    output.createVariable(var, 'f4', ('time','node'))

# Gotten from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-or-iterable-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    script_home = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Extract data from SSM netcdf output files")
    parser.add_argument("incdf", nargs="+", help="each input CDF file")
    parser.add_argument("outcdf",
            help="the output CDF file (created if it doesn't exist)")
    parser.add_argument("outvar",
            help="the variable to store extracted data in the output CDF")
    parser.add_argument("-d", dest="domain_node_shapefile",
            help="Specify a domain node shapefile")
    parser.add_argument("--invar", dest="input_var",
            help="Extract the values of a different output variable")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.add_argument("-c", "--chunk-size", type=int, dest="chunk_size",
            help="Process this many CDF files at once")
    parser.add_argument("--cache", dest="cache", action="store_true",
            help="Use a read/write cache in a temporary directory")
    parser.set_defaults(chunk_size=4,
            domain_node_shapefile=os.path.join(script_home, domain_nodes_shp),
            input_var="DOXG", verbose=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.cache:
        with tempfile.TemporaryDirectory() as tmpdir:
            exist_cdfs = []
            logging.info("Caching input files...")
            for infile in args.incdf:
                newpath = os.path.join(tmpdir, os.path.basename(infile))
                shutil.copy(infile, newpath)
                exist_cdfs.append(newpath)
            output_cdf = os.path.join(tmpdir, os.path.basename(args.outcdf))
            if os.path.exists(args.outcdf):
                logging.info("Caching output file...")
                shutil.copy(args.outcdf, output_cdf)
            do_extract(exist_cdfs, output_cdf, **vars(args))
            # Copy the resulting output CDF back
            logging.info("Saving output file...")
            shutil.copy(output_cdf, args.outcdf)
            logging.info("Finished.")
    else:
        do_extract(args.incdf, args.outcdf, **vars(args))

def do_extract(exist_cdfs, output_cdf, **kwargs):
    args = Namespace(**kwargs)
    logging.info("Determining scope of work...")
    indata = MFDataset(exist_cdfs) if len(exist_cdfs) > 1 else Dataset(exist_cdfs[0])
    node_ids = get_node_ids(args.domain_node_shapefile)
    times_ct = len(indata.dimensions['time'])
    logging.info("Initializing output file...")
    if not os.path.exists(output_cdf):
        outdata = init_output(output_cdf, times_ct, node_ids)
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        outdata = append_output(output_cdf)
    init_output_var(outdata, args.outvar)

    # Attempts to use the entire MFDataset don't seem to scale well.
    # Instead, I'm resorting to a blocking approach where MFDatasets are
    # created for only a few netCDF files at a time
    indata.close()
    i = 0
    total = 0
    logging.info("Beginning extraction...")
    start_time = time.perf_counter()
    for cdfchunk in chunks(exist_cdfs, args.chunk_size):
        c = MFDataset(cdfchunk) if len(cdfchunk) > 1 else Dataset(cdfchunk[0])
        chunk_times = len(c.dimensions['time'])
        data = c[args.input_var][:, -1, node_ids - 1]
        outdata[args.outvar][i:i+chunk_times,:] = data
        i += chunk_times
        c.close()
        if args.verbose:
            elapsed = (time.perf_counter() - start_time)
            to_go = elapsed * (times_ct / i - 1)
            total += data.size * data.itemsize
            logging.info("{0}/{1} ({2}s elapsed, {3}s to go, {4}KBps)".format(i,
                times_ct, int(elapsed), int(to_go), int(total/elapsed/1000)))
    logging.info("Extraction finished.")
    outdata.close()

if __name__ == "__main__": main()
