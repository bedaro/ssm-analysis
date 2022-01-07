#!/usr/bin/env python3

import time
import os
from optparse import OptionParser
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
    parser = OptionParser(usage="%prog [options] incdf1... outcdf outvar")
    parser.add_option("-d", dest="domain_node_shapefile",
            help="Specify a domain node shapefile")
    parser.add_option("--invar", dest="input_var",
            help="Extract the values of a different output variable")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.add_option("-c", "--chunk-size", type="int", dest="chunk_size",
            help="Process this many CDF files at once")
    parser.set_defaults(chunk_size=4,
            domain_node_shapefile=os.path.join(script_home, domain_nodes_shp),
            input_var="DOXG",
            verbose=False)
    (options, args) = parser.parse_args()

    exist_cdfs = args[:-2]
    output_cdf = args[-2]
    output_var = args[-1]

    if options.verbose:
        print("Determining scope of work...", flush=True)
    indata = MFDataset(exist_cdfs)
    node_ids = get_node_ids(options.domain_node_shapefile)
    times_ct = len(indata.dimensions['time'])
    if options.verbose:
        print("Initializing output file...", flush=True)
    if not os.path.exists(output_cdf):
        outdata = init_output(output_cdf, times_ct, node_ids)
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        outdata = append_output(output_cdf)
    init_output_var(outdata, output_var)

    # Attempts to use the entire MFDataset don't seem to scale well.
    # Instead, I'm resorting to a blocking approach where MFDatasets are
    # created for only a few netCDF files at a time
    indata.close()
    i = 0
    total = 0
    if options.verbose:
        print("Beginning extraction...", flush=True)
    start_time = time.perf_counter()
    for cdfchunk in chunks(exist_cdfs, options.chunk_size):
        c = MFDataset(cdfchunk)
        chunk_times = len(c.dimensions['time'])
        data = c[options.input_var][:, -1, node_ids - 1]
        outdata[output_var][i:i+chunk_times,:] = data
        i += chunk_times
        c.close()
        if options.verbose:
            elapsed = (time.perf_counter() - start_time)
            to_go = elapsed * (times_ct / i - 1)
            total += data.size * data.itemsize
            print("{0}/{1} ({2}s elapsed, {3}s to go, {4}KBps)".format(i,
                times_ct, int(elapsed), int(to_go), int(total/elapsed/1000)), flush=True)
    if options.verbose:
        print("Finished.", flush=True)
    outdata.close()

if __name__ == "__main__": main()
