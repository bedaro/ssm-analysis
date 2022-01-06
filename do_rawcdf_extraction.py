#!/usr/bin/env python3

import time
import os
from optparse import OptionParser
from netCDF4 import Dataset, MFDataset
import geopandas as gpd
import numpy as np

domain_nodes_shp = "gis/ssm domain nodes.shp"
input_var = "DOXG"

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

def main():
    script_home = os.path.dirname(os.path.realpath(__file__))
    parser = OptionParser(usage="%prog [options] incdf1... outcdf outvar")
    parser.add_option("-d", dest="domain_node_shapefile",
            help="Specify a domain node shapefile")
    parser.add_option("--invar", dest="input_var",
            help="Extract the values of a different output variable")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.set_defaults(domain_node_shapefile=os.path.join(script_home, domain_nodes_shp),
            input_var="DOXG",
            verbose=False)
    (options, args) = parser.parse_args()

    exist_cdfs = args[:-2]
    output_cdf = args[-2]
    output_var = args[-1]

    indata = MFDataset(exist_cdfs)
    node_ids = get_node_ids(options.domain_node_shapefile)
    times_ct = len(indata.dimensions['time'])
    if not os.path.exists(output_cdf):
        outdata = init_output(output_cdf, times_ct, node_ids)
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        outdata = append_output(output_cdf)
    init_output_var(outdata, output_var)

    start_time = time.perf_counter()
    for i in range(0,times_ct,60):
        stop = min(times_ct,i+60)
        outdata[output_var][i:stop,:] = indata[options.input_var][i:stop, -1, node_ids - 1]
        if options.verbose:
            elapsed = (time.perf_counter() - start_time)
            to_go = elapsed * (times_ct / stop - 1)
            print("{0}/{1} ({2}s elapsed, {3}s to go)".format(stop,
                times_ct, int(elapsed), int(to_go)))
    if options.verbose:
        print("Finished.")
    indata.close()
    outdata.close()

if __name__ == "__main__": main()
