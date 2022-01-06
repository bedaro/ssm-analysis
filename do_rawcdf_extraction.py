#!/usr/bin/env python3

import sys
import os
from netCDF4 import Dataset, MFDataset
import geopandas as gpd
import numpy as np

# TODO make these commandline options
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
    exist_cdfs = sys.argv[1:-2]
    output_cdf = sys.argv[-2]
    output_var = sys.argv[-1]

    indata = MFDataset(exist_cdfs)
    node_ids = get_node_ids(domain_nodes_shp)
    if not os.path.exists(output_cdf):
        times_ct = len(indata.dimensions['time'])
        outdata = init_output(output_cdf, times_ct, node_ids)
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        outdata = append_output(output_cdf)
    init_output_var(outdata, output_var)

    outdata[output_var][:] = indata[input_var][:, -1, node_ids - 1]
    indata.close()
    outdata.close()

if __name__ == "__main__": main()
