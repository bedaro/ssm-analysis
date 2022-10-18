#!/usr/bin/env python3

import time
import os
import tempfile
import shutil
import logging
from enum import Enum
from argparse import ArgumentParser, Namespace, FileType
from netCDF4 import Dataset, MFDataset
import geopandas as gpd
import numpy as np

domain_nodes_shp = "gis/ssm domain nodes.shp"
masked_nodes_txt = "gis/masked nodes.txt"

logger = logging.getLogger(__name__)

def get_node_ids(shps, masked):
    merged = None
    for i,shp in enumerate(shps):
        df = gpd.read_file(shp)
        df.set_index('node_id', inplace=True)
        logger.debug("Shapefile {0} has {1} nodes".format(shp, len(df)))
        if merged is None:
            merged = df.index
        else:
            merged = merged.union(df.index)
    logger.debug("get_node_ids found {0} nodes in {1} shapefiles".format(
        len(merged), len(shps)))

    masked_nodes = np.loadtxt(masked)
    merged = merged.difference(masked_nodes)
    logger.debug("{0} nodes left after masking".format(len(merged)))

    return merged.to_numpy()

DEFAULT_SIGLAYERS = [-0.01581139, -0.06053274, -0.12687974, -0.20864949,
                  -0.30326778, -0.40915567, -0.52520996, -0.65060186,
                  -0.78467834, -0.9269075 ]

def init_output(output_cdf, indata, nodes, **kwargs):
    args = Namespace(**kwargs)
    output = Dataset(output_cdf, "w")
    timeDim = output.createDimension('time', len(indata.dimensions['time']))
    nodeDim = output.createDimension('node', len(nodes))
    nodeVar = output.createVariable('node', "i4", ('node',))
    output['node'][:] = nodes
    timeVar = output.createVariable('time', "f4", ('time',))
    # Iterate over all output variables
    # If an extraction attribute is "all":
    # - add the 'siglay' dimension to the output if it's not already present
    # - include the 'siglay' dimension on the output variable
    # - add a 'zeta' output variable
    for var, attr in args.input_vars:
        if attr == InputAttr.ALL:
            siglayers = indata['siglay'][:] if 'siglay' in indata.variables else DEFAULT_SIGLAYERS
            output.createDimension('siglay', len(siglayers))
            output.createVariable('siglay', 'f4', ('siglay',))
            output['siglay'][:] = siglayers
            if 'zeta' in indata.variables:
                output.createVariable('zeta', 'f4', ('time','node'))
            break

    return output

def append_output(output_cdf):
    return Dataset(output_cdf, 'a')

def get_var_name(prefix, var, attr):
    out_name = prefix + var
    if attr != InputAttr.ALL:
        out_name += '_' + list(attr_strings.keys())[
                list(attr_strings.values()).index(attr)]
    return out_name

def init_output_vars(output, **kwargs):
    args = Namespace(**kwargs)
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        dims = ('time','siglay','node') if attr == InputAttr.ALL else ('time','node')
        if out_name in output.variables:
            if not args.force_overwrite:
                raise Exception(f'Output variable {out_name} exists. Use --force-overwrite to use anyway')
            if output[out_name].dimensions != dims:
                raise Exception(f'Output variable {out_name} has wrong dimensions {output[out_name].dimensions}\nbut I think it should have dimensions {dims}. Cannot continue.')
            # If we get here the variable is already present and looks fine
        else:
            output.createVariable(out_name, 'f4', dims)

# Gotten from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-or-iterable-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class InputAttr(Enum):
    ALL = 0
    BOTTOM = 1
    MAX = 2
    MIN = 3
    MEAN = 4
    # TODO add "photic" for the photic zone

attr_strings = {
    "all": InputAttr.ALL,
    "bottom": InputAttr.BOTTOM,
    "min": InputAttr.MIN,
    "max": InputAttr.MAX,
    "mean": InputAttr.MEAN
}

# Expands an input variable argument into a variable name and an attribute
# describing the vertical extraction method.
def colon_meta(string):
    var, attr = string.split(':', 2)
    return (var, attr_strings[attr])

def main():
    script_home = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Extract data from SSM netcdf output files")
    parser.add_argument("incdf", nargs="+", help="each input CDF file")
    parser.add_argument("outcdf",
            help="the output CDF file (created if it doesn't exist)")
    parser.add_argument("outprefix",
            help="a prefix for the extracted variables in the output CDF")
    parser.add_argument("-d", dest="domain_node_shapefiles", action="append",
            help="Specify a domain node shapefile")
    parser.add_argument("-m", dest="masked_nodes_file", type=FileType('r'),
            help="Specify a different masked nodes text file")
    parser.add_argument("--invar", dest="input_vars", type=colon_meta,
            action="append",
            help="Extract the values of a different output variable")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.add_argument("-c", "--chunk-size", type=int, dest="chunk_size",
            help="Process this many CDF files at once")
    parser.add_argument("--cache", dest="cache", action="store_true",
            help="Use a read/write cache in a temporary directory")
    parser.add_argument("--force-overwrite", action="store_true",
            help="Force overwriting of an existing output variable")
    # Cannot include default values of lists here, see
    # https://bugs.python.org/issue16399
    parser.set_defaults(chunk_size=4, verbose=False,
            masked_nodes_file=os.path.join(script_home, masked_nodes_txt))
    args = parser.parse_args()
    # This is the workaround
    if not args.input_vars:
        args.input_vars = [("DOXG",InputAttr.BOTTOM)]
    if not args.domain_node_shapefiles:
        args.domain_node_shapefiles = [os.path.join(script_home, domain_nodes_shp)]

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    #logger.setLevel(logging.DEBUG)

    if args.cache:
        with tempfile.TemporaryDirectory() as tmpdir:
            exist_cdfs = []
            logger.info("Caching input files...")
            for infile in args.incdf:
                newpath = os.path.join(tmpdir, os.path.basename(infile))
                shutil.copy(infile, newpath)
                exist_cdfs.append(newpath)
            output_cdf = os.path.join(tmpdir, os.path.basename(args.outcdf))
            if os.path.exists(args.outcdf):
                logger.info("Caching output file...")
                shutil.copy(args.outcdf, output_cdf)
            do_extract(exist_cdfs, output_cdf, **vars(args))
            # Copy the resulting output CDF back
            logger.info("Saving output file...")
            shutil.copy(output_cdf, args.outcdf)
            logger.info("Finished.")
    else:
        do_extract(args.incdf, args.outcdf, **vars(args))

def do_extract(exist_cdfs, output_cdf, **kwargs):
    args = Namespace(**kwargs)
    logger.info("Determining scope of work...")
    indata = MFDataset(exist_cdfs) if len(exist_cdfs) > 1 else Dataset(exist_cdfs[0])
    node_ids = get_node_ids(args.domain_node_shapefiles, args.masked_nodes_file)
    logger.info("Initializing output file...")
    if not os.path.exists(output_cdf):
        outdata = init_output(output_cdf, indata, node_ids, **vars(args))
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        outdata = append_output(output_cdf)
    init_output_vars(outdata, **vars(args))

    # Attempts to use the entire MFDataset don't seem to scale well.
    # Instead, I'm resorting to a blocking approach where MFDatasets are
    # created for only a few netCDF files at a time
    indata.close()
    i = 0
    total = 0
    logger.info("Beginning extraction...")
    start_time = time.perf_counter()
    times_ct = outdata.dimensions['time'].size
    for cdfchunk in chunks(exist_cdfs, args.chunk_size):
        c = MFDataset(cdfchunk) if len(cdfchunk) > 1 else Dataset(cdfchunk[0])
        chunk_times = len(c.dimensions['time'])

        data = copy_data(c, outdata, i, node_ids, **vars(args))
        i += chunk_times
        c.close()

        elapsed = (time.perf_counter() - start_time)
        to_go = elapsed * (times_ct / i - 1)
        total += np.sum([d.size * d.itemsize for k,d in data.items()])
        logger.info("{0}/{1} ({2}s elapsed, {3}s to go, {4}KBps)".format(i,
            times_ct, int(elapsed), int(to_go), int(total/elapsed/1000)))
    logger.info("Extraction finished.")
    outdata.close()

def copy_data(cdfin, cdfout, timeidx, node_ids, **kwargs):
    args = Namespace(**kwargs)
    times_ct = len(cdfin.dimensions['time'])
    alldata = {}
    # Copy zeta if it's needed
    if 'zeta' in cdfout.variables:
        alldata['zeta'] = cdfin['zeta'][:, node_ids - 1]
        cdfout['zeta'][timeidx:timeidx + times_ct, :] = alldata['zeta']
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        if attr == InputAttr.BOTTOM:
            slc = -1
        else:
            slc = slice(None)
        data = cdfin[var][:, slc, node_ids - 1]
        # TODO add "photic" case which will look rather different
        if attr == InputAttr.MIN:
            data = data.min(axis=1)
        elif attr == InputAttr.MAX:
            data = data.max(axis=1)
        elif attr == InputAttr.MEAN:
            data = data.mean(axis=1)
        logger.debug("data is shape " + str(data.shape))
        if len(cdfout[out_name].dimensions) == 3:
            cdfout[out_name][timeidx:timeidx+times_ct,:,:] = data
        else:
            cdfout[out_name][timeidx:timeidx+times_ct,:] = data
        alldata[out_name] = data
    return alldata

if __name__ == "__main__": main()
