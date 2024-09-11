#!/usr/bin/env python3

import time
import os
import tempfile
import shutil
import logging
from enum import Flag, auto
from argparse import ArgumentParser, Namespace, FileType
from configparser import ConfigParser

from netCDF4 import Dataset, MFDataset
import geopandas as gpd
import numpy as np

from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate
from fvcom.transect import Transect
from fvcom.control_volume import ControlVolume

domain_nodes_shp = "gis/ssm domain nodes.shp"
masked_nodes_txt = "gis/masked nodes.txt"

logger = logging.getLogger(__name__)

def copy_ncatts(inds, invar, outvar):
    # some variables are different between files in a MFDataset so we need
    # to access a private attribute to get the right attributes dict
    if isinstance(inds, MFDataset) and '_mastervar' in inds[invar].__dict__:
        outvar.setncatts(inds[invar]._mastervar.__dict__)
    else:
        outvar.setncatts(inds[invar].__dict__)

def init_output(output_cdf, indata, cv, **kwargs):
    args = Namespace(**kwargs)
    output = Dataset(output_cdf, "w")
    timeDim = output.createDimension('time', len(indata.dimensions['time']))
    node_ids = np.array(cv.nodes_list)
    nodeDim = output.createDimension('node', len(node_ids))
    nodeVar = output.createVariable('node', "i4", ('node',))
    output['node'][:] = node_ids
    for node_v in ('h','x','y'):
        v = output.createVariable(node_v, "f4", ('node',))
        copy_ncatts(indata, node_v, v)
        output[node_v][:] = indata[node_v][node_ids - 1]
    element_ids = np.array(cv.elements_list)
    eleDim = output.createDimension('nele', len(element_ids))
    eleVar = output.createVariable('nele', "i4", ('nele',))
    output['nele'][:] = element_ids
    timeVar = output.createVariable('time', "f4", ('time',))
    copy_ncatts(indata, 'time', timeVar)
    # Iterate over all output variables
    # If an extraction attribute is "all":
    # - add the 'siglay'/'siglev' dimensions to the output if they're not already present
    # - add 'siglay' and 'siglev' output variables
    # - include the 'siglay' dimension on the output variable
    # - add a 'zeta' output variable
    for var, attr in args.input_vars:
        if attr == InputAttr.ALL:
            if 'siglev' in indata.variables:
                dcoord = DepthCoordinate.from_output(indata)
            else:
                dcoord = DepthCoordinate.from_asym_sigma(cv.grid, p_sigma=1.5)
            siglayers = dcoord.zz[0:-1]
            output.createDimension('siglay', len(siglayers))
            v = output.createVariable('siglay', 'f4', ('siglay',))
            if 'siglay' in indata.variables:
                copy_ncatts(indata, 'siglay', v)
            output['siglay'][:] = siglayers
            siglevs = dcoord.z
            output.createDimension('siglev', len(siglevs))
            v = output.createVariable('siglev', 'f4', ('siglev',))
            if 'siglev' in indata.variables:
                copy_ncatts(indata, 'siglev', v)
            output['siglev'][:] = siglevs
            if 'zeta' in indata.variables:
                v = output.createVariable('zeta', 'f4', ('time','node'))
                copy_ncatts(indata, 'zeta', v)
            break

    return output

def append_output(output_cdf):
    return Dataset(output_cdf, 'a')

def get_var_name(prefix, var, attr):
    out_name = prefix + var
    if attr is not None and InputAttr.ALL not in attr:
        # iterating over a Flag isn't supported until Python 3.11. So
        # to make the corresponding list we need to do an O(n) search of
        # the InputAttr class
        attrl = [list(attr_strings.keys())[list(attr_strings.values()).index(a)] for a in InputAttr if a in attr]
        out_name += '_' + ','.join(attrl)
    return out_name

def init_output_vars(output, indata, **kwargs):
    args = Namespace(**kwargs)
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        dims = list(indata[var].dimensions)
        if attr != InputAttr.ALL and 'siglay' in dims:
            dims.remove('siglay')
        # Create any additional dimensions needed
        for d in dims:
            if d not in output.dimensions:
                output.createDimension(d, indata.dimensions[d].size)
        if out_name in output.variables:
            if not args.force_overwrite:
                raise Exception(f'Output variable {out_name} exists. Use --force-overwrite to use anyway')
            if list(output[out_name].dimensions) != dims:
                raise Exception(f'Output variable {out_name} has wrong dimensions {output[out_name].dimensions}\nbut I think it should have dimensions {dims}. Cannot continue.')
            # If we get here the variable is already present and looks fine
        else:
            v = output.createVariable(out_name, 'f4', dims)
            copy_ncatts(indata, var, v)

# Gotten from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-or-iterable-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class InputAttr(Flag):
    ALL = auto()
    BOTTOM = auto()
    SURFACE = auto()
    MAX = auto()
    MIN = auto()
    MEAN = auto()
    PHOTIC = auto()

attr_strings = {
    "all": InputAttr.ALL,
    "bottom": InputAttr.BOTTOM,
    "surf": InputAttr.SURFACE,
    "min": InputAttr.MIN,
    "max": InputAttr.MAX,
    "mean": InputAttr.MEAN,
    "photic": InputAttr.PHOTIC
}

# Expands an input variable argument into a variable name and an attribute
# describing the vertical extraction method.
def colon_meta(string):
    args = string.split(':', 2)
    if len(args) > 1:
        var, attrs = args
        attr = np.bitwise_or.reduce([attr_strings[attr] for attr in attrs.split(',')])
    else:
        var = args[0]
        attr = None
    return (var, attr)

def main():
    script_home = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Extract data from SSM netcdf output files")
    parser.add_argument("incdf", nargs="+", help="each input CDF file")
    parser.add_argument("outcdf",
            help="the output CDF file (created if it doesn't exist)")
    parser.add_argument("outprefix",
            help="a prefix for the extracted variables in the output CDF")
    parser.add_argument("-d", dest="domain_node_shapefiles", nargs='*',
            help="Specify a domain node shapefile")
    parser.add_argument('--sections', '-s', nargs='*',
            help='Specify sections bounding a control volume for the domain')
    parser.add_argument('--sections-config', type=FileType('r'),
            help='Specify a section config file')
    parser.add_argument("-m", dest="masked_nodes_file", type=FileType('r'),
            help="Specify a different masked nodes text file")
    parser.add_argument("--no-masking", action="store_true",
                        help="Don't filter out masked nodes")
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
            masked_nodes_file=os.path.join(script_home, masked_nodes_txt),
            sections_config=os.path.join(script_home, 'SSM_Grid', 'sections.ini'))
    args = parser.parse_args()
    # This is the workaround
    if not args.input_vars:
        args.input_vars = [("DOXG",InputAttr.BOTTOM)]
    if not args.domain_node_shapefiles and not args.sections:
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
    indata = MFDataset(exist_cdfs) if len(exist_cdfs) > 1 else Dataset(exist_cdfs[0])
    grid = FvcomGrid.from_output(indata)
    cv = None
    if not os.path.exists(output_cdf):
        logger.info("Determining scope of work...")
        if args.domain_node_shapefiles:
            for shp in args.domain_node_shapefiles:
                df = gpd.read_file(shp).set_index('node_id')
                nodes = set(df.index)
                if cv is None:
                    cv = ControlVolume(grid=grid, nodes=nodes, calc=True)
                else:
                    cv = cv + nodes
            logger.debug("get_node_ids found {0} nodes in {1} shapefiles".format(
                len(cv.nodes), len(args.domain_node_shapefiles)))
        else:
            allsections = ConfigParser()
            allsections.read_file(args.sections_config)
            transects = []
            for sect in args.sections:
                waypoints = np.array(allsections[sect]['waypoints'].split(' ')).astype(int)
                tr = Transect.shortest(grid, waypoints)
                transects.append(tr)
            cv = ControlVolume.from_transects(transects, calc=True)
            logger.debug(f"control volume defined by {len(transects)} transects has {len(cv.nodes)} nodes")
        if not args.no_masking:
            masked_nodes = np.loadtxt(args.masked_nodes_file).astype(np.int64)
            cv = cv - set(masked_nodes)
            logger.debug(f"{len(cv.nodes)} remain after masking")

        logger.info("Initializing output file...")
        outdata = init_output(output_cdf, indata, cv, **vars(args))
        outdata['time'][:] = indata['time'][:] / 3600 / 24
    else:
        logger.info("Opening existing output file...")
        outdata = append_output(output_cdf)
        cv = ControlVolume(grid=grid, nodes=set(outdata['node'][:]), calc=True)
    init_output_vars(outdata, indata, **vars(args))

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

        data = copy_data(c, outdata, i, cv, **vars(args))
        i += chunk_times
        c.close()

        elapsed = (time.perf_counter() - start_time)
        to_go = elapsed * (times_ct / i - 1)
        total += np.sum([d.size * d.itemsize for k,d in data.items()])
        logger.info("{0}/{1} ({2}s elapsed, {3}s to go, {4}KBps)".format(i,
            times_ct, int(elapsed), int(to_go), int(total/elapsed/1000)))
    logger.info("Extraction finished.")
    outdata.close()

def get_photic_mask(cdfin, node_ids):
    times_per_day = int(86400 / (cdfin['time'][1] - cdfin['time'][0]))
    # Compute a moving average for 24 hours
    cs = cdfin['IAVG'][:,:,node_ids - 1].cumsum(axis=0, dtype=float)
    cs[times_per_day:] -= cs[:-times_per_day]
    avglight = np.zeros_like(cs)
    for n in range(1, times_per_day):
        avglight[n-1,:] = cs[n - 1,:] / n
    avglight[times_per_day - 1:,:] = cs[times_per_day - 1:,:] / times_per_day

    # Mask all locations with no light (to catch when surface light is zero)
    # or light at depth is less than 1% of surface
    mask = (avglight[:] == 0) | (avglight[:] < avglight[:,[0],:] * 0.01)
    return mask

def copy_data(cdfin, cdfout, timeidx, cv, **kwargs):
    if 'siglev' in cdfin.variables:
        dcoord = DepthCoordinate.from_output(cdfin)
        cdfin['siglev'].set_auto_mask(False)
        #siglayers = cdfin['siglev'][:-1] - cdfin['siglev'][1:]
    else:
        dcoord = DepthCoordinate.from_asym_sigma(FvcomGrid.from_output(cdfin, calc=False), p_sigma=1.5)
    siglayers = dcoord.dz
    args = Namespace(**kwargs)
    times_ct = len(cdfin.dimensions['time'])
    alldata = {}
    photic_mask = None
    node_ids = np.array(cv.nodes_list)
    element_ids = None
    # Copy zeta if it's needed
    if 'zeta' in cdfout.variables:
        alldata['zeta'] = cdfin['zeta'][:, node_ids - 1]
        cdfout['zeta'][timeidx:timeidx + times_ct, :] = alldata['zeta']
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        if 'nele' in cdfin[var].dimensions:
            if element_ids is None:
                element_ids = np.array(cv.elements_list)
            spidx = element_ids - 1
        else:
            spidx = node_ids - 1
        if attr is None:
            data = cdfin[var][:, spidx]
        else:
            if InputAttr.BOTTOM in attr:
                slc = -1
            elif InputAttr.SURFACE in attr:
                slc = 0
            else:
                slc = slice(None)
            data = cdfin[var][:, slc, spidx]
            if InputAttr.PHOTIC in attr:
                if photic_mask is None:
                    photic_mask = get_photic_mask(cdfin, node_ids)
                data = np.ma.masked_array(data, mask=photic_mask)
            if InputAttr.MIN in attr:
                data = data.min(axis=1)
            elif InputAttr.MAX in attr:
                data = data.max(axis=1)
            elif InputAttr.MEAN in attr:
                data = np.ma.average(data, axis=1, weights=siglayers)
        logger.debug("data is shape " + str(data.shape))
        if len(cdfout[out_name].dimensions) == 3:
            cdfout[out_name][timeidx:timeidx+times_ct,:,:] = data
        else:
            cdfout[out_name][timeidx:timeidx+times_ct,:] = data
        alldata[out_name] = data
    return alldata

if __name__ == "__main__": main()
