#!/usr/bin/env python3

import sys
import os
import tempfile
import shutil
import logging
from dataclasses import dataclass, field
from multiprocessing import Pool
from enum import Flag, auto
from argparse import ArgumentParser, Namespace, FileType
from configparser import ConfigParser

from psutil import cpu_count
import scipy.interpolate as interp
from netCDF4 import Dataset, MFDataset
from pandas import Timestamp, Timedelta
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate
from fvcom.transect import Transect
from fvcom.control_volume import ControlVolume
from ssm_an_util import Progress

domain_nodes_shp = "gis/ssm domain nodes.shp"
masked_nodes_txt = "gis/masked nodes.txt"

logger = logging.getLogger(__name__)

def copy_ncatts(inds, invar, outvar):
    """Copy all attributes from one netCDF variable to another"""
    # some variables are different between files in a MFDataset so we need
    # to access a private attribute to get the right attributes dict
    if isinstance(inds, MFDataset) and '_mastervar' in inds[invar].__dict__:
        outvar.setncatts(inds[invar]._mastervar.__dict__)
    else:
        outvar.setncatts(inds[invar].__dict__)

# TODO would be good to simplify time tracking into a class that holds
# start time, end time, and the available times which can then produce slices
# as needed
def init_output(output_filename, indata, tstart, time_slc, cv, input_vars,
                zeta_per_run=False, **dsargs):
    output = Dataset(output_filename, "w", **dsargs)
    output.model_start = tstart.strftime('%Y-%m-%d %X')
    timeDim = output.createDimension('time', len(indata['time'][time_slc]))
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
    # Time units are changing from seconds to days
    timeVar.unit = 'days after model_start'
    # Iterate over all output variables
    # If an extraction attribute is "all", include depth-related info
    for var, attr in input_vars:
        if attr == InputAttr.ALL:
            if 'siglev' in indata.variables:
                dcoord = DepthCoordinate.from_output(indata)
            else:
                dcoord = DepthCoordinate.from_asym_sigma(cv.grid, p_sigma=1.5)
            dcoord.to_nc(output)
            if 'zeta' in indata.variables and not zeta_per_run:
                v = output.createVariable('zeta', 'f4', ('time','node'))
                copy_ncatts(indata, 'zeta', v)
            break

    return output

def append_output(output_cdf):
    return Dataset(output_cdf, 'a')

def get_var_name(prefix, var, attr=None):
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
    force = 'force_overwrite' in args and args.force_overwrite
    if 'zeta_per_run' in args and args.zeta_per_run:
        out_name = get_var_name(args.outprefix, 'zeta')
        dims = list(indata['zeta'].dimensions)
        if out_name in output.variables:
            if not force:
                raise Exception(f'Output variable {out_name} exists. Use --force-overwrite to use anyway')
            if list(output[out_name].dimensions) != dims:
                raise Exception(f'Output variable {out_name} has wrong dimensions {output[out_name].dimensions}\nbut I think it should have dimensions {dims}. Cannot continue.')
        else:
            v = output.createVariable(out_name, 'f4', dims)
            copy_ncatts(indata, 'zeta', v)
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        dims = list(indata[var if var != 'N2' else 'temp'].dimensions)
        if attr != InputAttr.ALL and 'siglay' in dims:
            dims.remove('siglay')
        # Create any additional dimensions needed
        for d in dims:
            if d not in output.dimensions:
                output.createDimension(d, indata.dimensions[d].size)
        if out_name in output.variables:
            if not force:
                raise Exception(f'Output variable {out_name} exists. Use --force-overwrite to use anyway')
            if list(output[out_name].dimensions) != dims:
                raise Exception(f'Output variable {out_name} has wrong dimensions {output[out_name].dimensions}\nbut I think it should have dimensions {dims}. Cannot continue.')
            # If we get here the variable is already present and looks fine
        else:
            v = output.createVariable(out_name, 'f4', dims)
            if var == 'N2':
                copy_ncatts(indata, 'temp', v)
                v.setncatts({'long_name': 'buoy_freq',
                                  'standard_name': 'buoy_freq',
                                  'units': 'sec-2'})
            else:
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
    parser.add_argument("--tstart", type=Timestamp, default="2014.01.01",
                        help="Specify the start time of the model (default: 2014 or read output)")
    parser.add_argument("--tfrom", type=Timestamp,
                        help="Start extract at this date/time")
    parser.add_argument("--tto", type=Timestamp,
                        help="Stop extract at this date/time")
    parser.add_argument("--invar", dest="input_vars", type=colon_meta,
            action="append",
            help="Extract the values of a different output variable")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.add_argument('-z', '--zeta-per-run', action='store_true',
            help="Extract a separate water elevation for each run")
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
            logger.info("get_node_ids found {0} nodes in {1} shapefiles".format(
                len(cv.nodes), len(args.domain_node_shapefiles)))
        else:
            allsections = ConfigParser()
            allsections.read_file(args.sections_config)
            transects = []
            for sect in args.sections:
                if sect not in allsections:
                    raise ValueError(f'Section {sect} not found in {args.sections_config}')
                waypoints = np.array(allsections[sect]['waypoints'].split(' ')).astype(int)
                tr = Transect.shortest(grid, waypoints)
                transects.append(tr)
            cv = ControlVolume.from_transects(transects, calc=True)
            logger.info(f"control volume defined by {len(transects)} transects has {len(cv.nodes)} nodes")
        if not args.no_masking:
            masked_nodes = np.loadtxt(args.masked_nodes_file).astype(np.int64)
            cv = cv - set(masked_nodes)
            logger.info(f"{len(cv.nodes)} remain after masking")
        # Determine time range
        all_times = indata['time'][:]
        first_time = (all_times >= (args.tfrom - args.tstart).total_seconds()).nonzero()[0][0] if args.tfrom else 0
        last_time = (all_times <= (args.tto - args.tstart).total_seconds()).nonzero()[0][-1] + 1 if args.tto else len(all_times)
        time_slc = slice(first_time, last_time)
        if len(cv.nodes) == 0:
            raise ValueError("No nodes to extract.")

        logger.info("Initializing output file...")
        outdata = init_output(output_cdf, indata, args.tstart, time_slc,
                              cv, args.input_vars, zeta_per_run=args.zeta_per_run)
        outdata['time'][:] = indata['time'][time_slc] / 3600 / 24
    else:
        logger.info("Opening existing output file...")
        outdata = append_output(output_cdf)
        if outdata.model_start:
            args.tstart = Timestamp(outdata.model_start)
        cv = ControlVolume(grid=grid, nodes=set(outdata['node'][:]), calc=True)
        all_times = indata['time'][:] / 3600 / 24
        if outdata['time'][0] not in all_times:
            logger.error(f"Start extraction time {outdata['time'][0]} not present in output file")
            logger.error(f"Output time range is {all_times[0]} - {all_times[-1]}")
            outdata.close()
            indata.close()
            sys.exit(1)
        first_time = (all_times == outdata['time'][0]).nonzero()[0][0]
        if outdata['time'][-1] not in all_times:
            logger.error(f"End extraction time {outdata['time'][-1]} not present in output file")
            logger.error(f"Output time range is {all_times[0]} - {all_times[-1]}")
            outdata.close()
            indata.close()
            sys.exit(1)
        last_time = (all_times == outdata['time'][-1]).nonzero()[0][0] + 1
        time_slc = slice(first_time, last_time)
        args.tfrom = args.tstart + Timedelta(all_times[first_time] * 3600 * 24, 's')
    init_output_vars(outdata, indata, **vars(args))
    logger.debug(f'First time: {first_time}; Last time: {last_time}')

    # Attempts to use the entire MFDataset don't seem to scale well.
    # Instead, I'm resorting to a blocking approach where MFDatasets are
    # created for only a few netCDF files at a time
    indata.close()
    i = 0 # The count of times actually copied
    t = 0 # The next time index to examine
    logger.info("Beginning extraction...")
    prog = Progress(outdata.dimensions['time'].size, force_log=not args.verbose,
                    logger=logger)
    times_ct = outdata.dimensions['time'].size
    for cdfchunk in chunks(exist_cdfs, args.chunk_size):
        # Stop condition: we've gone past args.tto
        if t >= last_time:
            break
        c = MFDataset(cdfchunk) if len(cdfchunk) > 1 else Dataset(cdfchunk[0])
        chunk_times = len(c.dimensions['time'])
        # Skip condition: chunk's times are entirely before args.tfrom
        # Note that t + chunk_times is the stop index and isn't actually copied
        # in this pass
        if t + chunk_times <= first_time:
            c.close()
            prog.skip(args.tfrom.strftime('%Y-%m-%d'))
            t += chunk_times
            continue

        chunk_first = max(first_time - t, 0)
        chunk_last = min(chunk_times, last_time - t)
        data = copy_data(c, outdata, cv, slice(chunk_first, chunk_last),
                         i, **vars(args))
        i += chunk_last - chunk_first
        t += chunk_times
        c.close()
        prog.update(i, np.sum([d.size * d.itemsize for k,d in data.items()]))

    prog.finish()
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

g = 9.81

def calc_maxn2(z, r, debug=False):
    spl = interp.PchipInterpolator(np.flip(z), np.flip(r))
    dpdz = spl.derivative()(z)
    if debug:
        fig, ax = plt.subplots()
        ax.plot(r, z,'.')
        zsp = np.linspace(z[0], z[-1], 100)
        ax.plot(spl(zsp), zsp)
    # dpdz is negative so use min() to get the "largest" value
    # Also rho values are in kg/m3 - 1000 so add 1000 for rho0
    return -g / (1000+z[dpdz.argmin()]) * dpdz.min()

def calc_alln2(z, r, debug=False):
    spl = interp.PchipInterpolator(np.flip(z), np.flip(r))
    dpdz = spl.derivative()(z)
    if debug:
        fig, ax = plt.subplots()
        ax.plot(r, z,'.')
        zsp = np.linspace(z[0], z[-1], 100)
        ax.plot(spl(zsp), zsp)
    return -g / (1000+z) * dpdz

def copy_data(cdfin, cdfout, cv, time_slc, timeidx, **kwargs):
    """Copy data from cdfin to cdfout

    Parameters
    ----------
    cdfin: The input Dataset or MFDataset to copy from
    cdfout: The Dataset to copy to, already initialized with dest variable
    cv: The ControlVolume that represents the spatial range to copy from
    time_slc: A slice object of input time indices to copy from
    timeidx: The destination output starting time index
    Extra arguments:
        outprefix: required; the prefix to add to the variable in cdfout
        input_vars: required; the list of name/attr tuple pairs representing variables to copy
        zeta_per_run: if True, make a copy of zeta from incdf and save it with the outprefix

    Returns a dict containing all data that was copied between files
    """
    if 'siglev' in cdfin.variables:
        dcoord = DepthCoordinate.from_output(cdfin)
        cdfin['siglev'].set_auto_mask(False)
    else:
        dcoord = DepthCoordinate.from_asym_sigma(cv.grid, p_sigma=1.5)
    siglayers = dcoord.dz
    args = Namespace(**kwargs)
    times_ct = len(cdfin['time'][time_slc])
    alldata = {}
    photic_mask = None
    node_ids = np.array(cv.nodes_list)
    element_ids = None
    # Copy zeta if it's needed
    if 'zeta_per_run' in args and args.zeta_per_run:
        zetavar = get_var_name(args.outprefix, 'zeta')
    else:
        zetavar = 'zeta'
    if zetavar in cdfout.variables:
        alldata[zetavar] = cdfin['zeta'][time_slc, node_ids - 1]
        cdfout[zetavar][timeidx:timeidx + times_ct, :] = alldata[zetavar]
    for var, attr in args.input_vars:
        out_name = get_var_name(args.outprefix, var, attr)
        if var in cdfin.variables and 'nele' in cdfin[var].dimensions:
            if element_ids is None:
                element_ids = np.array(cv.elements_list)
            spidx = element_ids - 1
        else:
            spidx = node_ids - 1
        if var == 'N2':
            s_name = get_var_name(args.outprefix, 'salinity')
            t_name = get_var_name(args.outprefix, 'temp')
            if s_name not in alldata:
                alldata[s_name] = cdfin['salinity'][time_slc,:,spidx]
                s = alldata[s_name]
            if t_name not in alldata:
                alldata[t_name] = cdfin['temp'][time_slc,:,spidx]
                t = alldata[t_name]
            # Use FVCOM's DENS2 method to compute density at every cell
            dens = (((s ** 3 * 6.76786136E-6 - s ** 2 * 4.8249614E-4 + s * 8.14876577E-1 - 0.22584586) * 
               (t ** 3 * 1.667E-8 - t ** 2 * 8.164E-7 + t * 1.803E-5) +
                1 - t ** 3 * 1.0843E-6 + t ** 2 * 9.8185E-5 - t * 4.786E-3) *
               (s ** 3 * 6.76786136E-6 - s ** 2 * 4.8249614E-4 + s * 8.14876577E-1 + 3.895414E-2) -
               (t - 3.98) ** 2 * (t + 283) / (503.57 * (t + 67.26)))
            zz = dcoord.zz[:-1] 
            M = len(cv.nodes_list)
            KBm1 = len(zz)
            assert dens.shape == (times_ct, KBm1, M), dens.shape
            dens_reshape = np.swapaxes(dens, 1, 2).reshape((-1,len(zz)))
            assert dens_reshape.shape == (times_ct * M, KBm1), dens_reshape.shape
            z_by_loc = (np.expand_dims(zz, 1) @ np.expand_dims((cdfin['h'][node_ids - 1] + alldata[zetavar]).flatten(), 0)).T
            assert z_by_loc.shape == (times_ct * M, KBm1), z_by_loc.shape
            # merge the two arrays into a single input array
            inp = np.swapaxes([z_by_loc,dens_reshape], 0, 1)
            assert inp.shape == (times_ct * M, 2, KBm1), inp.shape
            with Pool(min(cpu_count(logical=False), len(os.sched_getaffinity(0)))) as p:
                if attr == InputAttr.MAX:
                    data = np.reshape(p.starmap(calc_maxn2, inp), alldata[zetavar].shape)
                else:
                    data = np.reshape(p.starmap(calc_alln2, inp), s.shape)
        elif out_name in alldata:
            data = alldata[out_name]
        elif attr is None:
            data = cdfin[var][time_slc, spidx]
        else:
            if InputAttr.BOTTOM in attr:
                slc = -1
            elif InputAttr.SURFACE in attr:
                slc = 0
            else:
                slc = slice(None)
            data = cdfin[var][time_slc, slc, spidx]
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
