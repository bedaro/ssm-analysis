#!/usr/bin/env python3
"""
Extracts fields at a number of sections which may be used later for TEF
analysis of transport and transport-weighted properties.

This script is based heavily on Parker MacCready's x_tef/extract_sections.py
for LiveOcean and has many of the same options. The plot output is based, in
part, on Parker's x_tef/plot_physical_section.py

TODO
- A --cache option like rawcdf_extract so files are read/written in TMPDIR
"""

import time
from datetime import datetime
import os
import logging
import requests
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from multiprocessing import Pool
from functools import partial

from netCDF4 import Dataset, MFDataset
import numpy as np
import networkx as nx
from shapely.geometry import LineString
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
from matplotlib import ticker
import contextily as cx
from fvcom import FvcomGrid

root_logger = logging.getLogger()

# Gotten from https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    global MAX_JOBS
    parser = ArgumentParser(description="Extract sections of transport and tracer values.")
    parser.add_argument("incdf", nargs="+", help="each input NetCDF file")
    parser.add_argument("sectionsfile",
            help="the INI file describing the sections to extract")
    parser.add_argument("outdir",
            help="the output directory for NetCDF files")
    parser.add_argument("-c", "--chunk-size", type=int, dest="chunk_size",
            help="Process this many model NetCDF files at once")
    parser.add_argument("-p", "--make-plots", action="store_true",
            dest="make_plots", help="Generate plots of the transects")
    parser.add_argument("-u", "--uniform-velocity", action="store_true",
            dest="uniform_velocity",
            help="Assume spatially uniform velocity within each element")
    parser.add_argument("-x", "--ex_name", type=str,
            help="Experiment name")
    parser.add_argument("-d", "--output-start-date", type=str,
            help="Date corresponding to time 0")
    parser.add_argument("-j", "--max-jobs", type=int,
            help="Maximum number of parallel jobs")
    parser.add_argument("-0", "--date-string0", type=str,
            help="Date to begin extraction")
    parser.add_argument("-1", "--date-string1", type=str,
            help="Date for the end of extraction")
    parser.add_argument("-v", "--verbose", action="store_true",
            help="Print progress messages during the extraction")
    # TODO implement caching option for performance on Klone

    parser.set_defaults(chunk_size=4, max_jobs=len(os.sched_getaffinity(0)),
            date_string0='none', date_string1='none', ex_name='untitled',
            output_start_date='2014.01.01', verbose=False, make_plots=False,
            uniform_velocity=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    config = ConfigParser()
    config.read(args.sectionsfile)

    root = EXTRACTION_ROOT_DIR.format(outdir=args.outdir)
    if not os.path.isdir(root):
        os.mkdir(root)

    MAX_JOBS = args.max_jobs

    do_extract(args.incdf, config, args.outdir, **vars(args))

def sect_linestr(name, sect):
    """Build a LineString object from the X and Y values in sect"""
    return [name, LineString([(x, y) for x, y in zip(sect['x'], sect['y'])])]

EXTRACTION_ROOT_DIR = 'LiveOcean_output/tef/{outdir}'
EXTRACTION_OUT_PATH = EXTRACTION_ROOT_DIR + '/extractions/{name}.nc'
PLOT_ROOT_DIR = EXTRACTION_ROOT_DIR + '/physical_section_plots'
PLOT_PATH = PLOT_ROOT_DIR + '/{name}/{plot}.png'

def do_extract(exist_cdfs, sects_config, output_dir, **kwargs):
    logger = root_logger.getChild('do_extract')
    args = Namespace(**kwargs)
    indata = MFDataset(exist_cdfs) if len(exist_cdfs) > 1 else Dataset(exist_cdfs[0])
    # Suppress warnings
    indata['siglay'].set_auto_mask(False)
    indata['siglev'].set_auto_mask(False)

    logger.info("Building sections")
    sect_start = time.perf_counter()
    sections = find_sections(indata, sects_config)
    elapsed = (time.perf_counter() - sect_start)
    total_eles = np.sum([len(s["eles"]) for n,s in sections.items()])
    logger.info(f'Found {len(sections)} sections, {total_eles} elements to extract in {int(elapsed)} secs')

    logger.info("Determining scope of work")
    sd = datetime.strptime(args.output_start_date, '%Y.%m.%d')
    model_dates = pd.Timestamp(sd) + pd.to_timedelta(indata['time'][:], 's')
    model_date_df = pd.DataFrame({"intime": indata['time'][:] }, index=model_dates)
    # Convert these times to UNIX timestamps to match Parker's convention
    # See https://stackoverflow.com/a/40881958
    model_date_df['ocean_time'] = model_dates.values.astype(np.int64) / 10 ** 9
    if args.date_string0 != 'none':
        dt0 = datetime.strptime(args.date_string0, '%Y.%m.%d')
    else:
        dt0 = sd
    if args.date_string1 != 'none':
        dt1 = datetime.strptime(args.date_string1, '%Y.%m.%d')
    else:
        dt1 = model_dates[-1].to_pydatetime()
    meta = {
            'date_string0': dt0.strftime('%Y.%m.%d'),
            'date_string1': dt1.strftime('%Y.%m.%d'),
            'gtagex': args.ex_name
    }
    # Get the indices of the time variable from the model output to extract
    time_range = model_date_df.loc[(model_date_df.index >= dt0) &
            (model_date_df.index <= dt1), ['intime','ocean_time']]
    logger.info(f'Need to extract data for {len(time_range)} times')

    logger.info("Creating NetCDF output files")
    # Need to transform UTM coordinates of section points into lat/lon
    # to fit Parker's existing workflow
    with Pool(MAX_JOBS) as p:
        data = p.starmap(sect_linestr, sections.items())
    # Build the GDF using a UTM Zone 10 CRS and reproject it to lat/lon
    section_eles_gdf = gpd.GeoDataFrame(data, columns=("name", "geometry"),
            crs='epsg:32610').set_index('name').to_crs('epsg:4326')

    for name,section in sections.items():
        meta['section_name'] = name
        out_fn = EXTRACTION_OUT_PATH.format(outdir=output_dir, name=name)
        out_dir = os.path.dirname(out_fn)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        latlons = np.array(section_eles_gdf.loc[name, "geometry"].coords)
        outds = start_netcdf(indata, out_fn, len(time_range),
                section, indata.dimensions['siglay'].size,
                latlons[:,0], latlons[:,1], meta)
        # Populate simple fields
        outds['ocean_time'][:] = time_range['ocean_time']
        outds['h'][:] = section['h']
        outds['z0'][:] = section['z0']
        outds['DA0'][:] = section['da0']

        outds.close()

    # Attempts to use the entire MFDataset don't seem to scale well.
    # Instead, I'm resorting to a blocking approach where MFDatasets are
    # created for only a few netCDF files at a time
    i = 0
    total = 0
    logger.info("Extracting sections...")
    start_time = time.perf_counter()
    times_ct = len(time_range)
    for cdfchunk in chunks(exist_cdfs, args.chunk_size):
        c = MFDataset(cdfchunk) if len(cdfchunk) > 1 else Dataset(cdfchunk[0])
        times_available = time_range.loc[
                (time_range['intime'] >= c['time'][0]) &
                (time_range['intime'] <= c['time'][-1])]
        chunk_times = len(times_available)
        if chunk_times == 0:
            # Nothing to extract from this chunk
            continue
        c.close()

        data_size = 0
        with Pool(MAX_JOBS) as p:
            partial_fn = partial(copy_data, infiles=cdfchunk,
                    time_range=times_available, output_dir=output_dir,
                    uniform=args.uniform_velocity)
            data_size = np.sum(p.starmap(partial_fn, sections.items()))

        i += chunk_times

        elapsed = (time.perf_counter() - start_time)
        to_go = elapsed * (times_ct / i - 1)
        total += data_size
        logger.info("{0}/{1} ({2}s elapsed, {3}s to go, {4}KBps)".format(i,
            times_ct, int(elapsed), int(to_go), int(total/elapsed/1000)))

    if args.make_plots:
        logger.info("Generating plots")
        start_time = time.perf_counter()
        make_plots(indata, sections, output_dir)
        elapsed = time.perf_counter() - start_time
        logger.info(f'Plot generation completed in {elapsed:.1f} secs')
    indata.close()


def copy_data(name, section, infiles, time_range, output_dir, uniform=False):
    indata = MFDataset(infiles) if len(infiles) > 1 else Dataset(infiles[0])
    bytes_written = 0

    times_in = time_range['intime']
    times_out = time_range['ocean_time']
    tin_slc = slice(
            (indata['time'][:] == times_in[0]).nonzero()[0][0],
            (indata['time'][:] == times_in[-1]).nonzero()[0][0]+1)

    out_fn = EXTRACTION_OUT_PATH.format(outdir=output_dir, name=name)
    outds = Dataset(out_fn, 'a')

    tout_slc = slice(
            (outds['ocean_time'][:] == times_out[0]).nonzero()[0][0],
            (outds['ocean_time'][:] == times_out[-1]).nonzero()[0][0]+1)

    # We are going to need most node scalars at least twice, so pre-cache
    # them for all the elements in this section
    node_scalar_cache = {}
    all_sect_nodes = np.unique(np.array([
            indata['nv'][:,ele-1] for ele in section['eles']
        ])).astype(int)
    for n in all_sect_nodes:
        node_scalar_cache[n] = {
                'salt': indata['salinity'][tin_slc,:,n-1],
                'zeta': indata['zeta'][tin_slc,n-1]
        }

    for i,(ele,neis,x,y,n1,n2,mx1,my1,mx2,my2,pinv,da0) in enumerate(zip(
            section['eles'], section['neis'], section['x'], section['y'],
            section['n1'], section['n2'], section['xm'][:-1],
            section['ym'][:-1], section['xm'][1:], section['ym'][1:],
            section['xypinv'], section['da0'].T)):
        # Average scalar values from the nodes belonging to this element
        mynodes = indata['nv'][:,ele-1].astype(int)
        salt = np.mean(
                [node_scalar_cache[node]['salt'] for node in mynodes],
                axis=0)
        outds['salt'][tout_slc,:,i] = salt
        bytes_written += salt.size * salt.itemsize
        zeta = np.mean(
                [node_scalar_cache[node]['zeta'] for node in mynodes],
                axis=0)
        outds['zeta'][tout_slc,i] = zeta
        bytes_written += zeta.size * zeta.itemsize

        # Calculate u dot n (transport flux in m/s)
        u0 = indata['u'][tin_slc,:,ele-1]
        v0 = indata['v'][tin_slc,:,ele-1]
        dx1 = mx1 - x
        dy1 = my1 - y
        l1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
        dx2 = mx2 - x
        dy2 = my2 - y
        l2 = np.sqrt(dx2 ** 2 + dy2 ** 2)
        l = l1 + l2
        if uniform:
            # Uniform velocity model within the element, similar to Conroy
            tf = l1/l * (u0 * n1[0] + v0 * n1[1]) + l2/l * (u0 * n2[0] + v0 * n2[1])
        else:
            # X/Y nonuniform velocity model per the FVCOM manual
            nei_u = indata['u'][tin_slc,:,neis]
            nei_v = indata['v'][tin_slc,:,neis]
            du = (nei_u.T - u0.T).T
            dv = (nei_v.T - v0.T).T
            # This makes errors worse
            #if len(neis) == 2:
                # Add a zero velocity condition (corresponding to the edge)
            #    du = np.concatenate((du, np.expand_dims(-u0, 2)), axis=2)
            #    dv = np.concatenate((dv, np.expand_dims(-v0, 2)), axis=2)
            # First, reshape du and dv to semi-flatten them
            du_reshape = du.T.reshape(du.shape[2], (du.shape[0]*du.shape[1])).data
            dv_reshape = dv.T.reshape(dv.shape[2], (dv.shape[0]*dv.shape[1])).data
            # Get the a/b least squares constants
            abu_reshape = pinv @ du_reshape
            abv_reshape = pinv @ dv_reshape
            abu = abu_reshape.reshape(2, du.shape[1], du.shape[0]).T
            abv = abv_reshape.reshape(2, dv.shape[1], dv.shape[0]).T
            au = abu[:,:,0]
            bu = abu[:,:,1]
            av = abv[:,:,0]
            bv = abv[:,:,1]

            # Perform the calculation (see VelocityExtraction notebook for the
            # derivation)
            tf = (l1 * (n1[0] * (u0 + 0.5 * (au * dx1 + bu * dy1))
                        + n1[1] * (v0 + 0.5 * (av * dx1 + bv * dy1))
                       ) + l2 * (n2[0] * (u0 + 0.5 * (au * dx2 + bu * dy2))
                                 + n2[1] * (v0 + 0.5 * (av * dx2 + bv * dy2))
                                )) / l

        # Calculate q = tf times DA0
        # Final shape is (t, siglay)
        q = tf * da0
        outds['q'][tout_slc,:,i] = q
        bytes_written += q.size * q.itemsize

    outds.close()
    del node_scalar_cache

    return bytes_written

def find_sections(indata, sects_config):
    # Get basic grid data and compute what's missing
    z = indata['siglev'][:]
    zz = indata['siglay'][:]
    grid = FvcomGrid(np.array([indata['x'][:],indata['y'][:],indata['h'][:]]),
            indata['nv'][:].astype(int), calc=True)

    # Build a 2-D Graph of all the grid elements, indexed from 1
    # (the Graph gets constructed in each thread from the dict because
    # NetworkX Graphs aren't thread-safe)
    adj_dict = dict(enumerate(grid.nbe.swapaxes(1,0), 1))

    build_section_nameconf = partial(build_section, grid=grid, z=z, zz=zz,
            ele_adj_dict=adj_dict)
    with Pool(MAX_JOBS) as p:
        sections_list = p.starmap(build_section_nameconf,
                [(n, sects_config[n]) for n in sects_config.sections()])
    return dict(zip(sects_config.sections(), sections_list))

def build_section(name, config, grid, z, zz, ele_adj_dict):
    """Do all the unstructured grid math to identify the geometry of a transect

    This is the meat of the processing, and is based heavily on Ted Conroy's
    version of an FVCOM transport calculator Matlab script
    (https://github.com/tedconroy/ocean-model-codes/blob/master/fvcom/fvcom_calcfluxsect.m)
    """

    logger = root_logger.getChild('build_section')

    G = nx.Graph(ele_adj_dict)
    waypoints = np.array(config['waypoints'].split(" ")).astype(int)
    assert len(waypoints) >= 2, f'Not enough waypoints defined for section {name}'
    # Find all the elements by taking the shortest path in the graph
    # through the given waypoints
    eles = []
    for w,x in zip(waypoints[:-1], waypoints[1:]):
        eles.extend(nx.shortest_path(G, source=w, target=x,
            weight=lambda p1, p2, atts: grid.el_dist(p1, p2)))
    eles = np.array(eles)

    logger.debug(name, eles)
    ele_xs, ele_ys, ele_hs = tuple(grid.elcoord[:,eles-1])
    ele_nvs = grid.nv[:,eles-1]

    # Calculate the x/y/depth coordinates at the midpoints of the edges
    # between each element, plus the boundary edges
    xm = np.zeros(len(eles)+1)
    ym = np.zeros_like(xm)
    hnm = np.zeros_like(xm)

    # First, handle the first and last elements which have an edge on the
    # model boundary. Find those boundary locations
    for b in (0,-1):
        ele_idx = eles[b] - 1
        # The nodes listed in nv in the same locations as the neighbor
        # elements (from nbe) are the boundary nodes
        boundary_nodes = grid.nv[grid.nbe[:,ele_idx].nonzero()[0], ele_idx]
        # The midpoints of the boundary nodes are the start/end of the
        # transect
        xm[b], ym[b], hnm[b] = grid.ncoord[:, boundary_nodes-1].mean(axis=1)

    # Now fill in the rest
    for i,e1,e2 in zip(range(1, len(eles)), eles[:-1], eles[1:]):
        # These are the common nodes between the neighbor elements
        which_prev_nei = (grid.nbe[:, e1 - 1] == e2).nonzero()[0]
        non_nei_idxs = (np.arange(3) != which_prev_nei).nonzero()[0]
        common_nodes = grid.nv[non_nei_idxs, e1 - 1]
        xm[i], ym[i], hnm[i] = grid.ncoord[:, common_nodes-1].mean(axis=1)

    # Calculate the up-estuary unit vectors that define the flux surface
    # for each element
    ns1 = np.zeros((len(eles), 2))
    ns2 = np.zeros((len(eles), 2))
    # Vector from the centroid to the previous midpoint
    dx1 = xm[:-1] - ele_xs
    dy1 = ym[:-1] - ele_ys
    th1 = np.arctan2(-dy1, -dx1)
    th1 = (np.where(th1 < 0, th1 + 2 * np.pi, th1) - np.pi / 2) % (2 * np.pi)
    # Vector from the centroid to the next midpoint
    dx2 = xm[1:] - ele_xs
    dy2 = ym[1:] - ele_ys
    th2 = np.arctan2(dy2, dx2)
    th2 = (np.where(th2 < 0, th2 + 2 * np.pi, th2) - np.pi / 2) % (2 * np.pi)
    # Vector from previous to next midpoint (completes a triangle)
    dx3 = xm[1:] - xm[:-1]
    dy3 = ym[1:] - ym[:-1]
    # Calculate the unit normal of the third vector, pointing to the right
    th3 = np.arctan2(dy3, dx3)
    # Adjust bounds to be [0..2pi]
    th3 = np.where(th3 < 0, th3 + 2 * np.pi, th3)
    # Standardize the unit vector to point to the right of (dx3, dy3)
    th3 = (th3 - np.pi / 2) % (2 * np.pi)
    # If there's a direction override to the left, adjust the angle so
    # it points in the correct direction
    if 'upesty' in config and config['upesty'] == 'l':
        th1 += np.pi
        th2 += np.pi
        th3 += np.pi
    # Save the x/y coordinates of the unit vector
    ns1[:,0] = np.cos(th1)
    ns1[:,1] = np.sin(th1)
    ns2[:,0] = np.cos(th2)
    ns2[:,1] = np.sin(th2)

    # Compute the average depth of cell centers
    # Shape is (siglay, len(eles))
    z0 = np.expand_dims(zz, 1) @ np.expand_dims(ele_hs, 0)
    # Compute the horizontal dimension of each flow section
    # Shape is (len(eles),)
    a = (np.sqrt((xm[1:] - ele_xs) ** 2 + (ym[1:] - ele_ys) ** 2) +
        np.sqrt((ele_xs - xm[:-1]) ** 2 + (ele_ys - ym[:-1]) ** 2))
    # Compute the vertical thickness of each depth layer
    # Shape is (siglay, len(eles))
    zthick = np.expand_dims(z[:-1] - z[1:], 1) @ np.expand_dims(ele_hs, 0)
    da0 = a * zthick

    # Two more things to calculate:
    # neighbors of each element (won't have grid object later to get them)
    # pinv matrix for each element's neighbor or midpoint coordinates
    # (needed to compute velocity field inside element; see eq's 3.20 and
    # 3.21 of the FVCOM manual)
    allneis = (grid.nbe[:,eles-1] - 1).T
    elneis = []
    pinvs = []
    for i,(el,neis) in enumerate(zip(eles, allneis)):
        neis = neis[neis > -1]
        elneis.append(neis)
        dxy = (grid.elcoord[0:2, neis].T -
               grid.elcoord[0:2,el-1].T)
        #if len(neis) == 2:
            # We're on an edge, so add the midpoint of that edge as a
            # constraint that will be set to zero velocity later
        #    edgept = [xm[0], ym[0]] if i == 0 else [xm[-1], ym[-1]]
        #    dxy = np.concatenate((dxy, [edgept]))
        pinvs.append(np.linalg.pinv(dxy))

    return {
        "eles": eles,
        "neis": elneis,
        "xypinv": pinvs,
        "x": ele_xs,
        "y": ele_ys,
        "h": ele_hs,
        "xm": xm,
        "ym": ym,
        "n1": ns1,
        "n2": ns2,
        "hm": hnm,
        "z0": z0,
        "a": a,
        "da0": da0
    }


def start_netcdf(ds, out_fn, NT, sectiondata, NZ, Lon, Lat, Ldir, vn_list=('salt',)):
    """Create a NetCDF file for the extraction results.

    This is almost identical to Parker MacCready's start_netcdf function
    in x_tef/tef_fun.py with a few tweaks to copy variables/dimensions from
    an FVCOM NetCDF file, and store a bit of extra data for custom
    processing.
    """

    try: # get rid of the existing version
        os.remove(out_fn)
    except OSError:
        pass # assume error was because the file did not exist
    # generating some lists
    if vn_list == 'all':
        vn_list = []
        # all time-varying 3D variables on the siglay/node grid
        for vv in ds.variables:
            vdim = ds.variables[vv].dimensions
            if 'time' in vdim and 'siglay' in vdim and 'node' in vdim:
                vn_list.append(vv)
    # and some dicts of long names and units
    long_name_dict = dict()
    units_dict = dict()
    for vn in vn_list + ('ocean_time',):
        dsvn = vn if vn != 'ocean_time' else 'time'
        try:
            long_name_dict[vn] = ds.variables[dsvn].long_name
        except:
            long_name_dict[vn] = ''
        try:
            units_dict[vn] = ds.variables[dsvn].units
        except:
            units_dict[vn] = ''
    # add custom dict fields
    units_dict['ocean_time'] = 'sec from 1/1/1970'
    long_name_dict['q'] = 'transport'
    units_dict['q'] = 'm3 s-1'
    long_name_dict['lon'] = 'longitude'
    units_dict['lon'] = 'degrees'
    long_name_dict['lat'] = 'latitude'
    units_dict['lat'] = 'degrees'
    long_name_dict['h'] = 'depth'
    units_dict['h'] = 'm'
    long_name_dict['ele'] = 'FVCOM grid element index'
    units_dict['ele'] = 'index from 1'
    long_name_dict['z0'] = 'z on sigma-grid with zeta=0'
    units_dict['z0'] = 'm'
    long_name_dict['DA0'] = 'cell area on sigma-grid with zeta=0'
    units_dict['DA0'] = 'm2'
    long_name_dict['n'] = 'normal unit vector components in up-estuary direction'
    units_dict['n'] = 'none'

    # initialize netcdf output file
    foo = Dataset(out_fn, 'w')
    foo.createDimension('xi_sect', sectiondata['eles'].size)
    foo.createDimension('s_z', NZ)
    foo.createDimension('ocean_time', NT)
    foo.createDimension('sdir_str', 2)
    foo.createDimension('xy', 2)
    for vv in ['ocean_time']:
        v_var = foo.createVariable(vv, float, ('ocean_time',))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in vn_list + ('q',):
        v_var = foo.createVariable(vv, float, ('ocean_time', 's_z', 'xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['z0', 'DA0']:
        v_var = foo.createVariable(vv, float, ('s_z', 'xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['lon', 'lat', 'h']:
        v_var = foo.createVariable(vv, float, ('xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['ele']:
        v_var = foo.createVariable(vv, int, ('xi_sect'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]
    for vv in ['zeta']:
        v_var = foo.createVariable(vv, float, ('ocean_time', 'xi_sect'))
        v_var.long_name = 'Free Surface Height'
        v_var.units = 'm'
    for vv in ['n']:
        v_var = foo.createVariable(vv, float, ('xi_sect', 'xy'))
        v_var.long_name = long_name_dict[vv]
        v_var.units = units_dict[vv]

    # add static variables
    foo['lon'][:] = Lon
    foo['lat'][:] = Lat
    foo['ele'][:] = sectiondata['eles']
    bisectors = sectiondata['n1'] + sectiondata['n2']
    foo['n'][:] = (bisectors.T /
            np.sqrt(bisectors[:,0] ** 2 + bisectors[:,1] ** 2)).T

    # add global attributes
    foo.gtagex = Ldir['gtagex']
    foo.date_string0 = Ldir['date_string0']
    foo.date_string1 = Ldir['date_string1']

    return foo

def plot_locations(ax, grid, sections_gdf, sectiondata, all_x=None, all_y=None):
    p = sections_gdf.plot(ax=ax, color='tab:red', zorder=2)
    # Add up-estuary flow direction arrows
    for sect in sectiondata.values():
        ax.quiver((sect['x'] + sect['xm'][:-1]) / 2,
                (sect['y'] + sect['ym'][:-1]) / 2, sect['n1'][:,0],
                sect['n1'][:,1], zorder=3, alpha=0.7)
        ax.quiver((sect['x'] + sect['xm'][1:]) / 2,
                (sect['y'] + sect['ym'][1:]) / 2, sect['n2'][:,0],
                sect['n2'][:,1], zorder=3, alpha=0.7)
    xmin, xmax, ymin, ymax = p.axis()
    if len(sections_gdf) == 1:
        # Zoom out a bit and adjust the aspect ratio
        r = 0.5
        min_ar = 0.4
        max_ar = 1
        min_dy = (xmax - xmin) / max_ar
        min_dx = (ymax - ymin) * min_ar
        if ymax - ymin < min_dy:
            ymin += (ymax - ymin) / 2 - min_dy / 2
            ymax += (ymax - ymin) / 2 + min_dy / 2
        elif xmax - xmin < min_dx:
            xmin += (xmax - xmin) / 2 - min_dx / 2
            xmax += (xmax - xmin) / 2 + min_dx / 2
        zoom_fact = (.5 - 1/(2*r))
        xmin, xmax, ymin, ymax = (xmin + zoom_fact * (xmax - xmin),
                                  xmax - zoom_fact * (xmax - xmin),
                                  ymin + zoom_fact * (ymax - ymin),
                                  ymax - zoom_fact * (ymax - ymin))
    domain_gdf = grid.elements_gdf()
    domain_gdf.boundary.plot(ax=ax, alpha=0.8, zorder=1)
    ax.set(ybound=(ymin,ymax), xbound=(xmin,xmax), xticklabels=(),
            yticklabels=())
    try:
        cx.add_basemap(ax, crs=domain_gdf.crs, url=cx.providers.Stamen.TonerLite)
    except requests.HTTPError as e:
        # Try with a max zoom level
        cx.add_basemap(ax, crs=domain_gdf.crs, url=cx.providers.Stamen.TonerLite, zoom=13)
    if len(sections_gdf) > 1:
        texts = sections_gdf.apply(
                lambda x: ax.annotate(x['name'], xy=x['geometry'].coords[0],
                    ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white',
                        alpha=0.6)]), axis=1)
        adjust_text(texts, all_x, all_y, arrowprops=dict(arrowstyle='-'))

def make_plots(indata, sections, outdir):
    """Make model grid and profile plots"""

    dir_name = PLOT_ROOT_DIR.format(outdir=outdir)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    grid = FvcomGrid(np.array([indata['x'][:],indata['y'][:],indata['h'][:]]),
            indata['nv'][:].astype(int))
    z = indata['siglev'][:]

    section_plot_nameconf = partial(make_section_plot, outdir=outdir,
            grid=grid, z=z)
    with Pool(MAX_JOBS) as p:
        res = p.starmap(section_plot_nameconf,
                sections.items())
        names = map(lambda x: x[0], res)
        all_x = np.concatenate([x[1] for x in res])
        all_y = np.concatenate([x[2] for x in res])
        sections_gdf = pd.concat([x[3] for x in res], ignore_index=True)

    if len(sections) > 1:
        # Plot all the sections in overhead view
        fig, ax = plt.subplots(figsize=(6,8))
        plot_locations(ax, grid, sections_gdf, sections, all_x=all_x, all_y=all_y)
        ax.set_title("All Sections")
        fig.savefig(dir_name + '/all_sections_map.png')
        plt.close(fig)

class SectionLocator(ticker.AutoLocator):
    """Automatically position A and B ticks at the extreme ends of the X-axis
    """
    def __call__(self):
        locs = super().__call__()
        bounds = self.axis.get_view_interval()
        prev_loc = np.where(locs < bounds[-1], locs, 0).argmax()
        # Add a tick to the very edge, unless it's too close to an existing
        # tick in which case replace the existing one
        if bounds[-1] - locs[prev_loc] < 0.05 * (bounds[-1] - bounds[0]):
            locs[prev_loc] = bounds[-1]
        else:
            locs = np.array(list(locs) + [bounds[-1]])
        return locs

class SectionFormatter(ticker.ScalarFormatter):
    """Automatically label the extreme ticks A and B"""
    def __call__(self, x, pos=None):
        bounds = self.axis.get_view_interval()
        if x == bounds[0]:
            return 'A'
        elif x == bounds[-1]:
            return 'B'
        else:
            return super().__call__(x, pos=(None if pos is None else pos))

def make_section_plot(name, sectiondata, outdir, grid, z):
    # Interweave the x/y and xm/ym arrays
    sect_x = np.zeros(sectiondata['x'].size + sectiondata['xm'].size)
    sect_y = np.zeros_like(sect_x)
    sect_x[0::2] = sectiondata['xm']
    sect_x[1::2] = sectiondata['x']
    sect_y[0::2] = sectiondata['ym']
    sect_y[1::2] = sectiondata['y']
    # Create a GeoDataFrame for the section with a LineString geometry
    geom = LineString([(x, y) for x, y in zip(sect_x, sect_y)])
    gdf = gpd.GeoDataFrame({'name': [name], 'geometry': [geom]},
            crs='epsg:32610')

    # Open the extraction file
    ds = Dataset(EXTRACTION_OUT_PATH.format(outdir=outdir, name=name))
    z0 = ds['z0'][:]
    da0 = ds['DA0'][:]
    # get time axis for indexing
    ot = ds['ocean_time'][:]
    dt = pd.Timestamp('1/1/1970 00:00') + pd.to_timedelta(ot, 'sec')
    tind = np.arange(len(dt))
    dt_ser = pd.Series(index=dt, data=tind)
    # time variable fields
    q = ds['q'][:]
    salt = ds['salt'][:]
    svmin = salt.mean() - 2*salt.std()
    svmax = salt.mean() + 2*salt.std()

    dir_name = os.path.dirname(PLOT_PATH.format(outdir=outdir, name=name, plot=''))
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    # Compute distances between each element along the transect, for the
    # salinity contour
    # Start with the legs from the previous midpoint to the centroid
    center_dists = np.sqrt((sectiondata['x'] - sectiondata['xm'][:-1]) ** 2 +
            (sectiondata['y'] - sectiondata['ym'][:-1]) ** 2)
    # Now add the legs from the previous centroid to the previous midpoint
    # (for all elements after the first one)
    center_dists[1:] += np.sqrt(
            (sectiondata['xm'][1:-1] - sectiondata['x'][:-1]) ** 2 +
            (sectiondata['ym'][1:-1] - sectiondata['y'][:-1]) ** 2)
    # Create a running total
    xsect = np.cumsum(center_dists) / 1000

    # For the colormesh edges: create a running distance for the section
    # midpoints
    dists = np.cumsum(sectiondata['a']) / 1000
    dists = np.concatenate(([0], dists))

    # Create a 2D array for the vertical edges of each cell
    depth_edges = z[:,np.newaxis] @ sectiondata['hm'][np.newaxis,:]

    for mm,dt_mo in dt_ser.groupby(lambda i: i.month):
        it0 = dt_mo[0]
        it1 = dt_mo[-1]

        # form time means
        qq = q[it0:it1,:].mean(axis=0)
        ss = salt[it0:it1,:].mean(axis=0)

        fig = plt.figure(figsize=(13,8))

        # Modified from Parker's code to use flat shading instead of nearest,
        # as this produces more accurately outer edges.
        ax = fig.add_subplot(221)
        cs = ax.pcolormesh(dists, depth_edges, 100*qq/da0, vmin=-10, vmax=10, cmap='bwr')
        fig.colorbar(cs)
        ax.set(title='Mean Velocity (cm/s) Month = ' + str(mm),
                ylabel='Depth (m)',xbound=(0,dists[-1]), xticklabels=())
        ax.xaxis.set_major_locator(SectionLocator())

        ax = fig.add_subplot(223)
        cs = ax.pcolormesh(dists, depth_edges, ss, vmin =svmin, vmax=svmax, cmap='rainbow')
        fig.colorbar(cs)
        contour_interval = .2
        ax.contour(xsect*np.ones((z0.shape[0],1)), z0, ss,
            np.arange(0,35,contour_interval), colors='black', linewidths=.4)
        ax.set(title='Mean Salinity (C.I. = %0.1f)' % (contour_interval),
            xlabel='Transect Distance (km)', ylabel='Depth (m)',
            xbound=(0,dists[-1]))
        ax.xaxis.set_major_locator(SectionLocator())
        ax.xaxis.set_major_formatter(SectionFormatter())

        # Add section location map
        ax = fig.add_subplot(122)
        plot_locations(ax, grid, gdf, {name: sectiondata})
        for lbl,x,y in zip(('A','B'),sectiondata['xm'][[0,-1]],sectiondata['ym'][[0,-1]]):
            ax.annotate(lbl, xy=(x, y), ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white',
                        alpha=0.6)])
        nnnn = ('0000' + str(mm))[-4:]
        plotname = 'plot_' + nnnn
        fig.savefig(PLOT_PATH.format(outdir=outdir, name=name, plot=plotname))
        plt.close(fig)

    ds.close()
    return name, sect_x, sect_y, gdf

if __name__ == "__main__": main()
