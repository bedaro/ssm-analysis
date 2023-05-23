#!/usr/bin/env python3

from argparse import ArgumentParser
from configparser import ConfigParser
import os
import os.path as path
from multiprocessing import Pool
import pickle
import logging

from netCDF4 import Dataset, MFDataset
import numpy as np
from scipy.interpolate import interp1d
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd
from xdg import BaseDirectory
from joblib import Memory

from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate
from fvcom.transect import Transect
from fvcom.control_volume import ControlVolume

# From LiveOcean; add LiveOcean/alpha to conda.pth
from zfun import filt_godin

root_logger = logging.getLogger()

DEFAULT_CONFIG = {
    'paths': {},
    'rates': {
        'DOXG': 'REAERDO SODTM1S',
        'NO3': 'JNO3TM1S',
        'NH4': 'JNH4TM1S'
    }
}

# From https://stackoverflow.com/a/312464
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

memory = Memory(path.join(BaseDirectory.xdg_cache_home, "tef_budget"))

@memory.cache
def extract_data(all_files, cv_nodes, node_areas, statevars=[], rates={}):
    cv_nodes = cv_nodes - 1

    allVs = []
    allVSs = []
    allVCs = {s: [] for s in statevars}
    allRs = {s: {r: [] for r in rates[s]} for s in statevars}
    ds = Dataset(all_files[0])
    dcoord = DepthCoordinate.from_output(ds)
    cv_h = ds['h'][cv_nodes-1]
    ds.close()
    for cdfchunk in chunks(all_files, 4):
        with MFDataset(cdfchunk) if len(cdfchunk) > 1 else Dataset(cdfchunk[0]) as ds:
            zetas = ds['zeta'][:,cv_nodes]
            total_vol = node_areas * (np.expand_dims(dcoord.dz, 1) @ np.expand_dims(cv_h, 0))
            # Adjust for tides
            total_vol = total_vol * (1 + np.swapaxes(np.broadcast_to(
                zetas / cv_h, (dcoord.kb - 1, zetas.shape[0], zetas.shape[1])
            ), 0, 1))
            # total_vol is now shape (time, kb-1, nodes)
            allVs.append(total_vol.sum(axis=(1,2)))
            s = ds['salinity'][:,:,cv_nodes]
            allVSs.append((total_vol * s).sum(axis=(1,2)))
            for s in statevars:
                c = ds[s][:,:,cv_nodes]
                allVCs[s].append((total_vol * c).sum(axis=(1,2)))
                if s not in rates:
                    continue
                for r in rates[s]:
                    v = ds[r]
                    if 'siglay' in v.dimensions:
                        d = ds[r][:,:,cv_nodes]
                        allRs[s][r].append((total_vol * d).sum(axis=(1,2)))
                    else:
                        d = ds[r][:,cv_nodes]
                        allRs[s][r].append((node_areas * d).sum(axis=1))
    V = np.concatenate(allVs)
    VS = np.concatenate(allVSs)
    VC = {s: np.concatenate(allVCs[s]) for s in statevars}
    R = {s: {r: np.concatenate(allRs[s][r]) for r in rates[s]} for s in statevars}

    return (V, VS, VC, R)

def plot_budget(df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))
    else:
        fig = None
    l = None
    for c in df.columns:
        color = None
        if c[-5:-2] == 'out':
            color = l[0].get_color()
            linestyle = '.-.'
        elif c[-4:-2] == 'in':
            linestyle = '.--'
        else:
            linestyle = '.-'
        l = ax.plot(df[c].dropna(), linestyle, color=color, label=c, alpha=0.5)
    return fig, ax

def error_stats(df, unit, ax, fudge=0):
    lines = []
    cvchg = df.columns[0]
    err_cols = [cvchg,'$Q_R$','error'] if '$Q_R$' in df.columns else [cvchg, 'error']
    error = df[err_cols].dropna()
    mean_error = error['error'][fudge:].mean()
    if '$Q_R$' in df.columns:
        error_per_qr = np.nan if error['$Q_R$'].mean() == 0 else mean_error / error['$Q_R$'].mean()
        lines.append(f'Mean error: {mean_error:.2f} {unit} ({error_per_qr * 100:.2f}% of mean $Q_R$)')
    else:
        lines.append(f'Mean error: {mean_error:.2f} {unit}')
    error_per_cvchg = np.abs(error['error'] / error[cvchg])[fudge:].mean()
    lines.append(f'Error per {cvchg}: {error_per_cvchg*100:.2f}%')
    rmse = np.sqrt((error['error'][fudge:] ** 2).sum()/len(error['error'][fudge:]))
    if '$Q_R$' in df.columns:
        rmse_per_qr = np.nan if error['$Q_R$'].mean() == 0 else rmse / error['$Q_R$'].mean()
        lines.append(f'RMSE: {rmse:.2f} {unit} ({rmse_per_qr * 100:.2f}% of mean $Q_R$)')
    else:
        lines.append(f'RMSE: {rmse:.2f} {unit}')
    ax.text(0.98, 0.03, '\n'.join(lines), verticalalignment='bottom',
            horizontalalignment='right', transform=ax.transAxes,
            color='tab:red',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 4})

def file_path(string):
    if not os.path.isfile(string):
        raise FileNotFoundError(string)
    return string

def dir_path(string):
    if not os.path.isdir(string):
        raise NotADirectoryError(string)
    return string

def main():
    parser = ArgumentParser(description='Assemble TEF budgets')
    parser.add_argument('model_output', nargs='+', type=file_path,
                        help='Model output dataset')
    parser.add_argument('--configfile', type=file_path,
                        help='Specify a config file for output location, rates, etc')
    parser.add_argument('--output-start-date', default='2014-01-01',
                        type=pd.Timestamp, help='Model output start date')
    parser.add_argument('rivers', type=Dataset, help='Rivers input file (as NetCDF)')
    parser.add_argument('tef_item', help='TEF extraction of this model')
    parser.add_argument('--sections', required=True, nargs='+',
                        help='Names of sections that define a control volume')
    parser.add_argument('--statevars', nargs='*',
                        help='Names of state variables to make extra budgets for')
    parser.add_argument('--name', '-n', help='The name to give this budget')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Turn on verbose output')
    parser.add_argument('--notideavg', action='store_true',
                        help='Turn off tide averaging where possible')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logger = root_logger.getChild('main')

    # Check config file
    config = ConfigParser()
    # Preserve case sensitivity
    config.optionxform = lambda option: option
    if args.configfile is None:
        config.read_dict(DEFAULT_CONFIG)
    else:
        config.read(args.configfile)
    # Assemble rates dict
    rates = {}
    if 'rates' in config.sections():
        for r in config['rates']:
            rates[r] = config['rates'][r].split(' ')

    if 'LOo' in config['paths']:
        indir0 = path.join(config['paths']['LOo'], 'tef')
    else:
        import Lfun
        Ldir = Lfun.Lstart()
        indir0 = path.join(Ldir['LOo'], 'tef')
    extracts_path = path.join(indir0, args.tef_item)
    if not path.isdir(extracts_path):
        raise NotADirectoryError(extracts_path)

    # Check the model output
    hydro_output = MFDataset(args.model_output)
    for s in args.statevars:
        assert s in hydro_output.variables, f'Constituent {s} is not in the model output'
        for r in rates[s]:
            assert r in hydro_output.variables, f'Rate {r} is not in the model output'
    grid = FvcomGrid.from_output(hydro_output)

    # Check TEF sections
    section_files = {}
    section_extractions = {}
    transects = []
    for s in args.sections:
        f = os.path.join(extracts_path, f'extractions/{s}.nc')
        section_files[s] = f
        section_extractions[s] = Dataset(f)
        transects.append(Transect(grid, section_extractions[s]['ele'][:]))

    # Define the control volume
    cv = ControlVolume.from_transects(transects)
    cv_node_list = np.array(cv.nodes_list)
    logger.info(f'Found {len(cv_node_list)} nodes in control volume')

    # Check TEF bulk data
    for s, d in zip(args.sections, cv.transect_directions()):
        bulkpath = path.join(extracts_path, 'bulk', s + '.p')
        with open(bulkpath, 'rb') as f:
            bulk_result = pickle.load(f)
            assert len(args.statevars) == 0 or 'QQC' in bulk_result, 'No TEF constituents available for statevar budgets'
            for v in args.statevars:
                assert v in bulk_result['QQC'], f'Requested constituent {v} is not present in TEF data {bulkpath}'

    rivers_in_cv = set(args.rivers['node'][:]) & cv.nodes
    logger.info(f'Found {len(rivers_in_cv)} river nodes in control volume')

    run_name = '_'.join(args.sections) if args.name is None else args.name
    outdir = path.join(extracts_path, 'budgets', run_name)
    logger.info(f"All outputs will be in {outdir}")
    os.makedirs(outdir, exist_ok=True)

    logger.info('All inputs look good, proceeding to budget calculations')

    # Match up different output times
    onesect = next(iter(args.sections))
    # model output times are in seconds
    # Get times from a section extraction since they are at a guaranteed 1-hr
    # frequency
    section_dates = pd.Timestamp('1970-01-01') + pd.to_timedelta(section_extractions[onesect]['ocean_time'][:], 's')
    section_output_times = ((section_dates - pd.Timestamp(section_extractions[onesect].date_string0)) /
                            np.timedelta64(1, 's')).astype(int).to_numpy()
    # The output times from the model run may be at a higher frequency, but
    # there should be overlap
    model_output_times = hydro_output['time'][:]
    # river times are in hours; convert them
    river_times = args.rivers['time'][:] * 3600
    # The model can potentially run past the boundary conditions, and if that
    # happened we need to truncate the output
    cut_indices_model = (model_output_times > river_times.max()).nonzero()[0]
    if len(cut_indices_model) > 0:
        time_removed_model = (model_output_times[cut_indices_model.max()] -
                              model_output_times[cut_indices_model.min()]) / 87600
        logger.warning(f'Need to remove {time_removed_model:.2f} days of model output from end of run')
        right_cut_model = cut_indices_model.min()
        right_cut_deriv = right_cut_model - 1
        model_output_times = model_output_times[:right_cut_model]
    else:
        right_cut_model = None
        right_cut_deriv = None
    cut_indices_section = (section_output_times > river_times.max()).nonzero()[0]
    if len(cut_indices_section) > 0:
        right_cut_section = cut_indices_section.min()
        section_output_times = section_output_times[:right_cut_section]
        section_dates = section_dates[:right_cut_section]
    else:
        right_cut_section = None
    logger.debug(f'TIMES: section {len(section_dates)}; river {len(river_times)}')

    # Compute node areas
    node_tces = grid.tces_gdf()
    node_areas = node_tces.loc[cv_node_list, 'geometry'].area.to_numpy()

    # Extract all control volume data from model outputs
    V, VS, VC, R = extract_data(args.model_output, cv_node_list, node_areas, args.statevars, rates)
    logger.debug(f'V.shape is {V.shape}')

    # Convert rate units
    # All model output rates are per day, so convert them to per sec for later
    # use. Also, sediment flux rates are in mg rather than g
    for s,sr in R.items():
        for r in sr.keys():
            sr[r] /= 86400
            if r.startswith('J') and r.endswith('TM1S'):
                sr[r] /= 1000
            elif r == 'SODTM1S':
                sr[r] *= -1

    # Volume balance
    # See https://stackoverflow.com/a/18993405
    dVdt = ndimage.gaussian_filter1d(V, sigma=1, order=1)[1:] / np.diff(hydro_output['time'][:])
    logger.debug(f'dVdt shape is {dVdt.shape}')

    # Compute Q_R from river data
    if len(rivers_in_cv) == 0:
        river_qs = np.zeros_like(args.rivers['time'][:])
    else:
        rivers_idxs = np.where(np.isin(np.ma.getdata(args.rivers['node'][:]),
                               list(rivers_in_cv)))[0]
        river_qs = args.rivers['discharge'][:,rivers_idxs].sum(axis=1)
    logging.debug(f'Q_R shape is {river_qs.shape}')
    fig, ax = plt.subplots()
    river_dates = pd.Timestamp(args.output_start_date) + pd.to_timedelta(args.rivers['time'][:], 'H')
    ax.plot(river_dates, river_qs)
    ax.set(ylabel='Flow ($m^3/s$)', title=f'Total $Q_R$ inside {", ".join(args.sections)}')
    fig.savefig(path.join(outdir, 'qr.png'))
    rivers_interp = interp1d(river_times, river_qs)(section_output_times)

    # Q_prism: tidal transport in and out of section
    qpr_in = {}
    qpr_out = {}
    transect_directions = cv.transect_directions()
    for s, d in zip(args.sections, transect_directions):
        direction = 1 if d else -1
        ds = section_extractions[s]
        # shape becomes (s_z, xi_sect, time)
        q = np.moveaxis(ds['q'][:], 0, 2)
        # shape becomes (xi_sect, time)
        qpr_in[s] = (np.where(q * direction > 0, q, 0) * direction).sum(axis=(0,1))[:right_cut_section]
        qpr_out[s] = (np.where(q * direction < 0, q, 0) * direction).sum(axis=(0,1))[:right_cut_section]

    model_dates = args.output_start_date + pd.to_timedelta(model_output_times[:], 's')
    vol_budget = build_vol_budget(model_dates[1:], section_dates,
                                  dVdt[:right_cut_deriv], qpr_in, qpr_out,
                                  rivers_interp, qtype='prism',
                                  tide_avg=not args.notideavg)
    vol_budget.to_excel(path.join(outdir, 'vol_budget.xlsx'))
    fig, ax = plot_budget(vol_budget)
    if args.notideavg:
        qr = vol_budget['$Q_R$'].dropna()
        ax.set(xbound=(qr.index[8400], qr.index[8460]))
    ax.set_ylabel("Flow ($m^3/s$)")
    ax.legend(loc='lower left')
    error_stats(vol_budget, '$m^3/s$', ax, fudge=83)
    fig.autofmt_xdate()
    fig.suptitle(f'Volume Budget for {", ".join(args.sections)}')
    fig.savefig(path.join(outdir, 'vol_budget.png'))

    # Salt balance
    dVSdt = ndimage.gaussian_filter1d(VS, sigma=1, order=1)[1:] / np.diff(hydro_output['time'][:])
    qspr_in = {}
    qspr_out = {}
    for s, d in zip(args.sections, transect_directions):
        direction = 1 if d else -1
        ds = section_extractions[s]
        # shape becomes (s_z, xi_sect, time)
        q = np.moveaxis(ds['q'][:], 0, 2)
        salt = np.moveaxis(ds['salt'][:], 0, 2)
        # shape becomes (xi_sect, time)
        tide_adj = (1 + ds['zeta'][:] / ds['h'][:]).T
        qspr_in[s] = (np.where(q * direction > 0, q, 0) * salt * tide_adj * direction).sum(axis=(0,1))[:right_cut_section]
        qspr_out[s] = (np.where(q * direction < 0, q, 0) * salt * tide_adj * direction).sum(axis=(0,1))[:right_cut_section]

    salt_budget = build_salt_budget(model_dates[1:], section_dates,
                                    dVSdt[:right_cut_deriv], qspr_in, qspr_out,
                                    qtype='prism', tide_avg=not args.notideavg)
    salt_budget.to_excel(path.join(outdir, 'salt_budget.xlsx'))
    fig, ax = plot_budget(salt_budget)
    if args.notideavg:
        ax.set(xbound=(section_dates[4400], section_dates[4460]))
    ax.set_ylabel("Salt transport ($psu\\ m^3/s$)")
    ax.legend(loc='lower left')
    error_stats(salt_budget, '$psu\\ m^3/s$', ax)
    fig.autofmt_xdate()
    fig.suptitle(f'Salt Budget for {", ".join(args.sections)}')
    fig.savefig(path.join(outdir, 'salt_budget.png'))

    # TEF budgets
    q_in = {}
    q_out = {}
    qs_in = {}
    qs_out = {}
    qc_in = {s: {} for s in args.statevars}
    qc_out = {s: {} for s in args.statevars}
    for s, d in zip(args.sections, transect_directions):
        with open(path.join(extracts_path, 'bulk', s + '.p'), 'rb') as f:
            bulk_result = pickle.load(f)
        qq = bulk_result['QQ']
        ss = bulk_result['SS']
        bulk_dates = pd.Timestamp('1/1/1970 00:00') + pd.to_timedelta(bulk_result['ot'].data, 'sec')
        qq_in = np.where(qq > 0, qq, 0)
        qq_out = np.where(qq < 0, qq, 0)
        if not d:
            qq_in *= -1
            qq_out *= -1
        q_in[s] = np.nansum(qq_in, axis=1)
        q_out[s] = np.nansum(qq_out, axis=1)
        qs_in[s] = np.nansum(qq_in * ss, axis=1)
        qs_out[s] = np.nansum(qq_out * ss, axis=1)
        if 'QQC' in bulk_result:
            for c,qqc in bulk_result['QQC'].items():
                if c not in qc_in:
                    continue
                qqc_in = np.where((qqc > 0) == d, qqc, 0)
                qqc_out = np.where((qqc < 0) == d, qqc, 0)
                if not d:
                    qqc_in *= -1
                    qqc_out *= -1
                qc_in[c][s] = np.nansum(qqc_in, axis=1)
                qc_out[c][s] = np.nansum(qqc_out, axis=1)
    tef_vol_budget = build_vol_budget(model_dates[1:], bulk_dates,
                                      dVdt[:right_cut_deriv], q_in, q_out,
                                      rivers_interp, tide_avg='notq',
                                      hourly_dates=section_dates)
    tef_vol_budget.dropna(inplace=True)
    tef_vol_budget.to_excel(path.join(outdir, 'tef_vol_budget.xlsx'))
    fig, ax = plot_budget(tef_vol_budget)
    qr = tef_vol_budget['$Q_R$'].dropna()
    ax.set(ylabel="Flow ($m^3/s$)")
    ax.legend(loc='lower left')
    error_stats(tef_vol_budget, '$m^3/s$', ax)
    fig.autofmt_xdate()
    fig.suptitle(f'TEF Volume Budget for {", ".join(args.sections)}')
    fig.savefig(path.join(outdir, 'tef_vol_budget.png'))

    tef_salt_budget = build_salt_budget(model_dates[1:], bulk_dates,
                                        dVSdt[:right_cut_deriv], qs_in, qs_out,
                                        tide_avg='notq',
                                        hourly_dates=section_dates)
    tef_salt_budget.to_excel(path.join(outdir, 'tef_salt_budget.xlsx'))
    fig, ax = plot_budget(tef_salt_budget)
    ax.set(ylabel="Salt transport ($psu\\ m^3/s$)")
    ax.legend(loc='lower left')
    error_stats(tef_salt_budget, '$psu\\ m^3/s$', ax)
    fig.autofmt_xdate()
    fig.suptitle(f'TEF Salt Budget for {", ".join(args.sections)}')
    fig.savefig(path.join(outdir, 'tef_salt_budget.png'))


    # Other TEF constituent budgets
    dVCdt = {s: ndimage.gaussian_filter1d(VC[s], sigma=1, order=1)[1:] / np.diff(hydro_output['time'][:]) for s in args.statevars}
    model_river_cst_map = {
        'NH4': 'nh4',
        'NO3': 'no32',
        'DOXG': 'doxg'
    }

    if len(rivers_in_cv) == 0:
        river_qcs = {s: np.zeros_like(args.rivers['time'][:]) for s in args.statevars}
    else:
        river_qcs = {}
        for s in args.statevars:
            river_qcs[s] = (args.rivers['discharge'][:,rivers_idxs] *
                            args.rivers[model_river_cst_map[s]][:,rivers_idxs]
                           ).sum(axis=1)
    rivers_qc_interp = {s: interp1d(river_times, river_qcs[s])(section_output_times) for s in args.statevars}

    for s in args.statevars:
        if s in rates and len(rates[s]):
            rdata = {r: filt_godin(R[s][r][1:right_cut_model]) for r in rates[s]}
        else:
            rdata = {}
        budget = build_cst_budget(model_dates[1:], section_dates, bulk_dates,
                                  dVCdt[s][:right_cut_deriv], qc_in[s],
                                  qc_out[s], rivers_qc_interp[s], rdata)
        budget.to_excel(path.join(outdir, f'tef_{s}_budget.xlsx'))
        fig, ax = plot_budget(budget)

        unit = hydro_output[s].units
        if unit.endswith(' meters-3'):
            # Cancel out the volumes
            unit = unit.removesuffix(' meters-3') + '/s'
        elif unit == 'MG/L':
            unit = 'g/s'
        else:
            unit += '-m^3/s'
        ax.set(ylabel=f"{s} removal/addition (${unit}$)")
        ax.legend()
        fig.autofmt_xdate()
        fig.suptitle(f'TEF {s} Budget for {", ".join(args.sections)}')
        fig.savefig(path.join(outdir, f'tef_{s}_budget.png'))

def build_vol_budget(model_dates, section_dates, dVdt, q_in, q_out, qr,
                     tide_avg=True, qtype=None, hourly_dates=None):
    model_df = pd.DataFrame({'$dV/dt$': dVdt}, index=model_dates)
    if hourly_dates is None:
        hourly_dates = section_dates
    if tide_avg == 'notq':
        model_df = model_df.loc[model_df.index.isin(hourly_dates)].copy()
        model_df['$dV/dt$'] = filt_godin(model_df['$dV/dt$'])
        qr = filt_godin(qr)
    section_data = {}
    sub = '' if qtype is None else qtype + ','
    for x in q_in.keys():
        section_data[x + ' $Q_\mathrm{' + sub + 'in}$'] = q_in[x]
        section_data[x + ' $Q_\mathrm{' + sub + 'out}$'] = q_out[x]
    qr_series = pd.Series(qr, index=hourly_dates, name='$Q_R$')

    vol_budget = model_df.join(pd.DataFrame(section_data, index=section_dates),
                               how='left').join(pd.DataFrame(qr_series), how='left').dropna()
    if tide_avg is True:
        # We have to make all the data hourly in order for the tidal averaging
        # to work correctly
        vol_budget.dropna(inplace=True)
        for c in vol_budget.columns:
            vol_budget[c] = filt_godin(vol_budget[c].to_numpy())

    vol_budget['error'] = -vol_budget['$dV/dt$'] + vol_budget[vol_budget.columns[1:]].sum(1)
    vol_budget.loc[vol_budget['$Q_R$'].isna(), 'error'] = np.nan
    return vol_budget

def build_salt_budget(model_dates, section_dates, dVSdt, qs_in, qs_out,
                     tide_avg=True, qtype=None, hourly_dates=None):
    if hourly_dates is None:
        hourly_dates = section_dates
    model_salt_df = pd.DataFrame({'$d(VS)/dt$': dVSdt}, index=model_dates)
    if tide_avg == 'notq':
        model_salt_df = model_salt_df.loc[model_salt_df.index.isin(hourly_dates)].copy()
        model_salt_df['$d(VS)/dt$'] = filt_godin(model_salt_df['$d(VS)/dt$'])
    salt_section_data = {}
    sub = '' if qtype is None else qtype + ','
    for x in qs_in.keys():
        salt_section_data[x + ' $(QS)_\mathrm{' + sub + ',in}$'] = qs_in[x]
        salt_section_data[x + ' $(QS)_\mathrm{' + sub + ',out}$'] = qs_out[x]
    salt_budget = model_salt_df.join(pd.DataFrame(salt_section_data, index=section_dates),
                                     how='left')

    if tide_avg is True:
        # We have to make all the data hourly in order for the tidal averaging
        # to work correctly
        salt_budget.dropna(inplace=True)
        for c in salt_budget.columns:
            salt_budget[c] = filt_godin(salt_budget[c].to_numpy())

    salt_budget['error'] = -salt_budget['$d(VS)/dt$'] + salt_budget[salt_budget.columns[1:]].sum(1, skipna=False)
    return salt_budget

def build_cst_budget(model_dates, hourly_dates, section_dates, dVCdt, qc_in,
                     qc_out, qcr, rates):
    model_df = pd.DataFrame({'$d(VC)/dt$': dVCdt}, index=model_dates)
    # Filter just the section dates (hourly) and tidally average
    model_df = model_df.loc[model_df.index.isin(hourly_dates)].copy()
    model_df['$d(VC)/dt$'] = filt_godin(model_df['$d(VC)/dt$'])
    section_data = {}
    for x in qc_in.keys():
        section_data[x + ' $Q^C_\mathrm{in}$'] = qc_in[x]
        section_data[x + ' $Q^C_\mathrm{out}$'] = qc_out[x]
    qr_series = pd.Series(qcr, index=hourly_dates, name='$Q^C_R$')

    df = model_df.join(pd.DataFrame(section_data, index=section_dates),
                       how='left').join(pd.DataFrame(qr_series), how='left')
    if len(rates):
        rates_df = pd.DataFrame(rates, index=model_dates)
        df = df.join(rates_df, how='left')

    df['source/sink'] = df['$d(VC)/dt$'] - df[df.columns[df.columns != '$d(VC)/dt$']].sum(1)
    return df.dropna()

if __name__ == "__main__": main()
