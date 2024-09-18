#!/usr/bin/env python3
# Apply Ecology's regressions to flow timeseries in a loading spreadsheet
# 
# See regscratch.ipynb to see development process and some caveats.
# 
# Current limitations:
# - PIP and organic phosphorus are not implemented
# - alkalinity is not altered, as this was originall based on observations
# - Ecology's regressions for Capitol Lake and Lake Washington are poor
#   fits. They used some extrapolated observational data.
# - Temperature regressions are also unimplemented since Ecology decided
#   to use Cedar River measured temperature instead.
# - DIC is computed from pH regressions and alkalinity, but it's not in
#   perfect agreement with Ecology's numbers. It tends to be 1-10 percent
#   lower.

from argparse import ArgumentParser, FileType
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ssm_write_fwinputs
import ssm_read_fwinputs

# Fraction of concentration which is labile
LR_FRACTIONS = {
    'doc': 0.9,
    'poc': 0.67,
    'don': 1,
    'pon': 1,
    'dop': 1,
    'pop': 1
}

# Carbonate equilibrium constants
KA1 = 10 ** -6.35
KA2 = 10 ** -10.33

CONSTIT_MAP = {
    # Don't alter temperature, salt, tss, algn, zoon, urea, psi, dsi, talk
    'doc': 'DOC',
    'poc': 'POC',
    'nh4': 'NH4N',
    'no32': 'NO23N',
    'don': 'DON',
    'pon': 'PON',
    # Organic phosphorus is not implemented
    'po4': 'OP',
    'doxg': 'DO'
    # pH to DIC is handled separately
}

logger = logging.getLogger("apply_regressions")
logging.addLevelName( logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

# Progress bar implementation from https://discuss.python.org/t/add-a-basic-progressbar-implementation-to-shutil/55142/6
class ProgressBar:
    """Display & update a progress bar"""
    TEXT_ABORTING = "Aborting..."
    TEXT_COMPLETE = "Complete!"
    TEXT_PROGRESS = "Progress"

    def __init__(self, bar_length=25, stream=sys.stdout):
        self.bar_length = bar_length
        self.stream = stream
        self._last_displayed_text = None
        self._last_displayed_summary = None

    def reset(self):
        """Forget any previously displayed text (affects subsequent call to show())"""
        self._last_displayed_text = None
        self._last_displayed_summary = None

    def _format_progress(self, progress, aborting):
        """Internal helper that also reports the number of completed increments and the displayed status"""
        bar_length = self.bar_length
        progress = float(progress)
        if progress >= 1:
            # Report task completion
            completed_increments = bar_length
            status = " " + self.TEXT_COMPLETE
            progress = 1.0
        else:
            # Truncate progress to ensure bar only fills when complete
            progress = max(progress, 0.0) # Clamp negative values to zero
            completed_increments = int(progress * bar_length)
            status = (" " + self.TEXT_ABORTING) if aborting else ""
        remaining_increments = bar_length - completed_increments
        bar_content = f"{'#'*completed_increments}{'-'*remaining_increments}"
        percentage = f"{progress*100:.2f}"
        progress_text = f"{self.TEXT_PROGRESS}: [{bar_content}] {percentage}%{status}"
        return progress_text, (completed_increments, status)

    def format_progress(self, progress, *, aborting=False):
        """Format progress bar, percentage, and status for given fractional progress"""
        return self._format_progress(progress, aborting)[0]

    def show(self, progress, *, aborting=False):
        """Display the current progress on the console"""
        progress_text, progress_summary = self._format_progress(progress, aborting)
        if progress_text == self._last_displayed_text:
            # No change to display output, so skip writing anything
            # (this reduces overhead on both interactive and non-interactive streams)
            return
        interactive = self.stream.isatty()
        if not interactive and progress_summary == self._last_displayed_summary:
            # For non-interactive streams, skip output if only the percentage has changed
            # (this avoids flooding the output on non-interactive streams that ignore '\r')
            return
        if not interactive or aborting or progress >= 1:
            # Final or non-interactive output, so advance to next line
            line_end = "\n"
        else:
            # Interactive progress output, so try to return to start of current line
            line_end = "\r"
        sys.stdout.write(progress_text + line_end)
        sys.stdout.flush() # Ensure text is emitted regardless of stream buffering
        self._last_displayed_text = progress_text
        self._last_displayed_summary = progress_summary

def iter_with_progress(iterable, *, max_iterations=None):
    """Display a progress bar while iterating over an iterable"""
    if max_iterations is None:
        # Iterable must define __len__ if max_iterations is not given
        max_iterations = len(iterable)
    progress_bar = ProgressBar()
    progress_bar.show(0.0)
    items_processed = 0
    for item in iterable:
        yield item
        items_processed += 1
        progress_bar.show(items_processed / max_iterations)
        if max_iterations is not None and items_processed == max_iterations:
            break # Terminate now even if the underlying iterator isn't complete

class RegData:
    def __init__(self, regdf):
        """Get all the regressions for all constituents as a giant dict

        Keys are station names, values are each a dict of all the regression
        parameter lists keyed by constituent (column header).

        Missing terms are NaN
        """
        nulls = pd.isnull(regdf)
        end = len(regdf) - 10
        regressions = {}
        for i in range(0, end, 11):
            if nulls['Station'][i]:
                # blank line
                continue
            name = regdf.iloc[i, 0]
            regressions[name] = {}
            for col in regdf.columns[1:]:
                regressions[name][col] = regdf[col][i+1:i+10].values
        self._regressions = regressions

    def __repr__(self):
        return f'<RegData: {repr(self._regressions)}>'

    def __getitem__(self, name):
        return self._regressions[name]

    def __contains__(self, name):
        return name in self._regressions

    def pick(self, name, regname, regc):
        """Decide which regression values apply

        Returns a tuple of the regression array and the overridden name
        """
        if regname in self:
            return self[regname][regc], regname
        # Sloppiness in spreadsheet
        if name in self:
            return self[name][regc], name
        if name in ('Kitsap NE', 'Kitsap_Hood', 'Port Gamble'):
            #actual_regname = 'Sinclair/Dyes Inlet' if regc in ('DOC','POC') else 'Big Beef Creek'
            actual_regname = 'Big Beef Creek'
        elif name == 'Hamma Hamma R':
            # Ecology's spreadsheet says this source uses Duckabush for 
            # DOC/POC only (Skok for rest) but that's inconsistent with
            # the loading data I have from them. See regscratch.ipynb
            actual_regname = 'Duckabush River'
        elif name == 'NW Hood':
            # Ecology's spreadsheet says this source uses Duckabush for 
            # DOC/POC only (Big Beef for rest) but that's inconsistent with
            # the loading data I have from them. See regscratch.ipynb
            actual_regname = 'Duckabush River'
        elif name == 'Skokomish R':
            #actual_regname = 'Skookum Creek' if regc in ('POC') else 'Skokomish River'
            actual_regname = 'Skokomish River'
        else:
            raise ValueError(f'{name} {regname} {regc}')
        return self[actual_regname][regc], actual_regname

def dates_to_year_frac(dtindex):
    """Convert a datetimeindex to an index of year fraction"""
    return dtindex.day_of_year / np.where(dtindex.is_leap_year, 366, 365)

def do_regression(x, *bs):
    """Compute a river regression based on flow and year fraction

    Extra args are the regression parameters"""
    q, yf = x
    bs0 = np.where(pd.isna(np.array(bs)), 0, bs)

    predict_log = bs0[0] + (np.array([np.log10(q), np.log10(q) ** 2,
        np.sin(2*np.pi * yf), np.cos(2*np.pi * yf), np.sin(4*np.pi * yf),
        np.cos(4*np.pi * yf)]).T @ np.expand_dims(bs0[1:7], 1))
    # transform to linear space and apply smearing
    predict = np.power(10, predict_log[:,0]) * (bs[7] if not np.isnan(bs[7]) else 1)
    # Apply max in bs[8]
    if not np.isnan(bs[8]):
        predict = np.where(predict > bs[8], bs[8], predict)
    # Fill any NaNs that came from zero flow with zeroes
    predict = np.where(q == 0, 0, predict)
    return predict

def update_by_regression(loading_dfs, inflows_df, regdata, sids=None, not_sids=None):
    newdata = loading_dfs['data'].copy()

    sid_indexer = loading_dfs['nodes']['Source Type'] == 'River'
    if sids is not None:
        sid_indexer &= loading_dfs['nodes']['FVCOM ID'].isin(sids)
    elif not_sids is not None:
        sid_indexer &= ~loading_dfs['nodes']['FVCOM ID'].isin(not_sids)
    # Keep track of what constituents were updated
    updated = {}

    for sid,group in loading_dfs['nodes'].loc[sid_indexer].groupby('FVCOM ID'):
        nodes = group.index.values
        name = inflows_df.loc[sid, 'SSM2_Name']
        regname = inflows_df.loc[sid, 'WQ Regression']
        logger.info(f'{sid}: {name} (node {", ".join(nodes.astype(str))})')
        if pd.isnull(regname):
            logger.warning(f'No regressions available for {sid} ({name}), skipping')
            continue
        updated[sid] = []

        # Gather discharge and year fraction data
        q = newdata.loc[newdata.index.get_level_values(1).isin(nodes), 'discharge'].groupby(level=0).sum()
        assert not np.any(pd.isna(q))
        a = inflows_df.loc[sid, 'Drainage Area (mi2)'] * 1.6093 ** 2
        x = (q / a, dates_to_year_frac(q.index))
        assert not np.any(pd.isna(x[0])), "Discharge after division contains NaNs"
        for inputc, regc in CONSTIT_MAP.items():
            bs, actual_regname = regdata.pick(name, regname, regc)
            if np.all(pd.isnull(bs)):
                logger.warning(f'Skipping {regc} for {sid} ({name}), no regression available{f" (from {regname})" if name != regname else ""}')
                continue
            p = do_regression(x, *bs)
            assert not np.any(pd.isna(p)), f'{regc} from {actual_regname} contains NaNs'
            # Handle constituents that need fractionation
            if inputc not in newdata.columns and 'l' + inputc in newdata.columns:
                logger.info(f'Updating {regc} labile and refractory{f" (using {actual_regname})" if name != actual_regname else ""}')
                for n in nodes:
                    newdata.loc[(slice(None),n), 'l' + inputc] = LR_FRACTIONS[inputc] * p
                    newdata.loc[(slice(None),n), 'r' + inputc] = (1 - LR_FRACTIONS[inputc]) * p
                updated[sid].append('l' + inputc)
                updated[sid].append('r' + inputc)
            else:
                logger.info(f'Updating {regc}{f" (using {actual_regname})" if name != actual_regname else ""}')
                for n in nodes:
                    newdata.loc[(slice(None),n), inputc] = p
                updated[sid].append(inputc)
        # Organic N (TODO implement organic P the same way)
        regmethod = int(inflows_df.loc[sid, 'WQ Regression Method'][0])
        dtpn_bs, actual_regname = regdata.pick(name, regname, 'DTPN')
        tpn_bs, actual_regname = regdata.pick(name, regname, 'TPN')
        if regmethod in (1, 2):
            din = newdata.xs(nodes[0], level=1)[['no32','nh4']].sum(axis=1)
            tpn = do_regression(x, *tpn_bs)
            if regmethod == 1: # DTPN and DTP
                dtpn = do_regression(x, *dtpn_bs)
                don = dtpn - din
                don = np.where(don < 0, 0.001, don)
                pon = tpn - dtpn
                pon = np.where(pon < 0, 0.001, pon)
                logger.info(f'Updating DON/PON labile and refractory (using DTPN)')
            else: # DTP only
                org_n = tpn - din
                don = np.where(org_n < 0, 0.001, 0.5 * org_n)
                pon = np.where(org_n < 0, 0.001, 0.5 * org_n)
                logger.info(f'Updating DON/PON labile and refractory (using TPN/OrgN)')

            for n in nodes:
                newdata.loc[(slice(None),n), 'ldon'] = LR_FRACTIONS['don'] * don
                newdata.loc[(slice(None),n), 'rdon'] = (1 - LR_FRACTIONS['don']) * don
                newdata.loc[(slice(None),n), 'lpon'] = LR_FRACTIONS['pon'] * pon
                newdata.loc[(slice(None),n), 'rpon'] = (1 - LR_FRACTIONS['pon']) * pon
            updated[sid].append('ldon')
            updated[sid].append('lpon')
            updated[sid].append('rdon')
            updated[sid].append('rpon')
        elif np.all(pd.isnull(tpn_bs)):
            logger.info('Skipping DON/PON (not available)')
        else:
            logger.info('Skipping DON/PON (available but not supposed to be used)')
        # pH
        bs, actual_regname = regdata.pick(name, regname, 'pH')
        if np.all(pd.isnull(bs)):
            logger.warning(f'Skipping pH (not available in {actual_regname})')
        else:
            p = do_regression(x, *bs)
            ph = np.where(p < 6.5, 6.5, p)
            # [H+] and [OH-] concentrations
            hpl = 10 ** -ph
            ohm = 10 ** (ph-14)
            # Alkalinity doesn't vary by year so just use what we have already
            alk = newdata.loc[(slice(None),nodes[0]), 'talk'].to_numpy()
            # Assume alkalinity is dominated by the carbonate system
            # We could include ammonium and phosphate species but these should be rounding errors
            # Carbonate speciation
            alph1 = KA1 * hpl / (hpl ** 2 + KA1 * hpl + KA1 * KA2)
            alph2 = KA1 * KA2 / (hpl ** 2 + KA1 * hpl + KA1 * KA2)
            # Eq 8.21b from Benjamin
            dic = ((alk + hpl - ohm) / (alph1 + 2 * alph2))
            # Set zero discharge periods to DIC=0
            mcal_new_dic = np.where(q == 0, 0, dic)
            logger.info(f'Updating pH{f" (using {actual_regname})" if name != actual_regname else ""}')
            for n in nodes:
                newdata.loc[(slice(None),n), 'dic'] = dic
            updated[sid].append('dic')

    return newdata, updated

def main():
    parser = ArgumentParser(description="Apply Ecology WQ River Regressions to flows")
    parser.add_argument("infile", type=FileType('rb'),
            help='The combined loadings from ssm_read_fwinputs.py')
    parser.add_argument("outfile", type=FileType('wb'),
            help="The destination spreadsheet file")
    parser.add_argument('-s', '--sources', nargs='+', type=int,
            help='Specify a list of sources (rivers) to adjust (default all)')
    parser.add_argument('-x', '--exclude-sources', nargs='+', type=int,
            help='Specify a list of sources (rivers) to exclude')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Print progress messages')
    parser.add_argument('-p', '--plot-output',
            help='Generate timeseries plots of all altered constituents in directory')
    parser.add_argument('--regressions-file', default='data/regression_metadata.xlsx')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    plot = (args.plot_output is not None)
    if plot:
        if not os.path.isdir(args.plot_output):
            os.mkdir(args.plot_output)
        plot_path = Path(args.plot_output)

    logger.info("Reading input data")
    dfs = ssm_write_fwinputs.read_data(args.infile)

    regressions_df = pd.read_excel(args.regressions_file, header=None,
            usecols='A:Q', sheet_name='2014 WQ Regression Coeffs', names=(
                'Station','Temp','DO','pH','NO23N','NH4N','TPN','DTPN',
                'PON','DON','OP','TP','DTP','POP','DOP','DOC','POC'))
    regdata = RegData(regressions_df)

    # Read the inflows
    # Need to drop Deschutes River row to prevent a duplicate entry
    inflows_df = pd.read_excel(args.regressions_file, sheet_name='SSM list of inflows').drop_duplicates(subset='SSM2_ID').set_index('SSM2_ID')

    newdata, updated = update_by_regression(dfs, inflows_df, regdata, sids=args.sources, not_sids=args.exclude_sources)

    if plot:
        logger.info('Generating plots showing what changed')
        it = iter_with_progress(updated.items()) if args.verbose else updated.items()
        for sid,constituents in it:
            data = dfs['nodes'].loc[dfs['nodes']['FVCOM ID'] == sid]
            nodes = data.index
            name = data.iloc[0]['Name']
            old = dfs['data'].xs(nodes[0], level=1)
            new = newdata.xs(nodes[0], level=1)
            cols = np.min((len(constituents), 4))
            rows = int(np.ceil(len(constituents) / 4))
            fig, axs = plt.subplots(rows, cols, figsize=(10, 3*rows))
            for c,ax in zip(constituents, axs.flatten()):
                l1, = ax.plot(old.index, np.where(old['discharge'] == 0, np.nan, old[c]))
                l2, = ax.plot(new.index, np.where(new['discharge'] == 0, np.nan, new[c]))
                ax.set_title(c)
            fig.suptitle(name)
            fig.legend((l1, l2), ('Original','Updated'), loc='lower right')
            fig.autofmt_xdate()
            fig.savefig(plot_path / f'{sid} {name}.png')
            plt.close(fig)

    logger.info("Writing new data spreadsheet")
    newdfs = {
        'nodes': dfs['nodes'],
        'vqdist': dfs['vqdist'],
        'data': newdata
    }
    ssm_read_fwinputs.write_spreadsheet(newdfs, args.outfile)

    logger.info("Finished.")

if __name__ == "__main__": main()
