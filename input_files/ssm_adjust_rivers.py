#!/usr/bin/env python3

from argparse import ArgumentParser, FileType, Action
import enum
import logging
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import ssm_write_fwinputs
import ssm_read_fwinputs

logger = logging.getLogger(__name__)

# Enum handling for ArgumentParser. See https://stackoverflow.com/a/60750535
class EnumAction(Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)

class ConcMethod(enum.Enum):
    KEEP = 'keep' # Don't adjust concentrations
    CONSTLOAD = 'constload' # Adjust concentrations so loading remains constant
    # TODO option to use Ecology's regressions

def adjust_stream(data, factors, conc_method):
    """Adjust each column of data DF by the factors Series, using conc_method to determine how

    data DF and factors Series both need to be singly indexed by DateTimeIndex. factors index
    values should be contained in data index.
    """
    cols = {
        'discharge': data['discharge'] * factors
    }
    if conc_method == ConcMethod.CONSTLOAD:
        # Intensive properties and other columns that would be really strange to preserve
        # loading for. These are kept constant
        intensive_cols = {'temp','doxg','dic','talk'}
        for c in intensive_cols:
            cols[c] = data[c]
        for c in set(data.columns) - {'discharge'} - intensive_cols:
            cols[c] = data[c] / factors
    elif conc_method == ConcMethod.KEEP:
        for c in set(data.columns) - {'discharge'}:
            cols[c] = data[c]
    return pd.DataFrame(cols, index=data.index)

def build_climatology(hydrograph):
    """Take the average hydrograph across all complete calendar years, then shift to water year timing and normalize"""
    h = hydrograph.assign(doy = hydrograph.index.dayofyear)
    # Drop leap days and fix dayofyear alignment
    noleap_indexer = ~((h.index.month == 2) & (h.index.day == 29))
    h.loc[h.index.is_leap_year & (h.index.month >= 3), 'doy'] -= 1
    # Realign day of year
    h['doy'] = (h['doy'] + 91) % 365 + 1
    h = h.loc[noleap_indexer]
    mean_daily_climatology = h.groupby('doy').mean().rolling(30).mean()
    return mean_daily_climatology

def classify_stream(climatology):
    """Use scipy.signal.find_peaks to characterize the hydrograph as rain-dominant, mixed, or snow"""
    allpeaks, junk = find_peaks(climatology, prominence=.0008)
    np.sort(allpeaks)
    peakvalues = np.sort(climatology[allpeaks])
    if allpeaks[-1] < 200:
        cls =  'Rain-dominated'
    elif len(allpeaks) == 1 or peakvalues[-2] < 0.003:
        cls = 'Snow'
    else:
        cls = 'Rain-snow Mix'
    return cls, allpeaks[-1]

def main():
    parser = ArgumentParser(description="Apply river flow multipliers")
    parser.add_argument("infile", type=FileType('rb'),
            help="The combined loadings from ssm_read_fwinputs.py")
    parser.add_argument("adjustments", type=FileType('rb'),
            help="The flow multipliers")
    parser.add_argument("-s", "--sheet-name", help="Sheet name for the flow multipliers")
    parser.add_argument("--source-type", choices=('River','Point Source','both'), default='River',
                        help="Limit adjustments to given source type")
    parser.add_argument("--regions", nargs="+",
                        help="Limit adjustments to given regions")
    parser.add_argument("--country", choices=('Canada','United States'),
                        help="Limit adjustments to given country")
    parser.add_argument("-c", "--concentration-method", type=ConcMethod,
            action=EnumAction, default=ConcMethod.KEEP)
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages")
    parser.add_argument("outfile", type=FileType('wb'),
            help="The destination spreadsheet file")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    logger.info("Reading input spreadsheet")
    dfs = ssm_write_fwinputs.read_data(args.infile)

    logger.info("Reading adjustment spreadsheet")
    adjs = pd.read_excel(args.adjustments, sheet_name=args.sheet_name, index_col=0) if 'sheet_name' in args else pd.read_excel(args.arguments, index_col=0)
    # Fill out the adjustments to cover the entire time period of the input file
    adjs = adjs.reindex(dfs['data'].index.levels[0]).fillna(1)

    logger.info("Identifying target sources")
    idxr = True
    if args.source_type != 'both':
        idxr &= dfs['nodes']['Source Type'] == args.source_type
    if args.country is not None:
        idxr &= dfs['nodes']['Country'] == args.country
    if args.regions is not None:
        idxr &= np.isin(dfs['nodes']['Region'], args.regions)
    if np.all(idxr):
        selected_rivers = dfs['nodes'][['FVCOM ID','Name','Source Type']]
        logger.info(f"Selected all {len(selected_rivers)} sources")
    else:
        selected_rivers = dfs['nodes'].loc[idxr, ['FVCOM ID','Name','Source Type']]
        logger.info(f"Selected {len(selected_rivers)}/{len(dfs['nodes'])} sources")

    logger.info("Adjusting sources")
    riv_characterizations = {}
    # Iterate through all selected sources
    for n,(fvid,name,typ) in selected_rivers.iterrows():
        # FIXME does not handle point sources correctly. Characterization of
        # hydrograph has only been developed for rivers right now.
        if fvid not in riv_characterizations:
            mean_daily_climatology = build_climatology(dfs['data'].xs(n, level=1)[['discharge']])['discharge']
            mean_daily_climatology /= mean_daily_climatology.sum()
            riv_characterizations[fvid], junk = classify_stream(mean_daily_climatology.values)
            logger.debug(f'{typ} {name} characterized as {riv_characterizations[fvid]}')
        # Pass the node time series data and appropriate adjustment series to adjust_stream
        adj = adjs[riv_characterizations[fvid]]
        dfs['data'].loc[(slice(None), n), :] = adjust_stream(dfs['data'].loc[(slice(None), n), :],
            adj, args.concentration_method)

    logger.info("Performing final checks")
    res = ssm_write_fwinputs.check_data(dfs, dfs['data'].index.levels[0].values[0])
    if not res[0]:
        logger.error(res[1])
        sys.exit(1)

    logger.info("Writing the new spreadsheet")
    ssm_read_fwinputs.write_spreadsheet(dfs, args.outfile)

if __name__ == "__main__": main()
