#!/usr/bin/env python3

from argparse import ArgumentParser, FileType
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

import ssm_write_fwinputs
import ssm_read_fwinputs

logger = logging.getLogger(__name__)

# Column names in a point source loading file
# that correspond to ssm_write_fwinputs.ALL_STATEVARS
COLS = (6,'I_TEMP', 'I_SALN', None, 'I_DAP', 'I_DFP', None, None, None,
        'I_DOC_F', 'I_DOC_S', 'I_OC_P_F', 'I_OC_P_S', 'I_NH3', 'I_NO3',
        None, 'I_ON_D', None, 'I_ON_P', None, 'I_PO4', 'I_OP_D', None,
        'I_OP_P', None, None, None, 'I_DO', None, None, None, None,
        None, 'I_DIC', 'I_Alk')

def read_loadings(nodes_df, loadings_path, start_date, days):
    dfs = []
    date_range_warning = False
    for i,group in nodes_df.groupby('FVCOM ID'):
        logger.info(f"FVCOM ID {i} ({group['Name'].iloc[0]})")
        srctype = group['Source Type'].iloc[0]
        f = list((loadings_path /
                ('point_sources' if srctype == 'Point Source' else 'nonpoint_sources')
            ).glob(f'{i}_*.xlsx'))
        if len(f) != 1:
            raise ValueError(f'No unique matching loading file found in {loadings_path} for FVCOM ID {i}, {srctype}')
        loading_data = pd.read_excel(f[0], index_col=0,
            skiprows=lambda x: x == 1)
        loading_data.index.name = 'Date'
        loading_data.sort_index(inplace=True)
        end_date = start_date + pd.to_timedelta(days, 'D')
        first_date = loading_data.index[0]
        last_date = loading_data.index[-1]
        if not date_range_warning and (end_date > last_date or start_date < first_date):
            logger.warning(f'{days} from {start_date.strftime("%Y-%m-%d")} requested, but loading data only available from {first_date.strftime("%Y-%m-%d")} to {last_date.strftime("%Y-%m-%d")}')
            date_range_warning = True
        indexer = ((loading_data.index >= start_date) &
            (loading_data.index <= end_date))
        data = {}
        col_copying = True
        try:
            for incol,outcol in zip(COLS, ssm_write_fwinputs.ALL_STATEVARS):
                if incol is None:
                    data[outcol] = np.zeros(len(indexer.nonzero()[0]))
                else:
                    data[outcol] = loading_data.loc[indexer, incol if type(incol) == str else loading_data.columns[incol]]
            col_copying = False
            for j,row in group.iterrows():
                df = pd.DataFrame(data, index=loading_data.loc[indexer].index)
                # Forward fill in missing days so if data interval is
                # monthly, the same values are filled in for every day
                dates = pd.date_range(start_date, end_date, freq='D')
                dates.name = 'Date'
                df = df.reindex(dates, method='ffill')
                df['Node'] = j
                df['discharge'] /= row['Dist Nodes']
                df = df.reset_index().set_index(['Date','Node'])
                dfs.append(df)
        except Exception as e:
            if col_copying:
                logger.critical(f"Error thrown on FVCOM ID {i}, column {incol}")
            else:
                logger.critical(f"Error thrown on FVCOM ID {i}, node {j}")
            raise
    return pd.concat(dfs)

def main():
    parser = ArgumentParser(description="Apply loading data in an existing spreadsheet")
    parser.add_argument("infile", type=FileType('rb'),
            help="The combined loadings from ssm_read_fwinputs.py")
    parser.add_argument("loading_path", type=Path,
            help="Path to the extracted Ecology loadings files")
    parser.add_argument("outfile", type=FileType('wb'),
            help="The destination spreadsheet file")
    parser.add_argument("-s", "--start-date", type=pd.Timestamp,
            default="2014.01.01",
            help="The date from the loadings to use as day zero")
    parser.add_argument("-d", "--days", type=int, default=366,
            help="The number of days to read")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the conversion")

    args = parser.parse_args()
    assert args.loading_path.is_dir()
    assert (args.loading_path / 'point_sources').is_dir()
    assert (args.loading_path / 'nonpoint_sources').is_dir()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    logger.info("Reading spreadsheet")
    dfs = ssm_write_fwinputs.read_data(args.infile)

    logger.info("Importing loading data")
    dfs['data'] = read_loadings(dfs['nodes'], args.loading_path,
        args.start_date, args.days)
    ssm_write_fwinputs.check_data(dfs, args.start_date)

    logger.info("Writing the new spreadsheet")
    ssm_read_fwinputs.write_spreadsheet(dfs, args.outfile)

    logger.info("Finished.")

if __name__ == "__main__": main()
