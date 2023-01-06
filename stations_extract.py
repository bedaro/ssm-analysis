#!/usr/bin/env python3
# Script version of the Extract_Stations.ipynb notebook, for running on a
# cluster node

from argparse import ArgumentParser, Namespace, FileType
from collections import deque
import time
import logging
import os
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_VARS = 52
STATE_VARS_BOTTOM = 104

def get_list_of_variable_names_from_line(line):
    return line.replace("Variables=", "").replace("\"", "").rstrip().split(",")

# Read in the stations file, creating a dict of every variable found
# including the corresponding time, station ID/node, and layer
def get_raw_data(f, water_vars, bottom_vars):
    total_size = os.path.getsize(f)
    start_time = time.perf_counter()
    with open(f) as fp:
        # The first line is just a label
        fp.readline()
        nstation, nlayer = np.loadtxt([fp.readline()]).astype(int)
        logger.debug("Nstation {0}, Nlayer {1}".format(nstation, nlayer))

        variables_list = get_list_of_variable_names_from_line(fp.readline())
        logger.debug(variables_list)
        variables_list.insert(0, "Time")

        data = {}
        for v in variables_list:
            data[v] = []
        times = []

        def read_block(varct, t):
            block = []
            for i, v in enumerate(variables_list):
                if v == 'Time':
                    data[v].append(t)
                    continue
                # The extra three is for the station, node, and layer
                if i >= varct + 3:
                    # Fill in empty data that's not applicable to this layer
                    data[v].append(np.nan)
                    continue
                if len(block) == 0:
                    block = deque(np.genfromtxt([fp.readline()], missing_values='*************'))
                data[v].append(block.popleft())

        try:
            i = 0

            while True:
                # Read the number of stations/layers and the time
                line = fp.readline()
                if not line:
                    break
                istation, ilayers, t = np.loadtxt([line])
                istation = int(istation)
                ilayers = int(ilayers)
                times.append(t)
                logger.debug("TIME", t)
                for s in range(istation):
                    for l in range(ilayers-1):
                        read_block(water_vars, t)
                    read_block(bottom_vars, t)
                i += 1
                if i % 10 == 0:
                    read_frac = fp.tell() * 100 / total_size
                    elapsed = (time.perf_counter() - start_time)
                    to_go = elapsed * (total_size / fp.tell() - 1)

                    logger.info("{0:.1f}% of file read, {1}s elapsed, {2}s to go, {3}KBps".format(
                        read_frac, int(elapsed), int(to_go),
                        int(fp.tell()/elapsed/1000)))
        except StopIteration:
            pass
    return data

def main():
    script_home = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Extract data from SSM stations output file")
    parser.add_argument("stationsfile",
            help="The ssm_station.out file")
    parser.add_argument("outfile", help="The output file name")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            help="Print progress messages during the extraction")
    parser.add_argument("--state-vars", type=int, default=STATE_VARS,
            help="Override the number of in-water variables to look for in the file")
    parser.add_argument("--state-vars-bottom", type=int, default=STATE_VARS_BOTTOM,
            help="Override the number of last-layer variables to look for in the file")
    parser.add_argument("-a", "--output-all", action="store_true",
            dest="output_all", help="Produce Excel files, one per layer, with all nodes/st vars")
    parser.add_argument("-n", "--output-node", type=int, action="append",
            help="Produce Excel outputs for the specified node(s)")
    parser.add_argument("-V", "--output-var", action="append",
            help="Produce Excel outputs for the specified variable name(s)")
    parser.set_defaults(verbose=False, output_all=False, state_vars=STATE_VARS,
            state_vars_bottom=STATE_VARS_BOTTOM)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    #logger.setLevel(logging.DEBUG)

    data = get_raw_data(args.stationsfile, args.state_vars,
            args.state_vars_bottom)
    df = pd.DataFrame(data)
    logger.info("Got {0} rows of data from {1}".format(len(df), args.stationsfile))
    del df['StationID']

    # Fix dtype for node and layer
    df['Node'] = df['Node'].astype(int)
    df['Layer'] = df['Layer'].astype(int)
    # Build the MultiIndex for time/node/layer
    mi = pd.MultiIndex.from_frame(df[["Time","Node","Layer"]])
    del df['Time']
    del df['Node']
    del df['Layer']
    df.set_index(mi, inplace=True)

    df.to_csv(args.outfile, compression="gzip")

    if args.output_node is not None:
        for n in args.output_node:
            selection = df.loc[:, n, :][args.output_var]
            out_file = args.outfile.replace(".csv.gz", "") + "_station_{0}.xlsx".format(n)
            logger.info("Saving {0} rows of {1} variables for node {2} to {3}".format(
                len(selection), len(selection.columns), n, out_file))
            selection.to_excel(out_file)

    if args.output_all:
        for layer in df.index.levels[2]:
            out_excel = re.sub('\..*', f'_l{layer:02}.xlsx', args.outfile)
            logger.info(f"Dumping all layer {layer} data to {out_excel}")
            with pd.ExcelWriter(out_excel) as writer:
                for stvar in df.columns:
                    # Use dropna to avoid creating columns of NaNs for state
                    # variables not present in this layer
                    data = df.loc[:, :, layer][stvar].unstack(level=1).dropna(axis=1, how='all')
                    if len(data.columns):
                        data.to_excel(writer, sheet_name=stvar)

    logger.info("Finished.")

if __name__ == "__main__": main()
