#!/usr/bin/python3

import subprocess
from argparse import ArgumentParser, Namespace
import glob
import tempfile
import os
import unittest
import sys
from netCDF4 import MFDataset, Dataset
import numpy as np

ssm_history = None
nc = None

class TestHistCdfConversion(unittest.TestCase):
    def _compare_vars(self, var, tolerance):
        # The SSM history files output data at the very first iteration, but
        # the netCDF does not. Time indices have to be offset because of this.
        self.assertLess(np.abs(self._modelds[var][:] - self._testds[var][1:]).max(), tolerance)

    @classmethod
    def setUpClass(cls):
        global ssm_history, nc

        cls._tdr = tempfile.TemporaryDirectory()
        test_cdf_output = os.path.join(cls._tdr.name, "test.nc")
        subprocess.check_call(['./ssmhist2cdf','--last-is-bottom',
            test_cdf_output] + ssm_history)

        # Create rival datasets to compare
        cls._modelds = MFDataset(nc)
        cls._testds = Dataset(test_cdf_output)

    def test_times(self):
        # Times should be within a few min of each other if they were created
        # correctly
        self._compare_vars('time', 120)

    # Check other variables
    def test_statevars(self):
        self._compare_vars('DOXG',     1e-4)
        self._compare_vars('salinity', 1e-4)
        self._compare_vars('temp',     1e-4)
        self._compare_vars('NO3',      1e-4)
        self._compare_vars('PO4',      1e-4)

def main():
    global ssm_history, nc
    parser = ArgumentParser(description="Test ssmhist2cdf with model output in ssm_history and NetCDF format")
    parser.add_argument("ssm_history_pattern", type=glob.glob,
            help="Glob pattern for SSM history text files")
    parser.add_argument("nc_pattern", type=glob.glob,
            help="Glob pattern for FVCOMICM .nc files")

    args = parser.parse_args()

    args.ssm_history_pattern.sort()
    args.nc_pattern.sort()

    ssm_history = args.ssm_history_pattern
    nc = args.nc_pattern

    sys.argv[1:] = []
    unittest.main()

if __name__ == "__main__": main()
