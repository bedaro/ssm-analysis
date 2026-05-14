#!/usr/bin/env python3

import unittest
import sys
import logging
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
from pandas import Timestamp

from fvcom.grid import FvcomGrid
from fvcom.control_volume import ControlVolume

# software under test
sys.path.insert(0, str(Path(__file__).parent.parent))
import rawcdf_extract

logging.basicConfig(level=logging.ERROR)

class TestTimeRange(unittest.TestCase):
    def test_start_end_simple(self):
        # From March 1 to March 30, daily interval
        values = np.arange(0, 2592000, 86400)
        start = Timestamp('2003.03.01')
        frm = False
        to = False
        rg = rawcdf_extract.TimeRange.from_t_start_end(values, start, frm, to)
        self.assertEqual(rg.first_time, 0)
        self.assertEqual(rg.last_time, 30)
        self.assertEqual(len(rg), 30)
        self.assertEqual(rg.out_offset, 0)

    def test_start_end_frm_to(self):
        values = np.arange(0, 2591000, 86400)
        start = Timestamp('2003.03.01')
        frm = Timestamp('2003.03.05')
        to = Timestamp('2003.03.25')
        rg = rawcdf_extract.TimeRange.from_t_start_end(values, start, frm, to)
        self.assertEqual(rg.first_time, 4)
        self.assertEqual(rg.last_time, 25)
        self.assertEqual(len(rg), 21)
        self.assertEqual(rg.out_offset, 0)

    def test_matching_no_offset(self):
        values1 = np.arange(0, 100, 2)
        values2 = np.arange(6, 60, 2)

        rg = rawcdf_extract.TimeRange.from_matching(values1, values2)
        self.assertEqual(rg.first_time, 3)
        self.assertEqual(rg.last_time, 30)
        self.assertEqual(len(rg), 27)
        self.assertEqual(rg.out_offset, 0)

    def test_matching_offset(self):
        values1 = np.arange(8, 100, 4)
        values2 = np.arange(0, 100, 4)
        
        rg = rawcdf_extract.TimeRange.from_matching(values1, values2)
        self.assertEqual(rg.first_time, 0)
        self.assertEqual(rg.last_time, 23)
        self.assertEqual(len(rg), 23)
        self.assertEqual(rg.out_offset, 2)

# Test cases needed:
# - copy_data
#   element-based data
#   using entire time slice
#   using multiple partial time slices to get entire span

class TestRawcdfExtract(unittest.TestCase):
    def setUp(self):
        # Read the fake model output
        self.ds = Dataset(Path(__file__).parent / 'data' / 'test_output.nc')
        self.ds2 = None
        self.grid = FvcomGrid.from_output(self.ds)
        # Build a test CV to extract from. All nodes are contiguous
        self.cv = ControlVolume(grid=self.grid, nodes={3, 4, 5, 7, 8, 9, 11, 12})

    def test_copy_attrs(self):
        ds2 = Dataset('fake.nc','w',diskless=True)
        d = ds2.createDimension('x',5)
        v = ds2.createVariable('v', np.float32, (d,))
        rawcdf_extract.copy_ncatts(self.ds, 'data_node', v)
        self.assertEqual(self.ds['data_node'].long_name, v.long_name)
        self.assertEqual(self.ds['data_node'].units, v.units)

    def test_init_output_all(self):
        tstart = Timestamp('2000-01-01')
        # Test data contains 20 points at 3000 second intervals, so it covers
        # a few hours
        tslc = slice(5, 15)
        invars = (('data_node', rawcdf_extract.InputAttr.ALL),)
        self.ds2 = rawcdf_extract.init_output('dummy.nc', self.ds, tstart,
                                              tslc, self.cv, invars,
                                              diskless=True)
        # Basic output checks
        self.assertEqual(tstart, Timestamp(self.ds2.model_start))
        self.assertEqual(self.cv.nodes, set(self.ds2['node'][:]))
        self.assertEqual(sorted(self.ds2['node'][:].tolist()), self.ds2['node'][:].tolist())
        self.assertEqual(self.cv.elements, set(self.ds2['nele'][:]))
        self.assertTrue('time' in self.ds2.dimensions)
        self.assertEqual(10, self.ds2.dimensions['time'].size)

        # Did depth info get included?
        self.assertEqual(self.ds['siglev'][:].tolist(),
                         self.ds2['siglev'][:].tolist())

    def test_init_output_vars(self):
        tstart = Timestamp('2000-01-01')
        tslc = slice(5, 15)
        invars = (('data_node', rawcdf_extract.InputAttr.ALL),)
        self.ds2 = rawcdf_extract.init_output('dummy.nc', self.ds, tstart,
                                              tslc, self.cv, invars,
                                              diskless=True)
        rawcdf_extract.init_output_vars(self.ds2, self.ds, outprefix='copy',
                                        input_vars=invars)
        self.assertEqual(self.ds['data_node'].long_name, self.ds2['copydata_node'].long_name)
        self.assertEqual(self.ds['data_node'].dimensions, self.ds2['copydata_node'].dimensions)

    @unittest.expectedFailure
    def test_init_output_vars(self):
        tstart = Timestamp('2000-01-01')
        tslc = slice(5, 15)
        invars = (('data_node', rawcdf_extract.InputAttr.ALL),)
        self.ds2 = rawcdf_extract.init_output('dummy.nc', self.ds, tstart,
                                              tslc, self.cv, invars,
                                              diskless=True)
        self.ds2.createVariable('copydata_node', 'f4', ('time',))
        rawcdf_extract.init_output_vars(self.ds2, self.ds, outprefix='copy',
                                        input_vars=invars)

    def test_copy_data_node_all(self):
        tstart = Timestamp('2000-01-01')
        tslc = slice(5, 15)
        invars = (('data_node', rawcdf_extract.InputAttr.ALL),)
        self.ds2 = rawcdf_extract.init_output('dummy.nc', self.ds, tstart,
                                              tslc, self.cv, invars,
                                              diskless=True)
        rawcdf_extract.init_output_vars(self.ds2, self.ds, outprefix='copy',
                                        input_vars=invars)
        copied = rawcdf_extract.copy_data(self.ds, self.ds2, self.cv, tslc, 0,
                                          outprefix='copy', input_vars=invars)

        self.assertIn('copydata_node', copied)
        # The first time slice
        self.assertEqual(self.ds['data_node'][5, :, 6].tolist(),
                         self.ds2['copydata_node'][0, :, 3].tolist())
        # A middle time slice
        self.assertEqual(self.ds['data_node'][7, :, 10].tolist(),
                         self.ds2['copydata_node'][2, :, 6].tolist())
        # The last time slice
        self.assertEqual(self.ds['data_node'][14, :, 4].tolist(),
                         self.ds2['copydata_node'][9, :, 2].tolist())

    def tearDown(self):
        self.ds.close()
        if self.ds2 is not None and self.ds2.isopen():
            self.ds2.close()

if __name__ == '__main__':
    unittest.main()
