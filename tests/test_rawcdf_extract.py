#!/usr/bin/env python3

import unittest
import sys

from netCDF4 import Dataset
import numpy as np
from pandas import Timestamp

from fvcom.grid import FvcomGrid
from fvcom.control_volume import ControlVolume

# software under test
sys.path.insert(0, sys.path[0] + '/..')
import rawcdf_extract

# Test cases needed:
# - copy_data
#   element-based data
#   using entire time slice
#   using multiple partial time slices to get entire span

class TestRawcdfExtract(unittest.TestCase):
    def setUp(self):
        # Read the fake model output
        self.ds = Dataset('data/test_output.nc')
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
