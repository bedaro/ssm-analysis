#!/usr/bin/env python3
# FIXME shorten test time by building a smaller grid and use FVCOM to
# generate a restart file for it

import unittest
from tempfile import NamedTemporaryFile
import sys
import os
import filecmp
import numpy as np
import numpy.testing as nptest

from fvcom.grid import FvcomGrid
from fvcom.depth import DepthCoordinate
from fvcom.state import FvcomState

class TestFvcomState(unittest.TestCase):

    def setUp(self):
        self.source_file = sys.path[0] + '/testdata/re_000001.dat'
        grid = FvcomGrid.from_gridfile(sys.path[0] + "/testdata/ssm_grd.dat")
        dcoord = DepthCoordinate.from_asym_sigma(11, grid)
        # All optional features were off when this restart file was generated
        self.fvcom_state = FvcomState.read_restart_file(self.source_file, grid, dcoord)

    def test_rewrite(self):
        with NamedTemporaryFile() as f:
            self.fvcom_state.to_restart_file(f.name)
            orig_size = os.stat(self.source_file).st_size
            new_size = os.stat(f.name).st_size
            self.assertEqual(orig_size, new_size,
                    f'Rewritten file size {new_size} differs from original size {orig_size}')
            self.assertTrue(filecmp.cmp(self.source_file, f.name),
                    "Rewritten file does not match source")

    def test_dens2(self):
        old_rho1 = self.fvcom_state.rho1
        self.fvcom_state.dens2()
        nptest.assert_allclose(old_rho1, self.fvcom_state.rho1, rtol=5e-3)

    def test_n2e3d(self):
        s = self.fvcom_state.s
        s1 = self.fvcom_state.s1
        nptest.assert_allclose(s, self.fvcom_state.n2e3d(s1), rtol=1e-3)

    def test_update(self):
        # Test node (1-based)
        m = 35
        z = 5
        # Element indices (1-based) which contain this node as a vertex
        ns = (self.fvcom_state.grid.nv[:] == m).any(axis=0).nonzero()[0] + 1
        # Copy the original temperature array and reset it but with one value
        # changed
        t1 = self.fvcom_state.t1.copy(order='K')
        old_t1 = t1[z,m-1]
        t = self.fvcom_state.t.copy(order='K')
        old_ts = self.fvcom_state.t[z,ns]
        old_rho1 = self.fvcom_state.rho1[z,m-1]
        old_rhos = self.fvcom_state.rho[z,ns]
        # Change the temperature
        t1[z,m-1] += 2
        # Invoke the object's setter by reassigning t1
        self.fvcom_state.t1 = t1

        self.assertAlmostEqual(self.fvcom_state.t1[z,m-1], old_t1 + 2, places=5)
        # Test that density got updated
        self.assertNotAlmostEqual(self.fvcom_state.rho1[z,m-1], old_rho1, places=5)
        # Test that the element fields got updated
        for i,n in enumerate(ns):
            self.assertNotAlmostEqual(old_ts[i], self.fvcom_state.t[z,n], places=5,
                                      msg=f'Element {i} temp did not get updated as expected')
            self.assertNotAlmostEqual(old_rhos[i], self.fvcom_state.rho[z,n], places=5,
                                      msg=f'Element {i} dens not get updated as expected')

if __name__ == '__main__':
    unittest.main()
