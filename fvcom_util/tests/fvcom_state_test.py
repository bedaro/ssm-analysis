#!/usr/bin/env python3

import unittest
from tempfile import NamedTemporaryFile
import os
import filecmp
import numpy as np
import numpy.testing as nptest

from fvcom.grid import FvcomGrid
from fvcom.state import FvcomState

class TestFvcomState(unittest.TestCase):

    def setUp(self):
        self.source_file = 'testdata/re_000001.dat'
        grid = FvcomGrid.from_gridfile("testdata/ssm_grd.dat")
        # All optional features were off when this restart file was generated
        self.fvcom_state = FvcomState.read_restart_file(self.source_file, grid, 11)

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
        m = 25
        z = 5
        # Element which contains this node as a vertex: just pick one
        n = (self.fvcom_state.grid.nv[:] == m).any(axis=0).nonzero()[0][0]
        # Copy the original temperature array and reset it but with one value
        # changed
        t1 = self.fvcom_state.t1.copy(order='K')
        old_t1 = t1[z,m]
        old_t = self.fvcom_state.t[z,n]
        old_rho1 = self.fvcom_state.rho1[z,m]
        old_rho = self.fvcom_state.rho[z,n]
        # Change the temperature
        t1[z,m] += 2
        # Invoke the object's setter by reassigning t1
        self.fvcom_state.t1 = t1

        self.assertEqual(self.fvcom_state.t1[z,m], old_t1 + 2)
        # Test that density got updated
        self.assertNotEqual(self.fvcom_state.rho1[z,m], old_rho1)
        # Test that the element fields got updated
        self.assertNotEqual(self.fvcom_state.t[z,n], old_t)
        self.assertNotEqual(self.fvcom_state.rho[z,n], old_rho)

if __name__ == '__main__':
    unittest.main()
