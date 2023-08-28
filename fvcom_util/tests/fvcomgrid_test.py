#!/usr/bin/env python3

import unittest
from fvcom.grid import *
import numpy as np

class TestFvcomGrid(unittest.TestCase):
    def test_init(self):
        # 2-D only
        grid = FvcomGrid(np.zeros((2, 20), int),
                np.zeros((3, 30), int))
        self.assertEqual(grid.m, 20)
        self.assertEqual(grid.n, 30)
        # Include depth
        grid2 = FvcomGrid(np.zeros((3, 20), int),
                np.zeros((3, 30), int))
        self.assertEqual(grid.m, 20)
        self.assertEqual(grid.n, 30)

    @unittest.expectedFailure
    def test_integrity_check(self):
        grid = FvcomGrid(nv=np.zeros((3, 20), int), ncoord=np.zeros((5, 30), int))

    @unittest.expectedFailure
    def test_calc_needed(self):
        grid = FvcomGrid(np.zeros((2, 20), int),
                np.zeros((3, 30), int))
        nbe = grid.nbe
        elcoord = grid.elcoord

    def test_nbe(self):
        m = 10
        nv = np.array([
            [1,2,3],
            [2,3,4],
            [2,4,5],
            [4,5,6],
            [2,5,7],
            [5,7,8],
            [7,8,9],
            [7,8,10],
            [2,7,10],
            [1,2,10]]).T
        grid = FvcomGrid(ncoord=np.zeros((2, m), int), nv=nv, calc=True)
        nbe = grid.nbe
        self.assertEqual((3, grid.n), nbe.shape)
        # First node
        self.assertTrue((nbe[:,0] == 2).any())
        self.assertTrue((nbe[:,0] == 10).any())
        self.assertTrue((nbe[:,0] == 0).any())

        # Fifth node
        self.assertTrue((nbe[:,4] == 3).any())
        self.assertTrue((nbe[:,4] == 9).any())
        self.assertTrue((nbe[:,4] == 6).any())

    def test_element_centers(self):
        nv = np.array([
            [1,2,3],
            [2,3,4],
            [2,4,5],
            [4,5,6],
            [2,5,7],
            [5,7,8],
            [7,8,9],
            [7,8,10],
            [2,7,10],
            [1,2,10]]).T
        ncoord = np.array([
            [1, 0],
            [1, 2],
            [0, 1],
            [0, 3],
            [1, 4],
            [0, 5],
            [2, 3],
            [2, 5],
            [3, 3],
            [2, 1]]).T
        grid = FvcomGrid(ncoord, nv, calc=True)
        elcoord = grid.elcoord
        self.assertEqual((2, grid.n), elcoord.shape)
        self.assertEqual(2/3, elcoord[0, 0])
        self.assertEqual(1, elcoord[1, 0])
        self.assertEqual(4/3, elcoord[0, 4])
        self.assertEqual(3, elcoord[1, 4])

    def test_uniform_triangular(self):
        grid1 = uniform_triangular(sz=3)
        self.assertEqual(6, grid1.m)
        self.assertEqual(4, grid1.n)
        self.assertTrue(np.allclose([1, 0], grid1.ncoord[:,1]))
        self.assertTrue(np.allclose([1.5, np.sqrt(3)/2], grid1.ncoord[:,4]))
        self.assertTrue(np.allclose([.5, np.sqrt(3)/6], grid1.elcoord[:,0]))

        grid2 = uniform_triangular(sz=6)
        self.assertEqual(21, grid2.m)
        self.assertEqual(25, grid2.n)
        self.assertTrue(np.allclose([3.5, np.sqrt(3)/2], grid2.ncoord[:,9]))
        self.assertTrue(np.allclose([3, 2*np.sqrt(3)/2], grid2.ncoord[:,13]))


if __name__ == '__main__':
    unittest.main()
