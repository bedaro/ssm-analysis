#!/usr/bin/env python3

import unittest
from fvcom.grid import FvcomGrid
from fvcom.transect import Transect
from fvcom.control_volume import ControlVolume
import numpy as np

class ControlVolumeTest(unittest.TestCase):
    def setUp(self):
        # Construct a simple splitting channel
        ncoord = np.array([
            [2,0], [4,0], [3,1], [2,2], [4,2],
            [1,3], [3,3], [5,3],
            [0,4], [2,4], [4,4], [6,4],
            [1,5], [3,5], [5,5],
            [2,6], [4,6], [3,7], [2,8], [4,8]
        ]).T
        nv = np.array([
            [1,2,3], [1,3,4], [2,3,5], [3,4,5],
            [4,6,7], [4,5,7], [5,7,8],
            [6,9,10], [6,7,10], [7,8,11], [8,11,12],
            [9,10,13], [10,13,14], [11,14,15], [11,12,15],
            [13,14,16], [14,16,17], [14,15,17],
            [16,17,18], [16,18,19], [17,18,20], [18,19,20]
        ]).T
        self.grid = FvcomGrid(ncoord, nv, calc=True)

    def test_nodes(self):
        cv = ControlVolume(grid=self.grid, nodes={1, 2, 3})
        self.assertAlmostEqual(cv.area, 8/3)
        cv2 = cv - {2}
        self.assertEqual(cv2.nodes, {1,3})

    def test_single_transect(self):
        """Make a single transect across the unified channel"""
        transect = Transect(self.grid, np.array([2,4,3]))
        cv = ControlVolume.from_transects([transect])
        self.assertEqual(cv.nodes, {1,2,3})
        self.assertEqual(cv.transect_directions(), [True])
        self.assertAlmostEqual(cv.area, 8/3)

    def test_two_transects_closed(self):
        """Make two transects along one side of the channel"""
        tr1 = Transect(self.grid, np.array([14, 18]))
        tr2 = Transect(self.grid, np.array([10, 7]))
        cv = ControlVolume.from_transects([tr1, tr2])
        self.assertEqual(cv.nodes, {8,11,12,15})
        self.assertEqual(cv.transect_directions(), [True, False])
        self.assertEqual(cv.area, 4)

    def test_two_transects_open(self):
        """Make two transects, one on each side of the channel"""
        tr1 = Transect(self.grid, np.array([8, 9]))
        tr2 = Transect(self.grid, np.array([14, 15]))
        cv = ControlVolume.from_transects([tr1, tr2])
        self.assertEqual(cv.nodes, {1,2,3,4,5,6,7,8,11,12})
        self.assertEqual(cv.transect_directions(), [True, True])
        self.assertEqual(cv.area, 11)

    def test_three_transects_closed(self):
        """Make three transects enclosing one channel split"""
        tr1 = Transect(self.grid, np.array([2,4,3]))
        tr2 = Transect(self.grid, np.array([8,9]))
        tr3 = Transect(self.grid, np.array([10,11]))
        cv = ControlVolume.from_transects([tr1, tr2, tr3])
        self.assertEqual(cv.nodes, {4,5,6,7,8})
        self.assertEqual(cv.transect_directions(), [False, True, True])
        self.assertAlmostEqual(cv.area, 19/3)

if __name__ == '__main__': unittest.main()
