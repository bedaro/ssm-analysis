#!/usr/bin/env python3

import unittest
from fvcom.grid import FvcomGrid
from fvcom.transect import Transect
import numpy as np

class TestTransect(unittest.TestCase):
    def setUp(self):
        # Construct a 4x4 node grid with unit coordinates
        ncoord = np.zeros((2, 16), int)
        for i in range(1, 5):
            ncoord[1,i*4-4:i*4] = i
            ncoord[0,i-1::4] = i
        nv = np.array([
            [1,5,6],
            [1,2,6],
            [2,6,7],
            [2,3,7],
            [3,4,7],
            [4,7,8],
            [5,9,10],
            [5,6,10],
            [6,10,11],
            [6,7,11],
            [7,11,12],
            [7,8,12],
            [9,10,13],
            [10,13,14],
            [10,14,15],
            [10,11,15],
            [11,15,16],
            [11,12,16]
        ]).T
        self.grid = FvcomGrid(ncoord, nv, calc=True)

    def test_create_from_array(self):
        transect = Transect(self.grid, np.array([4,3,10,9,16,15]))
        # Check border midpoints
        self.assertEqual(transect.midpoints[0,0], 2.5)
        self.assertEqual(transect.midpoints[1,0], 1)
        self.assertEqual(transect.midpoints[0,-1], 2.5)
        self.assertEqual(transect.midpoints[1,-1], 4)
        # Check some midpoints in the middle
        self.assertEqual(transect.midpoints[0,2], 2.5)
        self.assertEqual(transect.midpoints[1,2], 2)
        self.assertEqual(transect.midpoints[0,3], 2.5)
        self.assertEqual(transect.midpoints[1,3], 2.5)
        # Check some unit vectors
        self.assertAlmostEqual(transect.ns1[0,0], 2*np.sqrt(5)/5)
        self.assertAlmostEqual(transect.ns1[0,1], -np.sqrt(5)/5)
        self.assertAlmostEqual(transect.ns1[1,0], np.sqrt(2)/2)
        self.assertAlmostEqual(transect.ns1[1,1], np.sqrt(2)/2)

    def test_create_from_shortest_single(self):
        transect = Transect.shortest(self.grid, np.array([4,15]))
        self.assertEqual(transect.eles[0], 4)
        self.assertEqual(transect.eles[-1], 15)
        self.assertEqual(transect.eles[3], 9)

    def test_create_from_shortest_multi(self):
        transect1 = Transect.shortest(self.grid, np.array([4,15]))
        transect2 = Transect.shortest(self.grid, np.array([17,5]))

        transects = Transect.shortest(self.grid, np.array([[4,15],[17,5]]))
        self.assertEqual(len(transect1.eles), len(transects[0].eles))
        self.assertTrue((transect1.eles == transects[0].eles).all())
        self.assertEqual(len(transect2.eles), len(transects[1].eles))
        self.assertTrue((transect2.eles == transects[1].eles).all())

    def test_get_nodes(self):
        transect = Transect(self.grid, np.array([4,3,10,9,16,15]))
        nodes_up, nodes_down = transect.get_nodes()
        self.assertEqual(4, len(nodes_up))
        self.assertEqual(4, len(nodes_down))
        self.assertIn(7, nodes_up)
        self.assertIn(10, nodes_down)
        # Also check edge nodes
        self.assertIn(3, nodes_up)
        self.assertIn(14, nodes_down)

    def test_ele_xys(self):
        transect = Transect(self.grid, np.array([4,3,10,9,16,15]))
        xys = transect.ele_xys
        self.assertAlmostEqual(xys[0,0], 8/3)
        self.assertAlmostEqual(xys[1,0], 4/3)
        self.assertAlmostEqual(xys[0,3], 7/3)
        self.assertAlmostEqual(xys[1,3], 8/3)

    def test_distances(self):
        transect = Transect(self.grid, np.array([4,3,10,9,16,15]))
        a = transect.a
        self.assertAlmostEqual(a[0], (np.sqrt(5)+np.sqrt(2))/6)
        d = transect.center_dists()
        self.assertAlmostEqual(d[0], np.sqrt(5)/6)
        self.assertAlmostEqual(d[1], (np.sqrt(5)+2*np.sqrt(2))/6)
        self.assertAlmostEqual(d[2], (3*np.sqrt(5)+2*np.sqrt(2))/6)

if __name__ == '__main__': unittest.main()
