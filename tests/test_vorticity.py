#!/usr/bin/env python3

import unittest
import sys

import numpy as np

from fvcom.grid import FvcomGrid, uniform_triangular

# software under test
sys.path.insert(0, sys.path[0] + '/..')
import vorticity

class VorticityTest(unittest.TestCase):

    def test_simple_grid_pinv(self):
        grid = uniform_triangular(sz=3)

        # dxy for element 3 should be
        # [[  1/2  -sqrt(3)/6 ]
        #  [    0   sqrt(3)/3 ]
        #  [ -1/2  -sqrt(3)/6 ]]
        # the pseudo-inverse of this is
        # [[          1           0          -1 ]
        #  [ -sqrt(3)/3  -sqrt(3)/3  -sqrt(3)/3 ]
        pinvs = vorticity.find_pinvs(grid)
        pinv_2 = [[1,0,-1],[-np.sqrt(3)/3,2*np.sqrt(3)/3,-np.sqrt(3)/3]]
        self.assertTrue(np.allclose(pinv_2, pinvs[2]))

    def test_ele_vort_small(self):
        grid = uniform_triangular(sz=3)
        pinvs = vorticity.find_pinvs(grid)
        # Test using element 3, the only non-edge element

        # Start with trivial condition: no velocity
        u = np.zeros((1, 1, grid.n))
        v = np.zeros((1, 1, grid.n))
        vort = vorticity.ele_vort(
                *vorticity.ele_velfield(grid, 2, u, v, pinvs)
            )
        self.assertEqual(0, vort[0])

        # Another trivial condition: all velocity is uniform in
        # direction and magnitude so there's zero curl
        u = np.zeros((1, 1, grid.n)) + 4
        v = np.zeros((1, 1, grid.n)) + 4
        vort = vorticity.ele_vort(
                *vorticity.ele_velfield(grid, 2, u, v, pinvs)
            )
        self.assertEqual(0, vort[0])

        # Define a simple CCW vortex with normal, unit velocities
        # around stationary element 3
        u = np.array([[[.5, .5, 0, -1]]])
        v = np.array([[[-np.sqrt(3)/2, np.sqrt(3)/2, 0, 0]]])
        vort = vorticity.ele_vort(
                *vorticity.ele_velfield(grid, 2, u, v, pinvs)
            )
        # What should the solution be?
        # The linear model for the velocity field around element 3
        # is
        # U_3(x') = A_3 x'
        # 
        # and the least squares setup for A_3 given the element
        # data is
        #
        # [ -.5  -sqrt(3)/6 ] [ a_3^u  a_3^v ]   [ .5  -sqrt(3)/2 ]
        # [  .5  -sqrt(3)/6 ] [ b_3^u  b_3^v ] = [ .5   sqrt(3)/2 ]
        # [   0   sqrt(3)/3 ]                    [ -1           0 ]
        #
        # The least-squares solution for A_3:
        # 
        # A_3^T = [        0  sqrt(3) ]
        #         [ -sqrt(3)  0       ]
        # vorticity for element 3, then, is 2*sqrt(3)
        self.assertAlmostEqual(2*np.sqrt(3), vort[0,0])

    def test_ele_vort_edge(self):
        grid = uniform_triangular(sz=4)
        pinvs = vorticity.find_pinvs(grid)

        # Consider element #2 which has a single edge, and
        # has neighbor elements 4 and 5
        # Create edge shear: u = 1 at element 2, 2 at elements 4,5
        # No v component
        u = np.zeros((1, 1, grid.n))
        v = np.zeros((1, 1, grid.n))
        u[0,0,1] = 1
        u[0,0,3] = 2
        u[0,0,4] = 2
        vort = vorticity.ele_vort(
                *vorticity.ele_velfield(grid, 1, u, v, pinvs)
            )
        # The least-squares solution for this field is pretty easy:
        # 
        # A_2^T = [         0  0 ]
        #         [ 2*sqrt(3)  0 ]
        # so vorticity is -2*sqrt(3)
        self.assertAlmostEqual(-2*np.sqrt(3), vort[0,0])

if __name__ == '__main__':
    unittest.main()
