#!/usr/bin/env python3

import unittest
import sys
import numpy as np
import pandas as pd

# software under test
sys.path.insert(0, sys.path[0] + '/../input_files')
import ssm_adjust_rivers

class TestSsmAdjustRivers(unittest.TestCase):
    def test_adjust_stream_simple_index(self):
        adj = pd.Series([0.1, 1, 0.5])
        data = pd.DataFrame({
            'discharge': [5, 2, 3],
            'temp': [3, 2, 4],
            'doxg': [3, 2, 4],
            'dic': [3, 2, 4],
            'talk': [3, 2, 4],
            'somethingelse': [1, 3, 4]
        })
        res_keep = ssm_adjust_rivers.adjust_stream(data, adj, ssm_adjust_rivers.ConcMethod.KEEP)
        self.assertTrue((res_keep['discharge'] == [0.5, 2, 1.5]).all())
        self.assertTrue((res_keep['temp'] == data['temp']).all())
        self.assertTrue((res_keep['somethingelse'] == data['somethingelse']).all())

        res_constload = ssm_adjust_rivers.adjust_stream(data, adj, ssm_adjust_rivers.ConcMethod.CONSTLOAD)
        self.assertTrue((res_constload['discharge'] == [0.5, 2, 1.5]).all())
        self.assertTrue((res_constload['temp'] == data['temp']).all())
        self.assertTrue((res_constload['somethingelse'] == [10, 3, 8]).all())

    def test_adjust_stream_multiindex(self):
        dates = pd.date_range('2006.05.01', '2006.05.03', name='Date')
        # Only values for a single node are passed at once
        nodes = [1]
        idx = pd.MultiIndex.from_product([dates, nodes], names=['Date','Node'])

        # The adjustment Series uses a date range index only
        adj = pd.Series([0.25, 0.3, 0.1], index=dates)

        data = pd.DataFrame({
            'discharge': [5, 2, 3],
            'temp': [3, 2, 4],
            'doxg': [3, 2, 4],
            'dic': [3, 2, 4],
            'talk': [3, 2, 4],
            'somethingelse': [1, 3, 4]
        }, index=idx)
        res_keep = ssm_adjust_rivers.adjust_stream(data, adj, ssm_adjust_rivers.ConcMethod.KEEP)
        self.assertAlmostEqual((res_keep['discharge'] - [1.25, 0.6, 0.3]).max(), 0)
        self.assertTrue((res_keep['temp'] == data['temp']).all())
        self.assertTrue((res_keep['somethingelse'] == data['somethingelse']).all())

        res_constload = ssm_adjust_rivers.adjust_stream(data, adj, ssm_adjust_rivers.ConcMethod.CONSTLOAD)
        self.assertAlmostEqual((res_constload['discharge'] - [1.25, 0.6, 0.3]).max(), 0)
        self.assertTrue((res_constload['temp'] == data['temp']).all())
        self.assertAlmostEqual((res_constload['somethingelse'] - [4, 10, 40]).max(), 0)

        # Try assigning to a larger DataFrame
        allnodes = [1, 2, 3]
        idx2 = pd.MultiIndex.from_product([dates, allnodes], names=['Date','Node'])
        cs = [3, 2, 4, 6, 9, 3, 4, 6, 3]
        data2 = pd.DataFrame({
            'discharge': [5, 2, 3, 6, 4, 2, 1, 3, 9],
            'temp': cs,
            'doxg': cs,
            'dic': cs,
            'talk': cs,
            'somethingelse': cs
        }, dtype=np.float64, index=idx2)
        data2.loc[(slice(None), 1), :] = ssm_adjust_rivers.adjust_stream(data2.loc[(slice(None), 1), :], adj, ssm_adjust_rivers.ConcMethod.KEEP)
        self.assertAlmostEqual((data2['discharge'] - [1.25, 2, 3, 1.8, 4, 2, .1, 3, 9]).max(), 0)
        self.assertTrue((data2['temp'] == cs).all())
        self.assertTrue((data2['somethingelse'] == cs).all())

        data2.loc[(slice(None), 2), :] = ssm_adjust_rivers.adjust_stream(data2.loc[(slice(None), 2), :], adj, ssm_adjust_rivers.ConcMethod.CONSTLOAD)
        self.assertAlmostEqual((data2['discharge'] - [1.25, .5, 3, 1.8, 1.2, 2, .1, .3, 9]).max(), 0)
        self.assertTrue((data2['temp'] == cs).all())
        self.assertAlmostEqual((data2['somethingelse'] - [3, 8, 4, 6, 30, 3, 4, 60, 3]).max(), 0)

    def test_build_climatology(self):
        dates = pd.date_range('2005.01.01', '2006.12.31', freq='D')
        data = pd.DataFrame({
            'discharge': [13, 2, 18, 11, 12, 2, 4, 13, 8, 6, 6,
                19, 11, 14, 6, 10, 0, 11, 16, 6, 1, 12, 13, 18,
                16, 19, 0, 6, 11, 11, 3, 2, 7, 7, 10, 2, 17, 12,
                6, 9, 9, 6, 0, 7, 1, 6, 0, 1, 8, 6, 5, 14, 16,
                19, 7, 17, 11, 3, 4, 5, 1, 12, 9, 11, 7, 15, 6,
                6, 9, 9, 6, 8, 4, 5, 16, 2, 2, 13, 4, 3, 10, 0,
                12, 17, 15, 8, 7, 9, 19, 0, 6, 3, 14, 11, 9, 9,
                8, 3, 9, 0, 8, 13, 7, 10, 0, 19, 13, 9, 4, 4, 3,
                6, 15, 6, 11, 1, 16, 4, 15, 10, 6, 3, 16, 7, 10,
                4, 15, 0, 10, 11, 7, 8, 8, 6, 9, 19, 4, 10, 13,
                6, 10, 16, 5, 7, 6, 10, 19, 11, 18, 0, 19, 5, 4,
                3, 13, 6, 8, 0, 5, 11, 13, 10, 8, 1, 13, 6, 17,
                11, 5, 16, 13, 3, 0, 8, 4, 4, 2, 17, 0, 16, 3,
                9, 11, 2, 15, 1, 0, 9, 5, 8, 2, 12, 2, 9, 16,
                10, 13, 14, 3, 16, 17, 13, 2, 15, 17, 10, 16,
                12, 16, 8, 11, 8, 18, 3, 4, 8, 5, 6, 19, 16, 12,
                8, 9, 0, 4, 13, 15, 8, 1, 17, 0, 14, 2, 7, 16,
                17, 2, 14, 3, 4, 5, 12, 12, 14, 17, 17, 19, 4,
                17, 4, 19, 0, 2, 12, 2, 16, 0, 17, 2, 16, 2, 2,
                10, 3, 2, 9, 1, 19, 15, 14, 7, 18, 18, 11, 3,
                16, 14, 13, 3, 13, 18, 16, 15, 15, 11, 15, 19,
                18, 11, 8, 2, 4, 0, 17, 13, 12, 6, 1, 12, 19,
                19, 14, 6, 2, 7, 16, 17, 15, 3, 12, 19, 0, 2,
                13, 11, 10, 13, 10, 4, 18, 0, 19, 15, 10, 0, 5,
                9, 2, 6, 2, 17, 15, 14, 9, 7, 11, 18, 8, 15, 5,
                3, 0, 9, 16, 3, 0, 16, 11, 2, 13, 14, 19, 10,
                14, 14, 2, 18, 15, 1, 19, 3, 1, 18, 14, 1, 7,
                11, 1, 18, 18, 15, 10, 6, 6, 12, 4, 18, 4, 10,
                10, 8, 1, 12, 12, 18, 13, 10, 11, 8, 7, 1, 12,
                2, 6, 9, 1, 7, 9, 14, 5, 18, 14, 10, 16, 12,
                14, 17, 12, 11, 16, 16, 11, 4, 14, 13, 16, 14,
                14, 4, 15, 14, 18, 4, 9, 2, 3, 13, 0, 14, 11, 0,
                2, 3, 17, 7, 5, 10, 9, 18, 16, 8, 18, 7, 8, 19,
                14, 16, 0, 12, 12, 1, 16, 0, 7, 14, 8, 3, 10, 8,
                13, 16, 19, 6, 17, 12, 5, 5, 18, 4, 1, 8, 2, 6,
                18, 0, 9, 2, 14, 16, 16, 0, 14, 8, 4, 6, 0, 1,
                7, 3, 11, 10, 7, 7, 7, 0, 1, 1, 19, 15, 16, 5,
                14, 6, 12, 1, 5, 6, 11, 5, 17, 6, 4, 7, 11, 4,
                10, 18, 18, 14, 19, 18, 16, 1, 18, 5, 11, 16,
                15, 15, 2, 10, 7, 12, 12, 11, 5, 18, 15, 2, 5,
                0, 6, 17, 6, 15, 19, 2, 15, 6, 18, 13, 0, 14,
                19, 8, 13, 9, 1, 13, 8, 12, 17, 3, 13, 10, 13,
                10, 1, 1, 15, 19, 15, 11, 17, 12, 12, 13, 15,
                19, 5, 13, 0, 19, 17, 10, 15, 0, 18, 14, 18, 13,
                19, 11, 4, 12, 18, 8, 5, 14, 16, 4, 15, 12, 19,
                4, 8, 16, 5, 8, 5, 1, 2, 1, 9, 6, 15, 18, 8, 12,
                14, 2, 16, 7, 6, 5, 4, 17, 15, 14, 1, 11, 15,
                17, 6, 0, 17, 19, 19, 1, 5, 3, 3, 5, 14, 17, 17,
                6, 17, 8, 13, 1, 13, 13, 2, 12, 0, 13, 10, 3, 9,
                16, 9, 8, 17, 17, 2, 3, 4, 17, 4, 1, 8, 13, 17,
                18, 7, 4, 13, 4, 15, 3, 0, 15, 3, 10, 6, 14, 11,
                0, 8, 19, 14, 12, 17, 2, 7, 18, 5, 6, 1, 6, 11,
                6, 13, 8, 11, 2, 12, 2, 17, 4, 8, 3, 11, 11, 11,
                1, 3, 18, 2, 9, 15, 4, 6, 10, 6, 8, 18, 8, 10]
        }, index=dates)
        
        c = ssm_adjust_rivers.build_climatology(data)
        self.assertEqual(len(c), 365)
        self.assertAlmostEqual(c['discharge'].iloc[92], 9.16666667)

if __name__ == '__main__':
    unittest.main()
