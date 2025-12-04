#!/usr/bin/env python

import unittest
import sys
import pandas as pd
# software under test
sys.path.insert(0, sys.path[0] + '/../input_files')
import ssm_write_fwinputs

class TestSsmWriteFwinputs(unittest.TestCase):

    def test_repeat_data(self):
        index = pd.MultiIndex.from_arrays([[2, 4, 6, 8, 10, 12], list('abcdef')], names=['Hours','Letters'])
        df = pd.DataFrame({'word': ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']}, index=index)

        df_1repeat = ssm_write_fwinputs.repeat_data(df)
        self.assertEqual(12, len(df_1repeat))
        self.assertFalse(df_1repeat.index.has_duplicates)
        self.assertEqual(12, df_1repeat.index[5][0])
        self.assertEqual(14, df_1repeat.index[6][0])
        self.assertEqual(24, df_1repeat.index[-1][0])

        index = pd.MultiIndex.from_arrays([[2, 2, 4, 4, 6, 6], list('ababab')], names=['Hours','Letters'])
        df = pd.DataFrame({'word': ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot']}, index=index)

        df_1repeat = ssm_write_fwinputs.repeat_data(df)
        self.assertEqual(12, len(df_1repeat))
        self.assertFalse(df_1repeat.index.has_duplicates)
        self.assertEqual(6, df_1repeat.index[5][0])
        self.assertEqual(8, df_1repeat.index[6][0])
        self.assertEqual(8, df_1repeat.index[7][0])
        self.assertEqual(12, df_1repeat.index[-2][0])
        self.assertEqual(12, df_1repeat.index[-1][0])

        df_2repeat = ssm_write_fwinputs.repeat_data(df, times=2)
        self.assertEqual(18, len(df_2repeat))
        self.assertFalse(df_2repeat.index.has_duplicates)
        self.assertEqual(6, df_2repeat.index[5][0])
        self.assertEqual(8, df_2repeat.index[6][0])
        self.assertEqual(8, df_2repeat.index[7][0])
        self.assertEqual(12, df_2repeat.index[10][0])
        self.assertEqual(12, df_2repeat.index[11][0])
        self.assertEqual(18, df_2repeat.index[-2][0])
        self.assertEqual(18, df_2repeat.index[-1][0])

if __name__ == '__main__':
    unittest.main()
