"""Testing library for analysis.py"""

import unittest

import pandas as pd
import pandas.testing as pd_testing

import analysis


class AnalysisLib(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_grid_sample(self):
        # Setup
        df_test = pd.DataFrame({'var': ['A', 'B'], 'min': [0, 1], 'max': [1, 2], 'steps': [2, 3]})
        df_expect = pd.DataFrame(
            {'A': [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
             'B': [1.0, 1.5, 2.0, 1.0, 1.5, 2.0]}
        )

        # Run
        df_result = analysis.grid_sample(df_test)

        # Check equal
        self.assertEqual(df_expect, df_result)


if __name__ == '__main__':
    unittest.main()