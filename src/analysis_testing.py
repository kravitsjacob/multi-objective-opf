"""Testing library for analysis.py"""

import unittest

import pandas as pd
import pandas.testing as pd_testing
import pandapower.networks

import analysis


class AnalysisLib(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_get_generator_information(self):
        # Setup
        net = pandapower.networks.case_ieee30()

        # Run
        df_gen_info = analysis.get_generator_information(net)

        # Check
        test_bus = net.gen.iloc[0]['bus']
        test_power_max = net.gen.iloc[0]['max_p_mw']
        expected_power_max = df_gen_info[df_gen_info['bus'] == test_bus]['max_p_mw'][0]

        self.assertEqual(test_power_max, expected_power_max)

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
