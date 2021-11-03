"""Testing library for analysis.py"""

import unittest

import numpy as np
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

    # Setup
    net = pandapower.networks.case_ieee30()
    dict_coef = {
        'element': {0: 0.0, 1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0},
        'et': {0: 'res_ext_grid', 1: 'res_gen', 2: 'res_gen', 3: 'res_gen', 4: 'res_gen', 5: 'res_gen'},
        'bus': {0: 0, 1: 1, 2: 4, 3: 7, 4: 10, 5: 12},
        'a': {0: 10, 1: 10, 2: 20, 3: 10, 4: 20, 5: 10},
        'b': {0: 200, 1: 150, 2: 180, 3: 100, 4: 180, 5: 150},
        'c': {0: 100, 1: 120, 2: 40, 3: 60, 4: 40, 5: 100},
        'alpha': {0: 4.091, 1: 2.543, 2: 4.258, 3: 5.426, 4: 4.258, 5: 6.131},
        'beta_emit': {0: -5.554, 1: -6.047, 2: -5.094, 3: -3.55, 4: -5.094, 5: -5.555},
        'gamma': {0: 6.49, 1: 5.638, 2: 4.586, 3: 3.38, 4: 4.586, 5: 5.151},
        'xi': {0: 0.0002, 1: 0.0005, 2: 1e-06, 3: 0.002, 4: 1e-06, 5: 1e-05},
        'lambda': {0: 2.857, 1: 3.333, 2: 8.0, 3: 2.0, 4: 8.0, 5: 6.667},
        'beta_with': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        'beta_con': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    }
    net.df_coef = pd.DataFrame(dict_coef)

    def test_abido_2003(self, net=net):
        """
        Can't use paper as direct comparison due to different formulation, can compare orders of magnitude
        @article{abido_novel_2003,
            title = {A novel multiobjective evolutionary algorithm for environmental/economic power dispatch},
            volume = {65},
            issn = {0378-7796},
            url = {https://dx.doi.org/10.1016/s0378-7796(02)00221-3},
            doi = {10.1016/s0378-7796(02)00221-3},
            number = {1},
            journal = {Electric Power Systems Research},
            author = {Abido, M.A.},
            year = {2003},
            note = {Publisher: Elsevier BV},
            pages = {71--81},
        }
        """
        # Run
        ser_decisions = pd.Series({1: 10.0, 4: 29.0, 7: 52.0, 10: 100.0, 12: 52.0})
        ser_obj = analysis.mo_opf(ser_decisions, net)

        # Test
        cos_diff = abs(600.0 - ser_obj['F_cos'])
        emit_diff = abs(0.22 - ser_obj['F_emit'])
        self.assertLess(emit_diff, 0.1)
        self.assertLess(cos_diff, 50)

    def test_mo_opf_limits(self, net=net):
        # Run
        ser_decisions = pd.Series({1: 100.0, 4: 100.0, 7: 100.0, 10: 100.0, 12: 100.0})
        ser_obj = analysis.mo_opf(ser_decisions, net)

        # Check
        self.assertTrue(np.isnan(ser_obj['F_cos']))

    def test_get_nondomintated_default(self):
        # Setup
        df_test = pd.DataFrame({'A': [3, 2, 2], 'B': [3, 1, 2], 'C': [3, 2, 1]})
        df_expect = pd.DataFrame({'A': [2, 2], 'B': [1, 2], 'C': [2, 1]})

        # Run
        df_result = analysis.get_nondomintated(df_test, objs=['A', 'B', 'C'])

        # Check equal
        self.assertEqual(df_expect, df_result)

    def test_get_nondomintated_max(self):
        # Setup
        df_test = pd.DataFrame({'A': [2, 2, 1], 'B': [3, 3, 1], 'C': [3, 2, 1]})
        df_expect = pd.DataFrame({'A': [2, 1], 'B': [3, 1], 'C': [3, 1]})

        # Run
        df_result = analysis.get_nondomintated(df_test, objs=['A', 'B', 'C'], max_objs=['C'])

        # Check equal
        self.assertEqual(df_expect, df_result)


if __name__ == '__main__':
    unittest.main()
