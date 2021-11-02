"""Conduct analysis python script"""

import pandapower.networks
import pandas as pd

from src import analysis


def main():
    # Inputs
    n_steps = 2
    inputs = analysis.input_parse()
    df_coef = pd.read_csv(inputs['path_to_df_coef'])
    net = pandapower.networks.case_ieee30()

    # Get coefficients TODO
    a = 1

    # Sample decision space
    df_gen_info = analysis.get_generator_information(net)
    df_gen_info = df_gen_info[df_gen_info['et'] != 'ext_grid']  # External grid will be solved during power flow
    df_gridspecs = pd.DataFrame(
        {
            'var': ('bus ' + df_gen_info['bus'].astype(str)).tolist(),
            'min': df_gen_info['min_p_mw'].tolist(),
            'max': df_gen_info['max_p_mw'].tolist(),
            'steps': n_steps
        }
    )
    df_grid = analysis.grid_sample(df_gridspecs)

    # solve opf

    return 0


if __name__ == '__main__':
    main()