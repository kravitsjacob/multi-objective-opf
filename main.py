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
    net.df_coef = df_coef

    # Sample decision space
    df_gen_info = analysis.get_generator_information(net)
    df_gen_info = df_gen_info[df_gen_info['et'] != 'ext_grid']  # External grid will be solved during power flow
    df_gridspecs = pd.DataFrame(
        {
            'var': df_gen_info['bus'].astype(int).tolist(),
            'min': df_gen_info['min_p_mw'].tolist(),
            'max': df_gen_info['max_p_mw'].tolist(),
            'steps': n_steps
        }
    )
    df_grid = analysis.grid_sample(df_gridspecs)

    # solve opf
    df_results = df_grid.apply(
        lambda row: analysis.mo_opf(row, net),
        axis=1
    )
    return 0


if __name__ == '__main__':
    main()