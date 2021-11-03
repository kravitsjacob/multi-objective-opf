"""Conduct analysis python script"""

import pandapower.networks
import pandas as pd
import numpy as np

from src import analysis


def main():
    # Inputs
    n_steps = 2
    inputs = analysis.input_parse()
    net = pandapower.networks.case_ieee30()
    df_abido_coef = pd.read_csv(inputs['path_to_df_abido_coef'])

    # Fuel and cooling system type
    analysis.get_fuel_cool(df_abido_coef)

    # Get emission coefficients

    # Get water use coefficients


    # Get coefficients
    df_temp = df_abido_coef.copy()
    df_temp['p_mw'] = 50.0
    df_objective_components = analysis.compute_objective_terms(df_temp, t=np.nan)
    a = 1

    net.df_coef = df_abido_coef


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