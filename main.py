"""Conduct analysis python script"""

import os

import pandapower.networks
import pandas as pd
import dask.dataframe as dd

from src import analysis


def main():
    # Inputs
    n_steps = 2
    obj_labs = ['F_cos', 'F_emit', 'F_with', 'F_con']
    inputs = analysis.input_parse()

    if not os.path.exists(inputs['path_to_df_grid_results']):
        net = pandapower.networks.case_ieee30()
        df_abido_coef = pd.read_csv(inputs['path_to_df_abido_coef'])
        df_macknick_coef = pd.read_csv(inputs['path_to_df_macknick_coef'])

        # Fuel and cooling system type
        df_coef = analysis.get_fuel_cool(df_abido_coef)

        # Get emission coefficients
        df_coef = analysis.get_emission_coef(df_coef)

        # Get water use coefficients
        df_coef = analysis.get_water_use_rate(df_coef, df_macknick_coef)

        # Assign coefficients
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

        # Solve opf for each grid entry
        ddf_grid = dd.from_pandas(df_grid, npartitions=inputs['n_tasks'])
        df_grid_results = ddf_grid.apply(
            lambda row: analysis.mo_opf(row, net),
            axis=1,
            meta=pd.DataFrame(columns=obj_labs, dtype='float64')
        ).compute(scheduler='processes')

        df_grid_results = pd.concat([df_grid, df_grid_results], axis=1)
        df_grid_results = df_grid_results.dropna()

        # Save checkpoint
        df_grid_results.to_csv(inputs['path_to_df_grid_results'], index=False)

    if not os.path.exists(inputs['path_to_df_nondom']):
        # Load required checkpoints
        df_grid_results = pd.read_csv(inputs['path_to_df_grid_results'])

        # Nondominated filter
        df_nondom = analysis.get_nondomintated(df_grid_results, objs=obj_labs)

        # Save checkpoint
        df_nondom.to_csv(inputs['path_to_df_nondom'], index=False)

    return 0


if __name__ == '__main__':
    main()
