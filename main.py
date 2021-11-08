"""Conduct analysis python script"""

import os

import pandapower.networks
import pandas as pd
import dask.dataframe as dd
import numpy as np

from src import analysis
from src import viz


def main():
    # Inputs
    dec_labs = ['1', '4', '7', '10', '12', '0']
    obj_labs = ['F_cos', 'F_emit', 'F_with', 'F_con']
    obj_epsi = [10.0, 0.01, 1000.00, 1000.00]
    dec_labs_pretty = ['Gen ' + i + ' (MW)' for i in dec_labs]
    obj_labs_pretty = [
        'Cost ($/hr)',
        'Emissions (ton/hr)',
        'Withdrawal (gal/hr)',
        'Consumption (gal/hr)'
    ]
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
        df_gen_info = \
            df_gen_info[df_gen_info['et'] != 'ext_grid']  # External grid
        # solved during power flow
        df_gridspecs = pd.DataFrame(
            {
                'var': df_gen_info['bus'].astype(int).tolist(),
                'min': df_gen_info['min_p_mw'].tolist(),
                'max': df_gen_info['max_p_mw'].tolist(),
                'steps': inputs['n_steps']
            }
        )
        df_grid = analysis.grid_sample(df_gridspecs)
        print(f'Number of searches: {len(df_grid)}')

        # Solve opf for each grid entry
        ddf_grid = dd.from_pandas(df_grid, npartitions=inputs['n_tasks'])
        df_grid_results = ddf_grid.apply(
            lambda row: analysis.mo_opf(row, net),
            axis=1,
            meta=pd.DataFrame(columns=[0]+obj_labs, dtype='float64')
        ).compute(scheduler='processes')

        df_grid_results = pd.concat([df_grid, df_grid_results], axis=1)

        # Save checkpoint
        df_grid_results.to_csv(inputs['path_to_df_grid_results'], index=False)

    if not os.path.exists(inputs['path_to_df_nondom']):
        # Load required checkpoints
        df_grid_results = pd.read_csv(inputs['path_to_df_grid_results'])

        # Drop na
        df_grid_results = df_grid_results.dropna()

        # Nondominated filter
        df_nondom = analysis.get_epsilon_nondomintated(
            df_grid_results,
            objs=obj_labs,
            epsilons=obj_epsi
        )

        # Save checkpoint
        df_nondom.to_csv(inputs['path_to_df_nondom'], index=False)

    if not os.path.exists(inputs['path_to_nondom_objectives_viz']):
        # Load required checkpoints
        df_nondom = pd.read_csv(inputs['path_to_df_nondom'])

        # Formatting
        df_nondom = df_nondom.rename(
            dict(
                zip(dec_labs+obj_labs, dec_labs_pretty+obj_labs_pretty)),
            axis=1
        )
        df_nondom['Color Index'] = df_nondom['Cost ($/hr)']
        df_nondom = viz.set_color_gradient(
            df_nondom, colormap='viridis', label='Cost ($/hr)'
        )

        # Plot Objectives
        ticks = [np.arange(0, 10000, 50),
                 np.arange(0.0, 1.0, 0.02),
                 np.arange(0, 5100000, 500000),
                 np.arange(0, 59000, 2000)]
        limits = [[590, 1050], [0.17, 0.40], [-1, 5100000], [37000, 59000]]
        viz.static_parallel(
            df=df_nondom,
            columns=obj_labs_pretty,
            plot_colorbar=True,
            subplots_adjust_args={
                'left': 0.10,
                'bottom': 0.20,
                'right': 0.80,
                'top': 0.95,
                'wspace': 0.0,
                'hspace': 0.0
            },
            explicit_ticks=ticks,
            limits=limits
        ).savefig(
            inputs['path_to_nondom_objectives_viz']
        )

        # Plot Decisions
        viz.static_parallel(
            df=df_nondom,
            columns=dec_labs_pretty,
            plot_colorbar=True,
            subplots_adjust_args={
                'left': 0.10,
                'bottom': 0.20,
                'right': 0.80,
                'top': 0.95,
                'wspace': 0.0,
                'hspace': 0.0
            }
        ).savefig(
            inputs['path_to_nondom_decisions_viz']
        )

    if not os.path.exists(inputs['path_to_objective_correlation_viz']):
        # Load required checkpoints
        df_nondom = pd.read_csv(inputs['path_to_df_nondom'])

        # Formatting
        df_nondom = df_nondom.rename(
            dict(
                zip(dec_labs + obj_labs, dec_labs_pretty + obj_labs_pretty)),
            axis=1
        )

        viz.correlation_heatmap(
            df_nondom[obj_labs_pretty]
        ).savefig(inputs['path_to_objective_correlation_viz'])

    return 0


if __name__ == '__main__':
    main()
