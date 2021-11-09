"""Source code for analysis"""

import os
import configparser
import argparse
import itertools
import copy

import numpy as np
import pandas as pd
import pandapower as pp
import pymoo.util.nds.efficient_non_dominated_sort as ends
from pareto import pareto as pto


def input_parse(path_to_config=None):
    """

    Parameters
    ----------
    path_to_config: str
        Path to config file (gets preference over command line)

    Returns
    -------
    inputs: dict
        Dictionary of inputs
    """
    # Local vars
    config_inputs = configparser.ConfigParser()
    argparse_inputs = argparse.ArgumentParser()

    if path_to_config:
        config_inputs.read(path_to_config)
    else:
        # Command line arguments (these get priority)
        argparse_inputs.add_argument(
            '-c',
            '--config_file',
            type=str,
            action='store',
            help='Path to configuration file',
            required=True
        )
        argparse_inputs.add_argument(
            '-nt',
            '--n_tasks',
            type=int,
            default=2,
            action='store',
            help='Number of tasks for parallelization',
            required=False
        )
        argparse_inputs.add_argument(
            '-ns',
            '--n_steps',
            type=int,
            default=10,
            action='store',
            help='Number of steps for grid search',
            required=False
        )

        # Parse arguments
        argparse_inputs = argparse_inputs.parse_args()

        # Parse config file
        config_inputs.read(argparse_inputs.config_file)

    # Store inputs
    path_to_data = config_inputs['MAIN IO']['data']
    inputs = {
        'path_to_data': path_to_data,
        'path_to_df_abido_coef': os.path.join(
            path_to_data,
            config_inputs['INPUT']['path_to_df_abido_coef']
        ),
        'path_to_df_macknick_coef': os.path.join(
            path_to_data,
            config_inputs['INPUT']['path_to_df_macknick_coef']
        ),
        'path_to_df_grid_results': os.path.join(
            path_to_data,
            config_inputs['GENERATED_FILES']['path_to_df_grid_results']
        ),
        'path_to_df_nondom': os.path.join(
            path_to_data,
            config_inputs['GENERATED_FILES']['path_to_df_nondom']
        ),
        'n_tasks': argparse_inputs.n_tasks,
        'n_steps': argparse_inputs.n_steps,
        'path_to_nondom_objectives_viz': os.path.join(
            path_to_data,
            config_inputs['FIGURES']['path_to_nondom_objectives_viz']
        ),
        'path_to_nondom_decisions_viz': os.path.join(
            path_to_data,
            config_inputs['FIGURES']['path_to_nondom_decisions_viz']
        ),
        'path_to_objective_correlation_viz': os.path.join(
            path_to_data,
            config_inputs['FIGURES']['path_to_objective_correlation_viz']
        ),
    }

    return inputs


def get_generator_information(net, gen_types=('gen', 'sgen', 'ext_grid')):
    """
    Get DataFrame summarizing all the generator information

    Parameters
    ----------
    net: pandapowerNet
        Network to summarize
    gen_types: list
        Generator types

    Returns
    -------
    df_gen_info: DataFrame
        Summary of generator information in `net`
    """
    # Initialize local vars
    df_gen_info = pd.DataFrame()

    # Convert generator information dataframe
    for gen_type in gen_types:
        df_temp_gen_info = getattr(net, gen_type)
        df_temp_gen_info['et'] = gen_type
        df_temp_gen_info['element'] = df_temp_gen_info.index.astype(float)
        df_gen_info = df_gen_info.append(df_temp_gen_info)
    df_gen_info = df_gen_info.reset_index(drop=True)  # Drop duplicated indices

    return df_gen_info


def grid_sample(df_gridspecs):
    """
    Pandas-based grid sampling function

    Parameters
    ----------
    df_gridspecs: DataFrame
        Grid specifications, must have columns of
        ['var', 'min', 'max', 'steps']. These reflect the variable names,
        minimum value of grid sampling, maximum value of grid sampling,
        and number of steps respectively.

    Returns
    -------
    df_grid: DataFrame
        Dataframe of grid sampling. Will have columns of names specified in
        'var' list
    """
    # Get linear spaces
    linspace_list = []
    for i, row in df_gridspecs.iterrows():
        linspace_list.append(
            np.linspace(row['min'], row['max'], int(row['steps']))
        )

    # Create dataframe
    df_grid = pd.DataFrame(
        list(itertools.product(*linspace_list)),
        columns=df_gridspecs['var'].tolist()
    )

    return df_grid


def compute_objective_terms(df):
    """
    Compute objective terms

    Parameters
    ----------
    df: DataFrame
        DataFrame with columns ['p_mw', 'a', 'b', 'c', 'alpha', 'beta_emit',
         'gamma', 'xi', 'lambda', 'beta_with', 'beta_con']

    Returns
    -------
    df: DataFrame
        Dataframe with objective terms computed in objective columns
    """
    # Cost
    df['F_cos'] = \
        df['a'] + df['b'] * (df['p_mw'] / 100) + \
        df['c'] * (df['p_mw'] / 100) ** 2

    # Emissions
    df['F_emit'] = \
        0.01 * df['alpha'] + \
        0.01 * df['beta_emit'] * (df['p_mw'] / 100) + \
        0.01 * df['gamma'] * (df['p_mw'] / 100) ** 2 + \
        df['xi'] * np.exp(df['lambda'] * (df['p_mw'] / 100))

    # Withdrawal
    df['F_with'] = df['beta_with'] * df['p_mw']

    # Consumption
    df['F_con'] = df['beta_con'] * df['p_mw']

    return df


def mo_opf(ser_decisions, net):
    """
    Multi-objective optimal power flow

    Parameters
    ----------
    ser_decisions: Series
        Decision variables in series, index is bus number, values are power in
        MW
    net: pandapowerNet
        Network to assess, df_coef attribute

    Returns
    -------
    ser_results: Series
        Series of objective values and internal decision
    """
    # Local vars
    net = copy.deepcopy(net)

    # Apply decision to network
    ser_decisions.name = 'p_mw_decisions'
    net.gen = net.gen.merge(ser_decisions, left_on='bus', right_index=True)

    net.gen['p_mw'] = net.gen['p_mw_decisions']

    # Solve powerflow to solve for external generator
    pp.rundcpp(net)

    # Check if external generator is outside limits
    ext_grid_p_mw = net.res_ext_grid['p_mw'][0]
    ext_gen_in_limits = \
        net.ext_grid['min_p_mw'][0] < ext_grid_p_mw < \
        net.ext_grid['max_p_mw'][0]

    if ext_gen_in_limits:
        # Formatting results
        df_obj = get_generator_information(net, ['res_gen', 'res_ext_grid'])
        df_obj = df_obj.merge(
            net.df_coef,
            left_on=['element', 'et'],
            right_on=['element', 'et']
        )

        # Compute objectives terms
        df_obj = compute_objective_terms(df_obj)

        # Compute objectives
        df_obj_sum = df_obj.sum()
        f_cos = df_obj_sum['F_cos']
        f_emit = df_obj_sum['F_emit']
        f_with = df_obj_sum['F_with']
        f_con = df_obj_sum['F_con']
        ser_results = pd.Series(
            {
                0: ext_grid_p_mw,
                'F_cos': f_cos,
                'F_emit': f_emit,
                'F_with': f_with,
                'F_con': f_con
            }
        )
    else:
        ser_results = pd.Series(
            {
                0: np.nan,
                'F_cos': np.nan,
                'F_emit': np.nan,
                'F_with': np.nan,
                'F_con': np.nan
            }
        )

    print(ser_results)

    return ser_results


def get_fuel_cool(df_abido_coef):
    """

    Parameters
    ----------
    df_abido_coef: DataFrame
        Cost and emission coefficients from Abido (2003) paper

    Returns
    -------
    df_coef: DataFrame
        Coefficients dataframe with fuel and cooling systems assigned

    """
    # Local vars
    df_coef = df_abido_coef.copy()

    # Compute objectives terms
    df_abido_coef['p_mw'] = 50.0
    df_objective_components = compute_objective_terms(df_abido_coef)

    # Minimum emissions gets assigned nuclear
    nuc_idx = df_objective_components['F_emit'].idxmin()
    df_coef.loc[nuc_idx, 'fuel_type'] = 'Nuclear'
    df_coef.loc[nuc_idx, 'cooling_type'] = 'Once-through'

    # Maximum emissions gets assigned coal
    coal_idx = df_objective_components['F_emit'].nlargest(3).index
    df_coef.loc[coal_idx, 'fuel_type'] = 'Coal'
    df_coef.loc[coal_idx, 'cooling_type'] = 'Once-through'

    # Remaining assigned natural gas
    df_coef['fuel_type'] = df_coef['fuel_type'].fillna('Natural Gas')

    # Natural gas get split to have some tower and some once-through cooling
    df_coef['cooling_type'] = df_coef['cooling_type'].fillna('Tower', limit=1)
    df_coef['cooling_type'] = df_coef['cooling_type'].fillna('Once-through')

    return df_coef


def get_emission_coef(df_coef):
    """
    Overwrite emissions coefficients for nuclear generators

    Parameters
    ----------
    df_coef: DataFrame
        Coefficients dataframe with fuel and cooling systems assigned

    Returns
    -------
    df_coef: DataFrame
        Updated coefficients dataframe with nuclear emission coefficients
        adjusted
    """
    # Local vars
    emit_cols = ['alpha', 'beta_emit', 'gamma', 'xi', 'lambda']

    # Nuclear has no air pollution
    nuc_idx = df_coef[df_coef['fuel_type'] == 'Nuclear'].index
    df_coef.loc[nuc_idx, emit_cols] = 0.0

    return df_coef


def get_water_use_rate(df_coef, df_macknick_coef):
    """
    Assign water consumption and withdrawal rates based on
    macknick_operational_2012

    @article{macknick_operational_2012,
        title = {Operational water consumption and withdrawal factors for
        electricity generating technologies: a review
         of existing literature},
        doi = {10.1088/1748-9326},
        journal = {Environ. Res. Lett.},
        author = {Macknick, J and Newmark, R and Heath, G and Hallett, K C},
        year = {2012},
        pages = {11},
    }

    Parameters
    ----------
    df_coef: DataFrame
        Coefficients dataframe with fuel and cooling systems assigned

    df_macknick_coef: DataFrame
        Water use coefficients from Macknick (2012)

    Returns
    -------
    df_coef: DataFrame
        Coefficients dataframe with withdrawal and consumption rates assigned
        (gal/MWh)
    """
    df_coef = pd.merge(
        df_coef,
        df_macknick_coef,
        left_on=['fuel_type', 'cooling_type'],
        right_on=['fuel_type', 'cooling_type']
    )
    df_coef['beta_with'] = df_coef['withdrawal_rate_(gal/MWh)']
    df_coef['beta_con'] = df_coef['consumption_rate_(gal/MWh)']

    return df_coef


def get_nondomintated(df, objs, max_objs=None):
    """
    Get nondominated filtered DataFrame

    Parameters
    ----------
    df: DataFrame
        DataFrame for nondomination
    objs: list
        List of strings correspond to column names of objectives
    max_objs: list (Optional)
        List of objective to maximize

    Returns
    -------
    df_nondom: DataFrame
        Nondominatated DataFrame
    """
    # Create temporary dataframe for sorting
    df_sort = df.copy()

    # Flip objectives to maximize
    if max_objs is not None:
        df_sort[max_objs] = -1.0 * df_sort[max_objs]

    # Non-dominated sorting
    nondom_idx = ends.efficient_non_dominated_sort(df_sort[objs].values)
    df_nondom = df.iloc[nondom_idx[0]].sort_index()

    return df_nondom


def get_epsilon_nondomintated(df, objs, epsilons, max_objs=None):
    """
    Get epsilon nondominated filtered DataFrame

    Parameters
    ----------
    df: DataFrame
        DataFrame for nondomination
    objs: list
        List of strings correspond to column names of objectives
    epsilons: list
        List of floats specifying epsilons for pareto sorting
    max_objs: list (Optional)
        List of objective to maximize
    Returns
    -------
    df_nondom: DataFrame
        Nondominatated DataFrame
    """
    # Get indices of objectives
    objs_ind = [df.columns.get_loc(col) for col in objs]
    try:
        max_ind = [df.columns.get_loc(col) for col in max_objs]
    except TypeError:
        max_ind = None

    # Nondominated sorting
    non_dominated = pto.eps_sort(
        [list(df.itertuples(False))], objs_ind, epsilons, maximize=max_ind
    )

    # To DataFrame
    df_nondom = pd.DataFrame(non_dominated, columns=df.columns)

    # Get original index
    df_nondom = df.reset_index().merge(df_nondom).set_index('index')
    df_nondom.index.name = None

    return df_nondom
