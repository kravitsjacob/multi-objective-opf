"""Source code for analysis"""

import os
import configparser
import argparse
import itertools
import copy

import numpy as np
import pandas as pd
import pandapower as pp


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

        # Parse arguments
        argparse_inputs = argparse_inputs.parse_args()

        # Parse config file
        config_inputs.read(argparse_inputs.config_file)

    # Store inputs
    path_to_data = config_inputs['MAIN IO']['data']
    inputs = {
        'path_to_data': path_to_data,
        'path_to_df_coef': os.path.join(path_to_data, config_inputs['INPUT']['path_to_df_coef']),
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
    df_gen_info = df_gen_info.reset_index(drop=True)  # Important to eliminate duplicated indices

    return df_gen_info


def grid_sample(df_gridspecs):
    """
    Pandas-based grid sampling function

    Parameters
    ----------
    df_gridspecs: DataFrame
        Grid specifications, must have columns of ['var', 'min', 'max', 'steps']. These reflect the variable names,
        minimum value of grid sampling, maximum value of grid sampling, and number of steps respectively.

    Returns
    -------
    df_grid: DataFrame
        Dataframe of grid sampling. Will have columns of names specified in 'var' list
    """
    # Get linear spaces
    linspace_list = []
    for i, row in df_gridspecs.iterrows():
        linspace_list.append(np.linspace(row['min'], row['max'], int(row['steps'])))

    # Create dataframe
    df_grid = pd.DataFrame(list(itertools.product(*linspace_list)), columns=df_gridspecs['var'].tolist())

    return df_grid


def mo_opf(ser_decisions, net):
    # Local vars
    t = 5 * 1 / 60 * 1000  # minutes * hr/minutes * kw/MW
    net = copy.deepcopy(net)

    # Apply decision to network
    net.gen = net.gen.merge(ser_decisions, left_on='bus', right_index=True)
    net.gen['p_mw'] = net.gen[0]

    # Solve powerflow
    pp.rundcpp(net)

    # Formatting results
    df_obj = get_generator_information(net, ['res_gen', 'res_ext_grid'])
    df_obj = df_obj.merge(net.df_coef, left_on=['element', 'et'], right_on=['element', 'et'])

    # Compute objectives terms
    df_obj['F_cos'] = df_obj['a'] + df_obj['b'] * (df_obj['p_mw']/100) + df_obj['c'] * (df_obj['p_mw']/100)**2
    df_obj['F_emit'] = \
        0.01 * df_obj['alpha'] +\
        0.01 * df_obj['beta_emit'] * (df_obj['p_mw']/100) +\
        0.01 * df_obj['gamma'] * (df_obj['p_mw']/100)**2 +\
        df_obj['xi'] * np.exp(df_obj['lambda'] * (df_obj['p_mw']/100))
    df_obj['F_with'] = df_obj['beta_with'] * df_obj['p_mw'] * t
    df_obj['F_con'] = df_obj['beta_with'] * df_obj['p_mw'] * t

    # Compute objectives
    df_obj_sum = df_obj.sum()
    f_cos = df_obj_sum['F_cos']
    f_emit = df_obj_sum['F_emit']
    f_with = df_obj_sum['F_with']
    f_con = df_obj_sum['F_con']
    ser_obj = pd.Series(
        {
            'F_cos': f_cos,
            'F_emit': f_emit,
            'F_with': f_with,
            'F_con': f_con
        }
    )

    return ser_obj
