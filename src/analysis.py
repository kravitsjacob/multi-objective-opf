"""Source code for analysis"""

import os
import configparser
import argparse
import itertools

import numpy as np
import pandas as pd


def input_parse():
    """
    Parse inputs
    :return: dict
        Dictionary of inputs
    """
    # Local vars
    config_inputs = configparser.ConfigParser()
    argparse_inputs = argparse.ArgumentParser()

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
        linspace_list.append(np.linspace(row['min'], row['max'], row['steps']))

    # Create dataframe
    df_grid = pd.DataFrame(list(itertools.product(*linspace_list)), columns=df_gridspecs['var'].tolist())

    return df_grid
