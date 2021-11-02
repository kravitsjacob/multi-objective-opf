"""Source code for analysis"""

import os
import configparser
import argparse


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
        'path_to_pae': os.path.join(path_to_data, config_inputs['INPUT']['path_to_pae']),
    }

    return inputs
