"""Conduct analysis python script"""

import pandapower.networks
import pandas as pd

from src import analysis


def main():
    # Inputs
    inputs = analysis.input_parse()
    df_coef = pd.read_csv(inputs['path_to_df_coef'])
    net = pandapower.networks.case_ieee30()

    # Get coefficients
    a = 1

    # solve opf

    return 0


if __name__ == '__main__':
    main()