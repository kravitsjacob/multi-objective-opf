#!/bin/bash

# Load conda commands
eval "$(conda shell.bash hook)"

# Create conda environment
conda env create -f multi-objective-opf-env.yml

# Load conda environment
conda activate multi-objective-opf-env

# Install non-conda packages
pip install -U pymoo

# Run analysis
python main.py -c config.ini