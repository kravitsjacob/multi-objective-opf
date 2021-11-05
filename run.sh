#!/bin/bash

# Load conda commands
eval "$(conda shell.bash hook)"

# Create conda environment
conda env create -f environment.yml

# Load conda environment
conda activate multi-objective-opf-env

# Install non-conda packages
pip install -U pymoo
git clone https://github.com/matthewjwoodruff/pareto.py.git pareto
git clone https://github.com/matthewjwoodruff/pareto.py.git src/pareto

# Run analysis
python main.py -c config.ini