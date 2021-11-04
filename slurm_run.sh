#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --job-name=gosox
#SBATCH --partition=shas
#SBATCH --qos=condo
#SBATCH --account=ucb-summit-jrk
#SBATCH --time=0-05:00:00
#SBATCH --mail-user=kravitsjacob@gmail.com
#SBATCH --mail-type=END

module purge
source /curc/sw/anaconda3/2019.07/bin/activate

# Create conda environment
conda env create -f environment.yml

# Load conda environment
conda activate multi-objective-opf-env

# Install non-conda packages
pip install -U pymoo

# Run analysis
python -u main.py -c config.ini -n_tasks 24