#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100
#SBATCH --output=run.out
#SBATCH --job-name=gosox
#SBATCH --partition=shas
#SBATCH --qos=condo
#SBATCH --account=ucb-summit-jrk
#SBATCH --time=0-00:20:00
#SBATCH --mail-user=kravitsjacob@gmail.com
#SBATCH --mail-type=END

# Setup
module purge
source /curc/sw/anaconda3/2019.07/bin/activate

# Create conda environment
conda env create -f environment.yml

# Load conda environment
conda activate multi-objective-opf-env

# Install non-conda packages
pip install -U pymoo
git clone https://github.com/matthewjwoodruff/pareto.py.git pareto
git clone https://github.com/matthewjwoodruff/pareto.py.git src/pareto

# Run analysis
python -u main.py -c config.ini --n_tasks 32 --n_steps 10