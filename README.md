# multi-objective-opf
Multi-objective optimal power flow with cost, emissions, thermoelectric water withdrawal, and thermoelectric water consumption.

# I. Contents
```
multi-objective-opf
│   .gitattributes
│   .gitignore
│   config.ini: Configuration file with inputs and outputs
│   environment.yml: Conda environment
│   LICENSE
│   main.py: Python script to conduct analysis
│   README.md
│   run.sh: Bash script to run main.py with proper dependencies
│   slurm_run.sh: `run.sh` equivalent with the slurm batch manager
│   slurm_test_run.sh: Minimal run to test is run will work on slurm
|
├───.github: Github actions for running unit testing
│   └───workflows
│           python-package-conda.yml
│
│
├───multi-objective-opf-io-v1.0
│   ├───figures
│   │       nondom_decisions.pdf
│   │       nondom_hiplot.html: Interactive figure
│   │       nondom_objectives.pdf
│   │       objective_correlation.pdf
│   │
│   ├───generated_files
│   │       grid_results.csv: Results of grid search
│   │       nondom.csv: Epsilon non-dominated set of grid_results
│   │
│   └───manual_files
│           abido_2003_coefficients.csv: Emissions coefficients 
│           macknick_2012_coefficients.csv: Water usage coefficients
│
│
└───src: Source code
        analysis.py: Source code for analysis
        viz.py: Source code for visualization
        analysis_testing.py: Unit tests for functions in `analysis.py`
```

# II. How to Run
This tutorial assumes the use of [gitbash](https://git-scm.com/downloads) or a Unix-like terminal with github command line usage.
1. This project utilizes conda to manage environments and ensure consistent results. Download [miniconda](https://docs.conda.io/en/latest/miniconda.html) and ensure you can activate it from your terminal by running `$conda activate` 
    * Depending on system configuration, this can be an involved process [here](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) is a recommended thread.
3. Clone the repository using `$git clone https://github.com/kravitsjacob/multi-objective-opf.git` 
4. Change to the current working directory using `$cd <insert_path>/multi-objective-opf`
6. Run the analysis by running `$bash run.sh`
