# CODE README

Modelling and analysis for **'Simulation of glacier ice algal bloom on the Greenland Ice Sheet'** by Williamson and Tedstone, submitted to ISME, summer 2025.

Author: Andrew Tedstone (andrew.tedstone@unil.ch), July 2025.

For expected outputs of these codes, see the paper's data repository.

## Algae growth modelling

The algal growth model is configured to operate ice-sheet-wide. In this study we ran the model on the regional climate model MAR's v3.13 10 km polar stereographic domain.

### Input data

All model input data are taken directly from MAR outputs provided at 1-hour resolution by Xavier Fettweis, University of Liege, Belgium. The following MAR variables are required:

- MSK
- SHSN2 (snow pack height above ice)
- TT (near-surface temperature)
- SWD (Shortwave down radiation)

Our model setup assumes that these data are provided in annual NetCDF files for the period 1 May - 30 September.

### Running the model

The model is contained in `algal_growth_model.py`, which has several usage options based around three overarching suites of experiments.

Paths to input data and output locations are hard-coded in the top of the script. 

All model parameters are hard-coded at their relevant locations throughout the script.

The script was developed for a system without a task scheduler (UNIL FGSE Octopus). On this system it is best to run the model in `tmux` or `screen` so that you can detach, as model runs can take several hours.


#### Phenological sensitivity experiments

    python algal_growth_model.py -psens

Produces model outputs for the specified year(s), one NetCDF file generated for each unique parameter.


#### Environmental sensitivity experiments

    python algal_growth_model.py -esens

Produces model outputs for the specified year(s), one NetCDF file generated for each unique parameter.


#### Quasi-Monte Carlo simulations

This is divided into two parts: (1) experiments design and (2) simulation. In experiment design, we generate 512 unique parameter sets spanning the parameter space, which we export to a CSV file. In simulation, we run the model between 1 May - 30 September every year for each of the parameter sets, providing the CSV file as input.

To generate the experiments:

    python algal_growth_model.py -init

To run the simulations, proceed year-by-year, e.g. for the year 2000:

    python algal_growth_model.py -y 2000 -e expt_parameters_ibio179.csv

On octopus I used the following in a shell script:

    mamba init
    mamba activate geospatial
    for y in {2000..2022}; do
    python algal_growth_model.py -y $y -e /work/atedstone/williamson/2025-05/expt_parameters_ibio179.csv;
    done


### Model outputs

A given output NetCDF file contains three arrays of daily model outputs:

- `today_prod` - gross daily growth, G_G.
- `today_prod_net` - net daily growth, G_N.
- `cum_growth` - daily population size, P.
- `daily_prod_hrs` - daily number of productive hours.

Some model runs conducted for this study were not set up to export `today_prod_net`. In these cases, the variable was added to each NetCDF afterwards, using the script `write_net_growth.py`, by calculating the net growth as a function of gross growth and the previous day's population size.


## Analysis

### Pre-processing/data reduction

Concerning the QMC ensemble, there are three computationally intensive workflows of pre-processing required ahead of analysis/interpretation.

#### Metrics of each QMC run

Use `calculate_qmc_run_summary_metrics.py`.

For every QMC experiment in the ensemble, produce annual maps of key metrics:

- Last productive day of year
- Maximum daily population size
- Net total growth

Export these into one NetCDF per QMC run, spanning all the years of analysis. Output file name format: `model_outputs_exp<ID>_summarystats.nc`.


#### Metrics of annual QMC ensembles

Use `calculate_qmc_ensemble_metrics.py`.

This takes files from the previous script as inputs. It calculates annual metrics of the whole QMC ensemble at specific quantiles (median, 25th, 75th), for all variables in the input files. Output file name format: `model_outputs_QMCE_<year>_q<quantile>.nc`.


#### Daily time series from ensemble

Use `calculate_qmc_daily_MIQR.py`. This script runs on a per-year basis. Use it to generate time/x/y netcdfs of the median, 25th and 75th percentile ensemble values. These netcdfs are used as inputs for:

- Fig. 5 comparison of measured vs modelled


N.b. that there is some redundancy/overlap with respect to the function `get_qmc_ts()` in `main_analysis.py` - this function works only at a single X/Y coordinate whereas the script here generates a time series for all cells.


#### Compilation of field (in-situ) datasets

We compile in-situ cell counts made by others using the script `prepare_validation_datasets.py`. This script ingests a mixture of (a) files taken directly from data repositories and (b) files created by AT/CW by tabulating cell counts taken from tables/text in the original manuscripts. This script produces a signal output CSV file that is taken forward to `main_analysis.py`.


### Figures and statistics

All figures and statistics are produced using `main_analysis.py` opened as a Jupyter Notebook. (Use the Jupytext extension).

Note that full analysis requires various ancillary files (e.g. Greenland drainage basin definitions) to be available. The paths are specified in the 'Paths/Settings' section of the Notebook. All data are available as specified in the 'Data Availability' section of the manuscript.


#### Interim processing files which are redundant due to final figure outputs

* `sens_analysis_s6_ibio_{param}.csv` --> Fig 1 xlsx
* `sens_analysis_s6_ploss_{param}.csv` --> Fig 1 xlsx
* `qmc_ts_Gn_{site}_{year}.csv` --> Fig 3 xlsx
* `qmc_ts_dailypop_{site}_{year}.csv` --> Fig 3 xlsx
