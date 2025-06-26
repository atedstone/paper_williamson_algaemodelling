# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Calculate metrics of entire ensemble
#
# This script uses the outputs produced by `calculate_qmc_run_summary_metrics.py`.
#
# It produces one NetCDF file for each quantile of interest, containing the whole time series.
#
# This script should be run 'headless' (e.g. with `screen`) because it takes some time.

# %%
import os
import numpy as np
import xarray as xr
import datetime as dt

from dask.distributed import LocalCluster as MyCluster
from dask.distributed import Client

# %%
WORK_ROOT = '/work/atedstone/williamson/'
RESULTS = os.path.join(WORK_ROOT, 'results')
NQMC = 512
YR_ST = 2021
YR_END = 2022

fn_annual_summary = os.path.join(WORK_ROOT, RESULTS, 'model_outputs_QMCE_{year}_summary_q{q}.nc')


# %%
# Open all maxes at once
# stack by experiment...
def open_year_ensemble_summary(year):
    return xr.open_mfdataset(
        os.path.join(WORK_ROOT, RESULTS, 'model_outputs_exp*_summarystats.nc'), 
        preprocess=lambda ds:ds.sel(TIME=str(year)), 
        concat_dim='expt', combine='nested')


# %%
if __name__ == '__main__':
    cluster = MyCluster()
    client = Client(cluster)
    cluster.scale(n=6)
    
    for year in range(YR_ST, YR_END+1):
        print(year)
        d = open_year_ensemble_summary(year)
        d.median(dim='expt').to_netcdf(fn_annual_summary.format(year=year, q=50))
        d.quantile(0.25, dim='expt').to_netcdf(fn_annual_summary.format(year=year, q=25))
        d.quantile(0.75, dim='expt').to_netcdf(fn_annual_summary.format(year=year, q=75))
