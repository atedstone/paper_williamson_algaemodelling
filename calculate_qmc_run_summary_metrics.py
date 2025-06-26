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
# # Calculate summary metrics for each QMC run
#
# Here, we are interested in generating annual maps of:
#
# - Last productive day of year
# - Maximum daily population size
# - Net growth
#
# To do this, we need to iterate through all the QMC runs, each of which have one NetCDF file per year.
#
# Outputs a single netCDF file for each QMC run.
#
# This script should be run 'headless' (e.g. with `screen`) because it takes some time.

# %%
import os
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd

from dask.distributed import LocalCluster as MyCluster
from dask.distributed import Client

# %%
WORK_ROOT = '/work/atedstone/williamson/'
RESULTS = os.path.join(WORK_ROOT, 'results')
NQMC = 512
YR_ST = 2000
YR_END = 2022
START_POP = 179

MAR_REF = os.path.join(WORK_ROOT, 'MARv3.14.0-10km-daily-ERA5-2022.nc')

def open_model_run(path):
    """ Load a model run at given path and update x/y coordinates to those from MAR reference run """
    run = xr.open_mfdataset(path)
    run['x'] = mar.x
    run['y'] = mar.y
    return run


# %%
if __name__ == '__main__':

    #cluster = MyCluster()
    #client = Client(cluster)
    #cluster.scale(6)
    # Open 'reference' MAR run, mainly for the ice sheet mask
    mar = xr.open_dataset(MAR_REF)
    mar['x'] = mar['x'] * 1000
    mar['y'] = mar['y'] * 1000
    mar = mar.rio.write_crs('epsg:3413')

    for expt_id in range(35, NQMC+1):
    
        print(expt_id)
    
        # Open all years of this experiment run
        expt_outputs = open_model_run(os.path.join(WORK_ROOT, '2025-05', f'model_outputs_*_exp{expt_id}.nc'))
        expt_outputs = expt_outputs.chunk({'TIME':153})
        
        # Find annual last productive day in each grid cell.
        store = []
        for year in range(YR_ST, YR_END+1):
            d = expt_outputs.sel(TIME=str(year))
            doy = d.TIME.dt.dayofyear
            d['TIME'] = doy
            lpd = d.today_prod.cumsum(dim='TIME').idxmax(dim='TIME')
            lpd.coords['TIME'] = pd.Timestamp(year, 1, 1)
            lpd = lpd.expand_dims({'TIME':[pd.Timestamp(year, 1, 1)]})
            store.append(lpd.compute())
        
        last_prod_doy = xr.concat(store, dim='TIME') 
        # Forward-fill the end DOY to cover the full year, each year
        # And we manually force this to continue to the end of 2022.
        last_prod_doy = last_prod_doy.reindex(TIME=pd.date_range(last_prod_doy['TIME'].isel(TIME=0).values, '2023-01-01', freq='1D'), method='ffill').compute()
    
        # Calculate metrics
        # ... Maximum daily population size
        valid_pop = expt_outputs.cum_growth.where(expt_outputs.TIME.dt.dayofyear <= last_prod_doy).where(expt_outputs.cum_growth > START_POP).where(mar.MSK > 50)
        annual_pop_max = valid_pop.resample(TIME='1YS').max(dim='TIME').compute()
        # ... Total annual net growth, still in ng DW ml-1 as we retain the x,y information
        valid_net_growth = expt_outputs.today_prod_net.where(expt_outputs.TIME.dt.dayofyear <= last_prod_doy).where(expt_outputs.cum_growth > START_POP).where(mar.MSK > 50)
        annual_net_growth_sum = valid_net_growth.resample(TIME='1YS').sum(dim='TIME')
        annual_net_growth_sum = annual_net_growth_sum.where(annual_net_growth_sum > 0).compute()
        
        # Save these results to disk
        to_save = xr.Dataset({'annual_net_growth_sum':annual_net_growth_sum, 'annual_pop_max':annual_pop_max, 'last_bloom_day':last_prod_doy.resample(TIME='1YS').first()})
        to_save.to_netcdf(os.path.join(WORK_ROOT, RESULTS, f'model_outputs_exp{expt_id}_summarystats.nc'))
    
        expt_outputs.close()
