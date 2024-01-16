# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Algal Growth Model - Python version
#
# Based on `chris_ted_jan24.Rmd`. Created 16 Jan 2024.
#
# Uses MAR outputs with hourly TT, SHSN2 and SWD variables to run daily algal productvity model.
#
# Options to run sensitivity analysis included.
#
# Exports to NetCDFs with same basic dimensions as MAR inputs.

# %%
import numpy as np
import pandas as pd
from datetime import datetime

import xarray as xr
import rioxarray
from copy import deepcopy

def carbon_func_106pg(x):
    x = x / 0.84  # assuming 0.84 ng DW per cell
    x = x * 106  # convert from cells per ml to pg C ml assuming 106 pg C per cell
    x = x * 1000  # pg C ml to pg C per L
    x = x * 1.061  # pg C per l to pg C per m2 using conversion from Williamson et al. 2018
    x = x * 10 ** 6  # pg C per m2 to pg C per km2
    x = x * 10 ** -15  # pg C per km2 to kg of C per km2
    total_kg_C_km2 = np.sum(x)
    total_kg_C_pixel = total_kg_C_km2 * 100  # assuming pixels = 10 * 10 km = 100 km2
    return {'total_kg_C_km2': total_kg_C_km2, 'total_kg_C_pixel': total_kg_C_pixel}


def daily_prod(x):
    return 2936.966 / (1 + ((2936.966 - 796.239) / 796.239) * np.exp(-0.000232 * x))


def prepare_model_inputs(
    netcdf_path,
    light_thres = 10., 
    temp_thres = 0.5, 
    snow_thres = 0.02):

    # Connect to net cdf - allows seeing variable etc metadata
    nc_file = xr.open_dataset(netcdf_path)
    nc_file = nc_file.rename({'Y19_288':'y', 'X14_163':'x'})
    nc_file = nc_file.rio.write_crs('epsg:3413')

    msk = nc_file.MSK
    msk = msk.where(msk > 0)

    # Read in snow pack height above ice (SHSN2)
    shsn2 = nc_file.SHSN2

    # Read in shortwave-down
    light = nc_file.SWD

    # Read in temperature
    temp = nc_file.TT

    # Get x and y coordinates of the model area
    x = nc_file.x
    y = nc_file.y

    hourly_masked = (light > light_thres) & (temp > temp_thres) & (shsn2 > snow_thres) 
    daily = hourly_masked.resample(TIME='1D').sum().squeeze()
    daily.name = 'daily_prod_hrs'

    return daily



# %%
def run_model_annual(prod_hrs, initial_bio, percent_loss):
    
    bio = initial_bio
    
    store_cum_growth = []
    store_today_production = []
    for day in prod_hrs.TIME:

        hours = prod_hrs.sel(TIME=day)

        # Calculate possible production as a function of existing population size
        possible_production = daily_prod(bio)

        # Calculate gross production
        today_production = possible_production * (hours / 24)

        # Do some renaming
        today_production.name = 'today_prod'
        today_production = today_production.squeeze()
        today_production = today_production.expand_dims({'TIME':[day.values]})

        # Calculate new population size...
        # 1. add today's growth to pop
        cum_growth = bio + today_production
        # 2. Remove the losses
        cum_growth = cum_growth - (cum_growth * percent_loss)
        # 3. Do not allow pop size to fall below initial value
        cum_growth = cum_growth.where(cum_growth > initial_bio, initial_bio)

        cum_growth.name = 'cum_growth'
        cum_growth = cum_growth.squeeze()

        bio = deepcopy(cum_growth)

        cum_growth = cum_growth.expand_dims({'TIME':[day.values]})

        store_cum_growth.append(deepcopy(cum_growth))
        store_today_production.append(deepcopy(today_production))

    cum_growth = xr.concat(store_cum_growth, dim='TIME')
    daily_production = xr.concat(store_today_production, dim='TIME')
    
    return (cum_growth, daily_production)


# %%
mar = xr.open_dataset('/Users/tedstona/Library/CloudStorage/Dropbox/work/tmp_shares/williamson_MAR_hourly/MARv3.13-ERA5/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc')
mar = mar.rename({'Y19_288':'y', 'X14_163':'x'})
mar = mar.rio.write_crs('epsg:3413')

# %%
initial_bio = 179
ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
ibio = [initial_bio*0.1, initial_bio, initial_bio*5, initial_bio*10]

for year in range(2000, 2023):
    print(year)
    pth = f'/Users/tedstona/Library/CloudStorage/Dropbox/work/tmp_shares/williamson_MAR_hourly/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
    hours = prepare_model_inputs(pth)
    for pl in ploss:
        
        bio = xr.DataArray(179, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, pl)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(f'/scratch/williamson/model_outputs_{year}_ibio179_ploss{pl}.nc')
        

for year in range(2000, 2023):
    print(year)
    pth = f'/Users/tedstona/Library/CloudStorage/Dropbox/work/tmp_shares/williamson_MAR_hourly/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
    hours = prepare_model_inputs(pth)
    for ib in ibio:

        bio = xr.DataArray(ib, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, 0.10)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(f'/scratch/williamson/model_outputs_{year}_ploss0.1_ibio{ib}.nc')  
