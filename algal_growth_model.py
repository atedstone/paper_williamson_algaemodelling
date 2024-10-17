# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
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

# %% trusted=true
# local
# inputs_path = '/Users/tedstona/Library/CloudStorage/Dropbox/work/tmp_shares/williamson_MAR_hourly/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
# mar_example = '/Users/tedstona/Library/CloudStorage/Dropbox/work/tmp_shares/williamson_MAR_hourly/MARv3.13-ERA5/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc'

# beo05
inputs_path = '/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
mar_example = '/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc'
output_path = '/flash/tedstona/williamson/outputs/'

# netcdf output compression options
# cum growth: float64, up to c. 30,000.
# today_prod: float64, up to c. 3,000.
# productive hours: currently int64. range 0-24
encoding = {
    'cum_growth':     {'dtype': 'int16', 'scale_factor': 1, '_FillValue': -9999},
    'today_prod':     {'dtype': 'int16', 'scale_factor': 1, '_FillValue': -9999},
    'daily_prod_hrs': {'dtype': 'int8',  'scale_factor': 1, '_FillValue': -9999} 
}

# %% trusted=true
import numpy as np
import pandas as pd
from datetime import datetime

import xarray as xr
import rioxarray
from copy import deepcopy
import os


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
    # Only run the model in pixels which have snow cover at the start of the model run
    # This line basically masks out interior ice sheet pixels, which always have SHSN2=0.
    shsn2 = shsn2.where(shsn2.isel(TIME=0) > 0)

    # Read in shortwave-down
    light = nc_file.SWD

    # Read in temperature
    temp = nc_file.TT

    # Get x and y coordinates of the model area
    x = nc_file.x
    y = nc_file.y

    hourly_masked = (light > light_thres) & (temp > temp_thres) & (shsn2 < snow_thres) 
    daily = hourly_masked.resample(TIME='1D').sum().squeeze()
    daily.name = 'daily_prod_hrs'

    return daily



# %% trusted=true
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(jobs=6)
client = Client(cluster)

# %% trusted=true
client


# %% trusted=true
def run_model_annual(prod_hrs, initial_bio, percent_loss):
    
    bio = deepcopy(initial_bio)
    
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


# %% trusted=true
mar = xr.open_dataset(mar_example)
mar = mar.rename({'Y19_288':'y', 'X14_163':'x'})
mar = mar.rio.write_crs('epsg:3413')

# %% trusted=true
# MAIN MODEL RUN

initial_bio = 179
pl = 0.1

for year in range(2000, 2023):
    print(year)
    pth = inputs_path.format(year=year)
    hours = prepare_model_inputs(pth)
    
    bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

    cg, dp = run_model_annual(hours, bio, pl)

    full_out = xr.merge([hours, cg, dp])
    full_out.to_netcdf(
        os.path.join(output_path, 'main_outputs', f'model_outputs_{year}_ibio{initial_bio}_ploss{pl}.nc'),
        encoding=encoding
    )

# %% trusted=true
# ENVIRONMENTAL SENSITIVITY RUNS

years = [2000, 2012]
initial_bio = 179
default_ploss = 0.10

# To snow
snow_depths = [0.01, 0.05, 0.10, 0.20, 0.40, 0.80]
for year in years:
    print(year)
    pth = inputs_path.format(year=year)
    for d in snow_depths:
        hours = prepare_model_inputs(pth, snow_thres=d)    
        
        bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, default_ploss)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(
            os.path.join(output_path, 'sensitivity_snowdepth', f'model_outputs_{year}_ibio{initial_bio}_ploss{default_ploss}_snow{d}.nc'),
            encoding=encoding
        )
        

# Sensitivity to near-surface temperature (0-1 c)
temps = [0, 0.25, 0.5, 1.0]
for year in years:
    print(year)
    pth = inputs_path.format(year=year)
    for t in temps:
        hours = prepare_model_inputs(pth, temp_thres=t)    
        
        bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, default_ploss)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(
            os.path.join(output_path, 'sensitivity_temp', f'model_outputs_{year}_ibio{initial_bio}_ploss{default_ploss}_temp{t}.nc'),
            encoding=encoding
        )


# To light
lights = [1, 10, 100, 200]
for year in years:
    print(year)
    pth = inputs_path.format(year=year)
    for li in lights:
        hours = prepare_model_inputs(pth, light_thres=li)    
        
        bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, default_ploss)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(
            os.path.join(output_path, 'sensitivity_light', f'model_outputs_{year}_ibio{initial_bio}_ploss{default_ploss}_light{li}.nc'),
            encoding=encoding
        )



# %% trusted=true
# PHENOLOGICAL SENSITIVITY RUNS

initial_bio = 179
default_ploss = 0.10
ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
ibio = [initial_bio*0.1, initial_bio, initial_bio*5, initial_bio*10]

# Sensitivity to population loss
for year in range(2000, 2023):
    print(year)
    pth = inputs_path.format(year=year)
    hours = prepare_model_inputs(pth)
    for pl in ploss:
        
        bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, pl)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(
            os.path.join(output_path, 'sensitivity_ploss', f'model_outputs_{year}_ibio179_ploss{pl}.nc'),
            encoding=encoding
        )
        
# Sensitivity to starting biomass
for year in range(2000, 2023):
    print(year)
    pth = inputs_path.format(year=year)
    hours = prepare_model_inputs(pth)
    for ib in ibio:

        bio = xr.DataArray(ib, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

        cg, dp = run_model_annual(hours, bio, default_ploss)

        full_out = xr.merge([hours, cg, dp])
        full_out.to_netcdf(os.path.join(output_path, 'sensitivity_ibio', f'model_outputs_{year}_ploss0.1_ibio{ib}.nc'),
                          encoding=encoding
        )

# %% trusted=true
ibio

# %% trusted=true
full_out.cum_growth.where(mar.MSK > 50).where(full_out.cum_growth > 179).sum(dim='TIME').plot()

# %% trusted=true
