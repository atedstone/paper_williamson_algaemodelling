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
import numpy as np
import pandas as pd
from datetime import datetime

import xarray as xr
import rioxarray
from copy import deepcopy
import os
import argparse

from scipy.stats import qmc

# %% trusted=true
# octopus
inputs_path = '/work/atedstone/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
mar_example = '/work/atedstone/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc'
output_path = '/work/atedstone/williamson/outputs/QMC/' 
from dask.distributed import LocalCluster as MyCluster
from dask.distributed import Client

# beo05
# inputs_path = '/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-{year}_05-09_ALGV.nc' 
# mar_example = '/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc'
# output_path = '/flash/tedstona/williamson/outputs/'
#from dask_jobqueue import SLURMCluster as MyCluster

# netcdf output compression options
# cum growth: float64, up to c. 30,000.
# today_prod: float64, up to c. 3,000.
# productive hours: currently int64. range 0-24
# encoding = {
#     'cum_growth':     {'dtype': 'int16', 'scale_factor': 1, '_FillValue': -9999},
#     'today_prod':     {'dtype': 'int16', 'scale_factor': 1, '_FillValue': -9999},
#     'daily_prod_hrs': {'dtype': 'int8',  'scale_factor': 1, '_FillValue': -9999} 
# }


# %% trusted=true
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


def run_model_annual(prod_hrs, initial_bio, percent_loss):
    
    bio = deepcopy(initial_bio)
    
    store_cum_growth = []
    store_today_gross_prod = []
    store_today_net_prod = []
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
        L = (cum_growth * percent_loss)
        cum_growth = cum_growth - L
        # 3. Do not allow pop size to fall below initial value
        cum_growth = cum_growth.where(cum_growth > initial_bio, initial_bio)

        cum_growth.name = 'cum_growth'
        cum_growth = cum_growth.squeeze()

        bio = deepcopy(cum_growth)

        cum_growth = cum_growth.expand_dims({'TIME':[day.values]})

        # Net daily production
        net_today_prod = np.maximum(0, today_production - L)
        net_today_prod.name = 'today_prod_net'

        store_cum_growth.append(deepcopy(cum_growth))
        store_today_gross_prod.append(deepcopy(today_production))
        store_today_net_prod.append(deepcopy(net_today_prod))

    cum_growth = xr.concat(store_cum_growth, dim='TIME')
    daily_production = xr.concat(store_today_gross_prod, dim='TIME')
    daily_net_production = xr.concat(store_today_net_prod, dim='TIME')
    
    return (cum_growth, daily_production, daily_net_production)


# Inverse cumulative distribution function for the triangular distribution
def icdf(sample, a, c, b):
    """
    sample : the column of values drawn.
    a : min
    c : mode
    b : max
    """
    x_values = np.zeros_like(sample)
    for i, u in enumerate(sample):
        if u <= (c - a) / (b - a):
            x_values[i] = a + np.sqrt((b - a) * (c - a) * u)
        else:
            x_values[i] = b - np.sqrt((b - a) * (b - c) * (1 - u))
    return x_values


# %% trusted=true
if __name__ == '__main__':

    p = argparse.ArgumentParser('Run algal bloom modelling. By default, run QMC ensemble.')
    p.add_argument('-y', type=int)
    p.add_argument('-e', type=str, help='path to experiments CSV file')
    p.add_argument('-init', action='store_true', help='initialise experiments CSV file then exit. (No other params needed)')
    p.add_argument('-psens', action='store_true', help='run single-parameter phenological sensitivity experiments. No QMC ensemble here.')
    p.add_argument('-esens', action='store_true', help='run single-parameter environmental sensitivity experiments. No QMC ensemble here.')
    args = p.parse_args()

    if args.psens:
        print('PHENOLOGICAL SENSITIVITY RUNS')

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
        
                cg, dp, dpn = run_model_annual(hours, bio, pl)
        
                full_out = xr.merge([hours, cg, dp, dpn])
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_out.data_vars}
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
        
                cg, dp, dpn = run_model_annual(hours, bio, default_ploss)
        
                full_out = xr.merge([hours, cg, dp, dpn])
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_out.data_vars}
                full_out.to_netcdf(os.path.join(output_path, 'sensitivity_ibio', f'model_outputs_{year}_ploss0.1_ibio{ib}.nc'),
                                  encoding=encoding
                )
                
    elif args.esens:
        print('ENVIRONMENTAL SENSITIVITY RUNS')
        years = [2016]
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
        
                cg, dp, dpn = run_model_annual(hours, bio, default_ploss)
        
                full_out = xr.merge([hours, cg, dp, dpn])
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_out.data_vars}
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
        
                cg, dp, dpn = run_model_annual(hours, bio, default_ploss)
        
                full_out = xr.merge([hours, cg, dp, dpn])
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_out.data_vars}
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
        
                cg, dp, dpn = run_model_annual(hours, bio, default_ploss)
        
                full_out = xr.merge([hours, cg, dp, dpn])
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_out.data_vars}
                full_out.to_netcdf(
                    os.path.join(output_path, 'sensitivity_light', f'model_outputs_{year}_ibio{initial_bio}_ploss{default_ploss}_light{li}.nc'),
                    encoding=encoding
                )
        
    elif args.init:
        print('QMC EXPERIMENT DESIGN')
        engine = qmc.Sobol(d=4)
        qmc_vals = engine.random(512)

        # Specify the triangular distributions of each variable
        # (min, mode, max)
        # These values directly define the shape of the PDF.

        # Threshold air temperature above which algae can bloom (degrees C)
        dist_t = (0.0, 0.5, 1.0)
        # Maximum snow depth below which ice algae can bloom (metres)
        dist_sd = (0.0, 0.05, 0.5)
        # SWD above which algae can bloom
        # N.b. Proxy for PAR. 50 Wm-2 is roughly 100 umol (1 W --> 2 mmol)
        dist_swd = (25, 50, 100)
        # Daily population loss term (proportion 0-->1)
        dist_ploss = (0.05, 0.10, 0.15)

        # Transform the QMC to the experiment parameters using the specified distibutions
        exps_t = icdf(qmc_vals[:,0], *dist_t)
        exps_sd = icdf(qmc_vals[:,1], *dist_sd)
        exps_swd = icdf(qmc_vals[:,2], *dist_swd)
        exps_ploss = icdf(qmc_vals[:,3], *dist_ploss)

        exps_pd = pd.DataFrame({'T_celsius':exps_t, 'snow_depth_metres':exps_sd, 'SWd_wm2':exps_swd, 'ploss':exps_ploss})
        exps_pd.index += 1
        fn = os.path.join(output_path, f'expt_parameters_ibio{initial_bio}.csv')
        exps_pd.to_csv(fn)
        print(f'Experiments design saved to {fn} .')

    else:
        print('QMC MODEL RUN')
        initial_bio = 179

        efn = args.e
        print(f'Loading experiments file {efn}')
        expts_pd = pd.read_csv(efn, index_col=0)

        cluster = MyCluster()
        cluster.scale(n=10)
        
        client = Client(cluster)

        mar = xr.open_dataset(mar_example)
        mar = mar.rename({'Y19_288':'y', 'X14_163':'x'})
        mar = mar.rio.write_crs('epsg:3413')

        #pt_s6_x = -2507730
        #pt_s6_y = -193438

        year = args.y
        pth = inputs_path.format(year=year)

        i = 1
        for ix, row in expts_pd.iterrows():
            print(f'Experiment {i}')
            
            hours = prepare_model_inputs(
                pth,
                light_thres=row['SWd_wm2'],
                temp_thres=row['T_celsius'],
                snow_thres=row['snow_depth_metres']
            )

            bio = xr.DataArray(initial_bio, dims=('y', 'x'),coords={'y':hours.y, 'x':hours.x})

            cg, dp, Gn = run_model_annual(hours, bio, row['ploss'])

            full_out = xr.merge([hours, cg, dp, Gn])
            
            full_out.attrs['exp_id'] = i
            full_out.attrs['param_swd'] = row['SWd_wm2']
            full_out.attrs['param_t'] = row['T_celsius']
            full_out.attrs['param_sd'] = row['snow_depth_metres']
            full_out.attrs['param_ploss'] = row['ploss']
            full_out.attrs['param_startpop'] = initial_bio

            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in full_out.data_vars}
            
            full_out.to_netcdf(
                os.path.join(output_path, f'model_outputs_{year}_exp{i}.nc'),
                encoding=encoding
            )
            i += 1
