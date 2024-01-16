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
# # Convert R model outputs to NetCDF

# %%
import xarray as xr
import rioxarray
import pandas as pd
import glob

# %%
mar = xr.open_dataset('/scratch/williamson/MARv3.13-10km-ERA5-2000_05-09_ALGV.nc')
mar = mar.rename({'Y19_288':'y', 'X14_163':'x'})
mar['x'] = mar['x'] * 1000
mar['y'] = mar['y'] * 1000


# %% [markdown]
# ## Derived parameters

# %%
def tif_dp_to_nc(
    filename, 
    dp_names = ["date_first_growth", "date_last_growth", "length_p_window","biomass_sum",
                "k", "k_p", "r", "t_gen", "max.bio", "last.bio",
                "kg.C.km2", "kg.C.pixel"]
    ):
    """
    Load derived parameters GeoTIFF into xarray Dataset.
    """
    dp = rioxarray.open_rasterio(filename)
    dp['x'] = mar['x'].values
    dp['y'] = mar['y'].values[::-1]
    
    ts = pd.Timestamp(
                year=int(filename.split('/')[-1][0:4]),
                month=1,
                day=1)
    
    store = {}
    for b in dp.band:
        tmp = dp.sel(band=b)
        tmp = tmp.drop_vars('band')
        store[dp_names[b.values-1]] = tmp
    dpm = xr.Dataset(store)
    dpm = dpm.expand_dims({'time':[ts]})
    #dpm = dpm.assign_coords({'time': ts})
    return dpm


# %%
files = glob.glob('/scratch/williamson/normal_model_runs/*_derived_parameters.tif')
store = []
for f in files:
    print(f)
    oneyear = tif_dp_to_nc(f)
    store.append(oneyear)
dp_all = xr.merge(store)

# %%
dp_all

# %%
dp_all.to_netcdf('/scratch/williamson/normal_model_runs/derived_parameters_all.nc')


# %% [markdown]
# ## Total bio

# %%
def tif_timeseries_to_nc(
    filename, 
    name,
    start_date,
    end_date,
    freq='1D',
    ):
    """
    Load annual/daily timeseries GeoTIFF into xarray Dataset.
    """
    tif = rioxarray.open_rasterio(filename)
    tif['x'] = mar['x'].values
    tif['y'] = mar['y'].values[::-1]
    
    ts = pd.date_range(start_date, end_date, freq=freq)
    
    tif['band'] = ts
    tif = tif.rename({'band':'time'})
    tif.name = name
    
    return tif


# %%
files = glob.glob('/scratch/williamson/normal_model_runs/*_total_bio.tif')
store = []
for f in files:
    print(f)
    year = int(f.split('/')[-1][0:4])
    oneyear = tif_timeseries_to_nc(f, 'total_bio', pd.Timestamp(year, 5, 1), pd.Timestamp(year, 9, 30))
    store.append(oneyear)
tbio_all = xr.merge(store)

# %%
tbio_all.to_netcdf('/scratch/williamson/normal_model_runs/total_bio_all.nc')

# %% [markdown]
# ## Productive Hours

# %%
files = glob.glob('/scratch/williamson/normal_model_runs/*_productive_hrs.tif')
store = []
for f in files:
    print(f)
    year = int(f.split('/')[-1][0:4])
    oneyear = tif_timeseries_to_nc(f, 'prod_hrs', pd.Timestamp(year, 5, 1), pd.Timestamp(year, 9, 30))
    store.append(oneyear)
phrs_all = xr.merge(store)
phrs_all.to_netcdf('/scratch/williamson/normal_model_runs/productive_hrs_all.nc')

# %%
