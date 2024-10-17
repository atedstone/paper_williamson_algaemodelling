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
# # Convert R model outputs as .grd files to netcdf

# %%
import geoutils as gu

import xarray as xr
import rioxarray
import matplotlib.pyplot as plt

import geopandas as gpd
import glob

# %%
gris_outline = gpd.read_file('/Users/tedstona/Library/CloudStorage/Dropbox/work/gis/gris_only_outline/greenland_icesheet_fix.shp')
gris_outline = gris_outline.to_crs(3413)


# %%

# %%
def grd_to_nc(filename):
    f = rioxarray.open_rasterio(filename)
    store = {}
    n = 0
    for name in f.long_name:
        tmp = f.isel(band=n)
        tmp = tmp.drop_vars('band')
        tmp.attrs['long_name'] = name
        store[name] = tmp
        n += 1
    ds = xr.Dataset(store)
    
    ds['x'] = ds['x'] * 1000
    ds['y'] = ds['y'] * 1000
    ds = ds.rio.write_crs('epsg:3413')
    
    year = int(filename.split('growth_data_')[1][0:4])
    ds.coords['time'] = year
    
    ds.to_netcdf(filename[:-3] + 'nc')
    return ds


# %%
files = glob.glob('/scratch/williamson/yearly_growth_raster_datasets/*.grd')
for f in files:
    print(f)
    _ = grd_to_nc(f)

# %%
ds

# %%
f = '/scratch/williamson/yearly_growth_raster_datasets/growth_data_2000_20000501_20000930.grd'

# %%
int(f.split('growth_data_')[1][0:4])

# %%
import cartopy.crs as ccrs
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)

plt.figure()
ax = plt.subplot(111, projection=crs)
ds.k.plot(ax=ax)
gris_outline.plot(ax=ax, color='none', edgecolor='y')

# %%
growth = xr.open_mfdataset('/scratch/williamson/yearly_growth_raster_datasets/*.nc', chunks=None)

# %%
