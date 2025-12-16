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
# From all the experiments of the ensemble, generate daily values of median, Q25 and Q75 population size for every cell in the model domain.

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
YR_ST = 2017
YR_END = 2017
START_POP = 179

MAR_REF = os.path.join(WORK_ROOT, 'MARv3.14.0-10km-daily-ERA5-2022.nc')

def add_id(ds):
    ds.coords['expid'] = ds.attrs['exp_id']
    return ds
    
def open_model_run(path):
    """ Load a model run at given path and update x/y coordinates to those from MAR reference run """
    run = xr.open_mfdataset(path, preprocess=add_id, concat_dim='expid', combine='nested')
    run['x'] = mar.x
    run['y'] = mar.y
    return run


# %%
if __name__ == '__main__':

    cluster = MyCluster()
    client = Client(cluster)
    cluster.scale(6)
    # Open 'reference' MAR run, mainly for the ice sheet mask
    mar = xr.open_dataset(MAR_REF)
    mar['x'] = mar['x'] * 1000
    mar['y'] = mar['y'] * 1000
    mar = mar.rio.write_crs('epsg:3413')

    for year in range(YR_ST, YR_END+1):
    
        print(year)
    
        # Open all years of this experiment run
        expt_outputs = open_model_run(os.path.join(WORK_ROOT, 'outputs/QMC', f'model_outputs_{year}_*.nc')).cum_growth
        #expt_outputs = expt_outputs.chunk({'TIME':153})

        med = expt_outputs.median(dim='expid')
        q25 = expt_outputs.quantile(.25, dim='expid').drop_vars('quantile')
        q75 = expt_outputs.quantile(.75, dim='expid').drop_vars('quantile')
        
        to_save = xr.Dataset({'q25':q25, 'med':med, 'q75':q75})
        to_save.to_netcdf(os.path.join(WORK_ROOT, RESULTS, f'model_outputs_{year}_XYT_dailypop_MIQR.nc'))
    
        expt_outputs.close()
        cluster = None

# %%
