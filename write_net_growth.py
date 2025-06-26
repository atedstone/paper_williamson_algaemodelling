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

# %%
import numpy as np
import pandas as pd
from datetime import datetime

import xarray as xr
import rioxarray
from copy import deepcopy
import os
import argparse
from glob import glob
import re


# %%
def calc_net_growth(fn, only_save=True, param_ploss=None):
    outputs = xr.open_dataset(fn)
    if param_ploss == None:
        param_ploss = outputs.attrs['param_ploss']
    Gn = outputs.today_prod - (param_ploss * outputs.cum_growth.shift(TIME=-1)).compute()
    Gn = Gn.where(Gn >= 0, 0)
    Gn.name = 'today_prod_net'
    outputs.close()
    Gn.to_netcdf(fn, mode='a')
    if only_save == False:
        return Gn
    else:
        return None


# %%
#files = glob('/work/atedstone/williamson/2025-05/*.nc')
#files = glob('/work/atedstone/williamson/outputs/sensitivity_ibio/*.nc')
#files = glob('/work/atedstone/williamson/outputs/sensitivity_ploss/*.nc')
#files = glob('/work/atedstone/williamson/outputs/sensitivity_snowdepth/*.nc')
#files = glob('/work/atedstone/williamson/outputs/sensitivity_light/*.nc')
files = glob('/work/atedstone/williamson/outputs/sensitivity_temp/*.nc')

# Acceptable ploss arguments: 
# - None (takes value from the NetCDF metadata) --- Use for QMC outputs
# - 'filename' (takes from filename) --- use for the sensitivity_ploss experiments
# - float value (used direct) --- use for all other sensitivity_* experiments
ploss = 0.1

n = len(files)
c = 1

print(f'{n} files found ... ')
print('First file:')
print(files[0])

for file in files:
    if c % 100 == 0:
        print(f'{c}/{n}')

    if ploss == 'filename':
        print(file.split('/')[-1])
        m = re.search('ploss0\.?[0-9]{0,2}', file.split('/')[-1])
        ploss = float(m.group(0).split('ploss')[1])
    
    _ = calc_net_growth(file, param_ploss=ploss)

    c += 1

# %%
