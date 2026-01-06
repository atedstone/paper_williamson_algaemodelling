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
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compile measured cell counts from the literature
#
# These cell counts are then used to undertake validation in `main_analysis.py`.

# %% trusted=true
import pandas as pd
import geopandas as gpd
import xarray as xr
from glob import glob
import rioxarray
import seaborn as sns
from shapely.geometry import Point
import os
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# %% trusted=true
# A 'reference' MAR output, mainly for the mask and georeferencing
MAR_REF = '/work/atedstone/williamson/MARv3.14.0-10km-daily-ERA5-2022.nc'
# Main run of algal growth model
MODEL_OUTPUTS_MAIN = '/work/atedstone/williamson/outputs/QMC/'
# Measured biomass datasets
MEAS_BIO = '/work/atedstone/williamson/glacier_algal_biomass_datasets/'
# Where CSV and figure files will be saved to
RESULTS = '/work/atedstone/williamson/results/'

# %% trusted=true
# Open 'reference' MAR run, mainly for the ice sheet mask
mar = xr.open_dataset(MAR_REF)
mar['x'] = mar['x'] * 1000
mar['y'] = mar['y'] * 1000
mar = mar.rio.write_crs('epsg:3413')

# %% trusted=true
## Principal study site locations
cmap = sns.color_palette('colorblind', n_colors=4)
coords = {
    'S6': {'p': Point((-49.38, 67.07)), 'color':cmap[0]},
    'UPE': {'p': Point((-53.55, 72.88)), 'color':cmap[1]},
    'Mittivak': {'p': Point((-37.8, 65.7)), 'color':cmap[2]},
    'South': {'p': Point((-46.8470, 61.1004)), 'color':cmap[3]}
}
pts_wgs84 = gpd.GeoDataFrame(
    { 
     'color':[coords[c]['color'] for c in coords]
    },
    index = [c for c in coords],
    geometry=[coords[c]['p'] for c in coords], 
    crs=4326
)
pts_ps = pts_wgs84.to_crs(3413)


# %% trusted=true
def ww_to_dw(cells_per_ml):
    """ Wet weight, e.g. cells per ml, to dry weight """
    # assuming 0.84 ng DW per cell (C.W. Feb 2024)
    return cells_per_ml * 0.84


# %% [markdown]
# ## Codify all available measurements 
#
# This section applies a consistent name for the biomass, allocates dates and coordinates, and applies appropriate averaging to each dataset.

# %% trusted=true
RENAME_TO = 'biomass'


# %% trusted=true
# Based on https://gist.github.com/tsemerad/5053378
# and https://gist.github.com/jeteon/89c41e4081d87b798d8006b16a52c695
def dms_to_dd(d, m, s):
    if d < 0:
        sign = -1
        d = np.abs(d)
    else:
        sign = 1
    dd = sign * (d + float(m)/60 + float(s)/3600)
    return dd


# %% trusted=true
# Import datasets
csvs = glob(os.path.join(MEAS_BIO, '*.csv'))
meas = {}
for csv in csvs:
    d = pd.read_csv(csv)
    csv_name = csv.split('/')[-1][:-4]
    if 'date' in d.columns:
        d['date'] = pd.to_datetime(d['date'], dayfirst=True)
    meas[csv_name] = d
    
# Load Stibal separately - directly from their Excel Workbook
meas['stibal_2017'] = pd.read_excel(os.path.join(MEAS_BIO, 'grl56634-sup-0002-2017gl075958_data_si.xlsx'), sheet_name='algal cells time series data')

# %% trusted=true
## South Greenland

## Chev data: explicitly noted as not spatially representative, so don't include here.
# chev provides date. Average all the sites on each day.
# bmc = 'ia.cells.ml'
# chev = meas['chevrollier_2022_2021_counts_south_gris'].groupby('date')[bmc].mean()
# chev = chev.reset_index()
# chev['date'] = pd.to_datetime(chev['date'], dayfirst=True)
# chev = chev.rename({bmc:RENAME_TO}, axis='columns')
# chev['geom'] = pts_wgs84.loc['South'].geometry
# chev['site'] = 'South'
# chev['d_id'] = 'chev'
# chev['biomass_col'] = bmc

# %% trusted=true
## South East Greenland 

# Halbach dataset
bmc = 'av.cells.ml'
halb = meas['halbach_2019_counts_se_greenland']
# Taken from Halbach et al. (2022) Table S1
halb_meta = {
    'he1_2':  {'geom': Point((dms_to_dd(-38,26,50), dms_to_dd(65,59,35))), 'date':dt.datetime(2019, 7, 26)},
    'mit1_2': {'geom': Point((dms_to_dd(-37,50, 2), dms_to_dd(65,41,39))), 'date':dt.datetime(2019, 7, 24)},
    'mit3_6': {'geom': Point((dms_to_dd(-37,50,25), dms_to_dd(65,41,38))), 'date':dt.datetime(2019, 7, 24)}
}
halb = halb.rename({bmc:RENAME_TO}, axis='columns')
halb = pd.merge(left=halb, right=pd.DataFrame(halb_meta).T, left_on='site', right_index=True)
halb['d_id'] = 'halb'
halb['biomass_col'] = bmc

# NO STANDARD DEVIATION REPORTED - SO DON'T INCLUDE IN OUR VALIDATION.
# Lutz Mittivakkat dataset does not provide exact dates, so force the mid-date of their campaign (6-23 July 2012)
# All values were acquired within a 1 km2 area, so also okay to force a single coordinate.
# bmc = 'cells.ml'
# lutz = meas['lutz_2012_se_greenland']
# lutz = lutz.rename({'grey_ice_samples':'site', bmc:RENAME_TO}, axis='columns')
# lutz['date'] = dt.datetime(2012, 7, 6+(23-6))
# # Estimate the centre point of Lutz et al. (2014) Figure 1.
# lutz['geom'] = Point((-37.87, 65.685))
# lutz['d_id'] = 'lutz'
# lutz['site'] = 'mit'
# lutz['biomass_col'] = bmc

# %% trusted=true
halb

# %% trusted=true
## S6

# stib provides date. Average all the samples on each day.
# All Stibal values were acquired at S6 (see paper).
bmc = 'cells/ml'
stib = meas['stibal_2017'].filter(items=['doy 2014', bmc], axis='columns').dropna()
stib['date'] = [dt.datetime.strptime(f'2014-{int(d)}', '%Y-%j') for d in stib['doy 2014']]

samples = stib.groupby('date')[bmc].count().reset_index()
samples = samples[bmc]
samples.name = 'n.samples'

sd = stib.groupby('date')[bmc].std().reset_index()
sd = sd[bmc]
sd.name = 'sd'

stib = stib.groupby('date').mean().reset_index()

stib = pd.concat([stib, samples, sd], axis=1)
stib['geom'] = pts_wgs84.loc['S6'].geometry
stib = stib.rename({bmc:RENAME_TO}, axis='columns')
stib = stib.drop(labels=['doy 2014'], axis='columns')
stib['site'] = 'S6'
stib['biomass_col'] = bmc
stib['d_id'] = 'stib'


# williamson 2016 @ S6 provides date
bmc = 'overall.cells.per.ml'
w16s6 = meas['williamson_2016_count_biomass_all_2016'].filter(items=['date', 'habitat', bmc], axis='columns').dropna()
#w16s6['date'] = [dt.datetime.strptime(d, '%d.%m.%y') for d in w16s6['date']]

# Retain only the low, medium and high habitats
w16s6 = w16s6[w16s6.habitat.isin(['l', 'm', 'h'])]
# Then only keep days on which all three habitats were sampled.
habs_check = w16s6.groupby(['date', 'habitat']).count()
hc = habs_check.unstack()
# On some days, only the high biomasss habitat was sampled. Only retain days on which all habitats were sampled.
valid_days = hc[
    (hc[('overall.cells.per.ml','h')] > 0) & 
    (hc[('overall.cells.per.ml','l')] > 0) & 
    (hc[('overall.cells.per.ml','m')] > 0)
]
w16s6 = w16s6[w16s6.date.isin(valid_days.index)]

samples = w16s6.groupby('date')[bmc].count().reset_index()
samples = samples[bmc]
samples.name = 'n.samples'

sd = w16s6.groupby('date')[bmc].std().reset_index()
sd = sd[bmc]
sd.name = 'sd'

w16s6 = w16s6.groupby('date')[bmc].mean().reset_index()
w16s6 = pd.concat([w16s6, samples, sd], axis=1)

w16s6['geom'] = pts_wgs84.loc['S6'].geometry
w16s6 = w16s6.rename({bmc:RENAME_TO}, axis='columns')
w16s6['d_id'] = 'w16_s6'
w16s6['site'] = 'S6'
w16s6['biomass_col'] = bmc

# %% trusted=true
w16s6

# %% trusted=true
## K-Transect

# Williamson space for time 2016.
# !!! check correct column is being used!
bmc = 'my.count'
w16k = meas['Williamson_dash_2016_space_for_time_counts'].filter(items=['date', 'habitat', bmc], axis='columns').dropna()
#w16k['date'] = [dt.datetime.strptime(d, '%d.%m.%y') for d in w16k['date']]
# Calculate averages for each site on each date
samples = w16k.groupby(['habitat', 'date'])[bmc].count().reset_index()
samples = samples[bmc]
samples.name = 'n.samples'

sd = w16k.groupby(['habitat', 'date'])[bmc].std().reset_index()
sd = sd[bmc]
sd.name = 'sd'

w16k = w16k.groupby(['habitat', 'date'])[bmc].mean().reset_index()
w16k = pd.concat([w16k, samples, sd], axis=1)

# Attach spatial information through a merge
# Coordinates taken from Williamson et al 2018 FEMS, matching to the 'habitat' names in the dataset.
w16k_meta = {
    'kanu': Point((-47.0154, 67.0003)),    # =S1a
    's1':   Point((-47.5433, 67.0631)),    # =S1b
    's2':   Point((-48.3064, 67.0571)),
    's3':   Point((-48.8929, 67.0913)),
    'h':    pts_wgs84.loc['S6'].geometry
}
w16k_meta = pd.Series(w16k_meta, name='geom')
w16k = pd.merge(left=w16k, right=pd.Series(w16k_meta), left_on='habitat', right_index=True)

w16k = w16k.rename({bmc:RENAME_TO, 'habitat':'site'}, axis='columns')
w16k['d_id'] = 'w16_dash'
w16k['biomass_col'] = bmc

# %% trusted=true
w16k

# %% trusted=true
## Kanger margin

# DO NOT INCLUDE AS SAMPLING STRATEGY WAS EXPLICITLY FOCUSED ON HIBIO ICE
# Based on Williamson et al. 2021 Frontiers: collected 8-10 August 2019, at Point 660.
# bmc = 'total_cells_ml'
# w21k = meas['Williamson_2021_sw_2019_algal_count'].filter(items=[bmc], axis='columns')
# samples = w21k.count()
# sd = w21k.std()
# print(w21k.median())
# w21k = w21k.mean()

# w21k['sd'] = sd.iloc[0]
# w21k['n.samples'] = int(samples.iloc[0])
# w21k['date'] = dt.datetime(2019, 8, 9)
# # Google Maps approx coordinate a few hundred metres inland of Pt 660
# w21k['geom'] = Point((-50.027204, 67.155493))
# w21k['d_id'] = 'w19_660'
# w21k = w21k.rename({bmc:RENAME_TO})
# w21k['site'] = 'P660'
# w21k['biomass_col'] = bmc
# w21k = pd.DataFrame(w21k).T

# %% trusted=true
## Upernavik

# We already have coordinate. Check units? 
bmc = 'cells.per.ml'
upe = meas['williamson_2018_upe_u_cell_counts'].filter(items=[bmc], axis='columns')
samples = upe.count()
sd = upe.std()
upe = upe.mean()
upe['sd'] = sd.iloc[0]
upe['n.samples'] = int(samples.iloc[0])
upe['date'] = dt.datetime(2018, 7, 26)
upe['geom'] = pts_wgs84.loc['UPE'].geometry
upe = upe.rename({bmc:RENAME_TO})
upe['site'] = 'UPE'
upe['biomass_col'] = bmc
upe['d_id'] = 'upe'
upe = pd.DataFrame(upe).T

# %% trusted=true
upe

# %% trusted=true
## Onuma Qanaaq
import re

onuma_store = []
# Treat the CSV loading separately as these are bunch of files in their own folder
csvs = glob(os.path.join(MEAS_BIO, 'onuma_et_al_2022_point_datasets', '*cell_vol*.csv'))
for csv in csvs:
    data = pd.read_csv(csv, parse_dates=['date'], dayfirst=True)
    site = re.search('S[1-4]{1}', csv).group(0)
    data['site'] = 'QAN-' + site
    onuma_store.append(data)

onuma = pd.concat(onuma_store)
onuma = onuma.filter(items=['date', 'Total_ave', 'Total_sd', 'site'])
onuma['biomass_col'] = 'Total_ave'
onuma = onuma.rename(columns={'Total_ave':RENAME_TO, 'Total_sd':'sd'})
onuma['n.samples'] = 4
onuma['d_id'] = 'onum'
onuma = onuma.reset_index()

# Add coordinates
onuma_pos = pd.read_csv(os.path.join(MEAS_BIO, 'onuma_et_al_2022_point_datasets', 'onuma_site_coords_eyeballed.csv'))
onuma_pos['site'] = 'QAN-' + onuma_pos['site']
onuma_pos = gpd.GeoDataFrame(onuma_pos, geometry=gpd.points_from_xy(onuma_pos.lon, onuma_pos.lat, crs=4326))
onuma_pos = onuma_pos.filter(items=['site', 'geometry'], axis='columns')
onuma = pd.merge(onuma, onuma_pos, left_on='site', right_on='site', how='left')
onuma = onuma.rename(columns={'geometry':'geom'})

# %% trusted=true
onuma.head()

# %% trusted=true
## Stibal 2013 pan-GrIS counts

additional_stibal = pd.read_csv(os.path.join(MEAS_BIO, 'stibal2017_counts2013.csv'), parse_dates=['date'], dayfirst=True)

# convert coords to decimal degrees
lat_dd = additional_stibal.lat_d + (additional_stibal.lat_m / 60)
lon_dd = additional_stibal.lon_d + (additional_stibal.lon_m / 60)
additional_stibal['geom'] = gpd.points_from_xy(lon_dd, lat_dd, crs='4326')

#additional_stibal['lon'] = lon_dd
#additional_stibal['lat'] = lat_dd
additional_stibal = additional_stibal.rename({'abundance_cells_ml':RENAME_TO, 'abundance_cells_ml_sd':'sd'}, axis='columns')
additional_stibal = additional_stibal.filter(items=['site', RENAME_TO, 'sd', 'date', 'geom'], axis='columns')
additional_stibal['d_id'] = 'stib13'
additional_stibal['biomass_col'] = 'abundance_cells_ml'
additional_stibal['n.samples'] = 3
additional_stibal = additional_stibal.reset_index()

# %% trusted=true
additional_stibal.head()

# %% trusted=true
## Join all datasets together

measurements = pd.concat([
#    chev, # removed - see note(1)
    halb,
#    lutz,
    stib,
    w16s6,
    w16k,
#    w21k, # removed - see note(2)
    upe,
    onuma,
    additional_stibal
], axis=0).reset_index()

# %% [markdown]
# *Note(1)* Chev. 2022, page 3: "The targeted surfaces were chosen to be roughly homogeneous on a wider surface in order to upscale the results for  1m2 areas, but are not representative of wider surfaces".
#
# *Note(2)* 2024-02-09: cells/ml in spreadsheet are one order of magnitude higher than all other datasets, suspect
#

# %% trusted=true
measurements

# %% trusted=true
# Clean the joined dataset
measurements = measurements.drop(labels=['index'], axis='columns')
measurements['biomass'] = pd.to_numeric(measurements['biomass'])

# Convert to Polar Stereo to match MAR
measurements = gpd.GeoDataFrame(measurements, geometry='geom', crs='epsg:4236')
measurements = measurements.to_crs(3413)

# %% trusted=true
pd.to_datetime(measurements.date).dt.year.unique()

# %% trusted=true
#measurements.to_file(os.path.join(RESULTS,'measurements_merged.gpkg'))

# %% trusted=true
measurements.to_excel(os.path.join(RESULTS, 'measurements_merged.xlsx'))

# %% trusted=true
