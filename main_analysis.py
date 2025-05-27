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
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Williamson & Tedstone GrIS Algal Growth: Model Analysis
#
# This Notebook analyses outputs from the Python implementation of the algal growth model found in `algal_growth_model.py`. It produces the figures for the manuscript.
#
# Outputs of Python model:
# - `daily_prod_hrs`
# - `cum_growth`
# - `today_prod`
#
# AT, Jan/Feb 2024

# %% trusted=true
# Main libraries
import os
from glob import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import statsmodels.api as sm
import datetime as dt
import geoutils as gu

# %% trusted=true
# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import dates
from matplotlib import rcParams

import cartopy.crs as ccrs
import seaborn as sns

sns.set_context('paper')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9

# %% trusted=true
# If running on a SLURM HPC cluster then spin up compute nodes
#from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import LocalCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(n=4)
client = Client(cluster)

# %% trusted=true
client

# %% [markdown]
# ## Paths/Settings

# %% trusted=true
# beo05
#INPUTS_PATH = '/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-*_05-09_ALGV.nc'
# WORK_ROOT = '/flash/tedstona/williamson/' 
# # A 'reference' MAR output, mainly for the mask and georeferencing
# MAR_REF = '/flash/tedstona/MARv3.14-ERA5-10km/MARv3.14.0-10km-daily-ERA5-2017.nc'
# # GrIS drainage basins
# BASINS_FILE = '/flash/tedstona/L0data/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions.shp'
# OUTLINE_FILE = '/flash/tedstona/L0data/gris_only_outline/greenland_icesheet_fix.shp'
# GRIS_BBOX_FILE = '/flash/tedstona/L0data/greenland_area_bbox/greenland_area_bbox.shp'
# SURF_CONTOURS_FILE = os.path.join('/flash/tedstona/L0data/GIMPDEM', 'gimpdem_90m_v01.1_EPSG3413_grisonly_contours_2km_i500', 'contour.shp')
DEM_FILE = os.path.join('/flash/tedstona/L0data/GIMPDEM', 'gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif')

# unil mac
INPUTS_PATH = None
WORK_ROOT = '/scratch/williamson/'
# A 'reference' MAR output, mainly for the mask and georeferencing
MAR_REF = '/Users/atedston/scratch/williamson/MARv3.14.0-10km-daily-ERA5-2022.nc'
# GrIS drainage basins
BASINS_FILE = '/Users/atedston/Dropbox/work/gis/doi_10.7280_D1WT11__v1/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions.shp'
OUTLINE_FILE = '/Users/atedston/Dropbox/work/gis/gris_only_outline/greenland_icesheet_fix.shp'
LAND_AREAS_FILE = '/Users/atedston/Dropbox/work/gis/ne_10m_land.shp'
GRIS_BBOX_FILE = '/Users/atedston/Dropbox/work/gis/greenland_area_bbox/greenland_area_bbox.shp'
SURF_CONTOURS_FILE = '/Users/atedston/Dropbox/work/gis/gimp_contours/gimpdem_90m_v01.1_EPSG3413_grisonly_contours_2km_i500/contour.shp'
DEM_FILE = os.path.join(WORK_ROOT, 'gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif')

# -----------------------------------------------------------------------------------
# Algal growth model sensitivity runs (accessed by wildcard in script)
MODEL_OUTPUTS_SENS_IBIO = os.path.join(WORK_ROOT, 'outputs/sensitivity_ibio')
MODEL_OUTPUTS_SENS_PLOS = os.path.join(WORK_ROOT, 'outputs/sensitivity_ploss')
MODEL_OUTPUTS_SENS_TEMP = os.path.join(WORK_ROOT, 'outputs/sensitivity_temp')
MODEL_OUTPUTS_SENS_LIGH = os.path.join(WORK_ROOT, 'outputs/sensitivity_light')
MODEL_OUTPUTS_SENS_SNOW = os.path.join(WORK_ROOT, 'outputs/sensitivity_snowdepth')

# Main run of algal growth model
MODEL_OUTPUTS_MAIN = os.path.join(WORK_ROOT, 'outputs/main_outputs')

# Save location for figures, CSV files
RESULTS = os.path.join(WORK_ROOT, 'results')

# %% trusted=true
# Growth model start and end
YR_ST = 2000
YR_END = 2022

# Default parameters to use during analysis unless otherwise specified
START_POP = 179
P_LOSS = 0.10
SNOW_DEPTH = 0.01
SURF_T = 0.5
LIGHT_T = 10


# %% [markdown]
# ## Analysis functions

# %% trusted=true
def open_model_run(path):
    """ Load a model run at given path and update x/y coordinates to those from MAR reference run """
    run = xr.open_mfdataset(path)
    run['x'] = mar.x
    run['y'] = mar.y
    return run

def ww_to_dw(cells_per_ml):
    """ Wet weight, e.g. cells per ml, to dry weight """
    # assuming 0.84 ng DW per cell (C.W. Feb 2024)
    return cells_per_ml * 0.84

def dw_to_ww(dw):
    """ Dry weight (e.g. from model) to cells per ml """
    # assuming 0.84 ng DW per cell (C.W. Feb 2024)
    return dw / 0.84

def to_carbon(x, grid_km=10):
    """ 
    x : DataArray. Provide in units of cells per ml
    grid_km : size of grid cell in kilometres
     
    returns: kg carbon per model cell
    """

    #convert from cells per ml to pg C ml assuming 106 pg C per cell
    x = x * 106

    #- pg C ml to pg C per L
    x = x * 1000

    #- pg C per l to pg C per m2 #using conversion from Williamson et al. 2018
    x = x * 1.061

    #- pg C per m2 to pg C per km2
    x = x * 10**6

    #- pg C per km2 to kg of C per km2
    x = x * 10**-15

    #total kg.C.per pixel
    #total kg of C per km2 * number of km2 per pixel 
    total_kg_C_pixel = x * grid_km**2
    
    return total_kg_C_pixel



# %% [markdown]
# ## Plotting functions

# %% trusted=true
def label_panel(ax, letter, xy=(0.04,0.93)):
    ax.annotate(letter, fontweight='bold', xy=xy, xycoords='axes fraction',
           horizontalalignment='left', verticalalignment='top', fontsize=9)


# %% [markdown]
# ## Load GIS data

# %% trusted=true

# %% trusted=true
gris_outline = gpd.read_file(OUTLINE_FILE)
gris_outline = gris_outline.to_crs(3413)

# World land areas
greenland = gpd.read_file(LAND_AREAS_FILE)
# Crop world land areas to Greenland and surrounding areas
bbox = gpd.read_file(GRIS_BBOX_FILE).to_crs(3413)
just_greenland = gpd.clip(greenland.to_crs(3413), bbox)

# Manually isolate contiguous Greenland polygon from the main multi-polygon.
jg = just_greenland.filter(items=[0], axis=0)
jgg = jg.loc[0].geometry
jgg_poly = list(jgg.geoms)[9]
jgg_gdf = gpd.GeoDataFrame({'ix':[1,]}, geometry=[jgg_poly], crs=3413)

# Surface elevation contours
# Source of process: atedstone:paper_rlim_detection_repo/plot_map_decadal_change.py
# shp : greenland_icesheet.shp from Horst, run through geopandas simplify and buffer operations as follows:
## shp = gpd.read_file('/flash/tedstona/L0data/gris_only_outline/greenland_icesheet_fix.shp')
## shp_simpl = shp.geometry.simplify(5000).buffer(0)
## shp_simpl.plot()
## shp_simpl.to_file('/flash/tedstona/L0data/gris_only_outline/greenland_icesheet_simplify5000_buffer0.shp')
# Then gdal warp:
# gdalwarp -cutline ../gris_only_outline/greenland_icesheet_simplify5000_buffer0.shp GimpIceMask_90m_v1.1_epsg3413.tif GimpIceMask_90m_v1.1_epsg3413_gris.tif
# gdal_calc.py -A gimpdem_90m_v01.1_EPSG3413.tif -B ../GIMPMASK/GimpIceMask_90m_v1.1_epsg3413_gris.tif --calc=A*B --outfile=gimpdem_90m_v01.1_EPSG3413_grisonly.tif
# gdal_fillnodata.py gimpdem_90m_v01.1_EPSG3413_grisonly.tif gimpdem_90m_v01.1_EPSG3413_grisonly_filled.tif
# gdalwarp -tr 2000 2000 gimpdem_90m_v01.1_EPSG3413_grisonly_filled.tif gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif
# gdal_contour -i 500 gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif gimpdem_90m_v01.1_EPSG3413_grisonly_contours_2km_i500 -f 'ESRI Shapefile' -a elev
surf_contours = gpd.read_file(SURF_CONTOURS_FILE).to_crs(3413)

# %% trusted=true
# Open 'reference' MAR run, mainly for the ice sheet mask
mar = xr.open_dataset(MAR_REF)
mar['x'] = mar['x'] * 1000
mar['y'] = mar['y'] * 1000
mar = mar.rio.write_crs('epsg:3413')

# %% trusted=true
# Sampling coordinates and colours
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
#pts.index = pts.site_name
pts_ps

# %% trusted=true
# Sanity-check the points
fig, ax = plt.subplots()
mar.MSK.plot(ax=ax)
pts_ps.plot(ax=ax, marker='x', color='r')

# %% trusted=true
basins = gpd.read_file(BASINS_FILE)
basins.index = basins.SUBREGION1

# %% trusted=true
basins


# %% [markdown]
# ## Quasi Monte Carlo analysis

# %% trusted=true scrolled=true
def analyse_qmc(year, s6=False):
    # Time-series results
    qmc = {}
    # Bloom max of each experiment
    bmax = []
    # Running standard deviation of bmax
    sd = []
    # Running standard deviation of sd
    sd_sd = []

    for exp in range(1, 513):
        print(exp)
        # Open the numbered experiment
        r = open_model_run(os.path.join(WORK_ROOT, '2025-05', f'qmc_{year}', f'model_outputs_{year}_exp{exp}.nc'))
        # If subset to s6 requested then do this now
        if s6:
            r = r.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest')
        # Identify pixels to include according to whether they saw any growth in the season
        incl = r.cum_growth.where(r.cum_growth > START_POP).count(dim='TIME')
        # Reduce to 1-D timeseries
        ts = r.cum_growth.where(mar.MSK > 50).where(incl > 1).median(dim=('x','y')).to_pandas() #.where(r.cum_growth > START_POP)
        # Append the metrics
        bmax.append(ts.max())
        sd.append(np.std(bmax))
        sd_sd.append(np.std(sd))
        # Save the time series
        qmc[exp] = ts
        qmc[exp].name = f'exp{exp}'

    return (qmc, bmax, sd, sd_sd)


# %% trusted=true scrolled=true
qmc00, bmax00, sd00, sd_sd00 = analyse_qmc(2000)

# %% trusted=true
plt.plot(sd)
plt.plot(sd_sd)

# %% trusted=true
from copy import deepcopy
bmax_gris = deepcopy(bmax)

# %% trusted=true
plt.hist(bmax, bins=np.arange(5000,30000,2000))

# %% trusted=true
sns.kdeplot(bmax_gris, fill=True, label='2012')
sns.kdeplot(bmax00, fill=True, label='2000')
plt.legend()

# %% trusted=true
qmc12pd = pd.concat(qmc12, axis=1)

# %% trusted=true scrolled=true
qmc12_s6, bmax12_s6, sd12_s6, sd_sd12_s6 = analyse_qmc(2012, s6=True)

# %% trusted=true scrolled=true
pd.DataFrame(qmc12_s6).plot(legend=False, alpha=0.1, color='tab:blue')

# %% trusted=true
qmc12pd.plot(alpha=0.1, color='tab:blue', legend=False)
qmc12pd.mean(axis=1).plot(color='k', linewidth=1.5)
qmc12pd.quantile(0.05, axis=1).plot(color='k', linewidth=1)
qmc12pd.quantile(0.95, axis=1).plot(color='k', linewidth=1)
sns.despine()


# %% trusted=true scrolled=true
qmc12pd.plot(alpha=0.1, color='tab:blue', legend=False)
qmc12pd.mean(axis=1).plot(color='k', linewidth=1.5)
qmc12pd.quantile(0.05, axis=1).plot(color='k', linewidth=1)
qmc12pd.quantile(0.95, axis=1).plot(color='k', linewidth=1)
sns.despine()qmc12pd.plot(alpha=0.1, color='tab:blue', legend=False)
qmc12pd.mean(axis=1).plot(color='k', linewidth=1.5)
qmc12pd.quantile(0.05, axis=1).plot(color='k', linewidth=1)
qmc12pd.quantile(0.95, axis=1).plot(color='k', linewidth=1)
sns.despine()

# %% trusted=true
incl.plot()

# %% [markdown]
# ---
# ## Sensitivity analysis

# %% [markdown]
# Even though the model sensitivity was tested over the full time frame (2000-2022), we don't need to use all these years - just the max and min should be sufficient.
#
# ### Identify the min and max years
#
# What metric to use? Options look something like:
#
# - MB, SMB
# - Melt extent (PMR)
# - MAR temperature <--
# - MAR productive hours
#

# %% trusted=true
mar_alg_inputs = xr.open_mfdataset(INPUTS_PATH)
mar_alg_inputs = mar_alg_inputs.squeeze()
mar_alg_inputs = mar_alg_inputs.rename({'Y19_288':'y', 'X14_163':'x'})

# %% trusted=true
TT_summer = mar_alg_inputs.TT.where(mar_alg_inputs['TIME.season'] == 'JJA').where(mar_alg_inputs.MSK > 50).resample(TIME='1AS').mean().mean(dim=('x','y')).compute()

# %% trusted=true
TT_summer.to_pandas().sort_values()

# %% trusted=true
TT_summer.plot()

# %% trusted=true
year_coldest = TT_summer.idxmin().values.astype('datetime64[Y]').astype(int) + 1970
year_coldest

# %% trusted=true
year_warmest = TT_summer.idxmax().values.astype('datetime64[Y]').astype(int) + 1970
year_warmest

# %% trusted=true
yr_recent_coldest = 2022
yr_recent_warmest = 2019

# %% [markdown]
# ### Sensitivity to snow depth

# %% trusted=true
regenerate = True
if regenerate:
    depths = [0.01, 0.05, 0.10, 0.2, 0.4, 0.8]
    site_store = {}
    gris_store = {}
    for d in depths:
        print(d)
        sens_sd = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_SNOW, f'*_snow{d}.nc'))
        gris_store[d] = sens_sd.cum_growth.where(mar.MSK > 50).where(sens_sd.cum_growth > START_POP).median(dim=('x','y')).to_pandas()
        v = sens_sd.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[d] = v
        
    sd_gris = pd.DataFrame(gris_store)
    sd_gris.to_csv(os.path.join(RESULTS, 'sens_snowdepth_time_series_gris.csv'))
    
    sd_site = pd.DataFrame(site_store)
    sd_site.to_csv(os.path.join(RESULTS, 'sens_snowdepth_time_series_S6.csv'))
else:
    pass

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5, 4))


## snow depth

# S6

sd_site[str(year_coldest)].plot(
    ax=axes[0,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,0].set_title('S6')

sd_site[str(year_warmest)].plot(
    ax=axes[1,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)

# Ice sheet wide

sd_gris[str(year_coldest)].plot(
    ax=axes[0,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,1].set_title('Ice Sheet')

sd_gris[str(year_warmest)].plot(
    ax=axes[1,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)


handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(
    loc=(1.05,0.05), 
    frameon=False, 
    handlelength=1
)



for ax in axes.flatten():
    ax.set_ylim(0, 130000)
    ax.set_xlabel('')

plt.subplots_adjust(hspace=0.4)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_snowdepth.pdf'), bbox_inches='tight')

# %% [markdown]
# ### Sensitivity to light

# %% trusted=true
regenerate = True
if regenerate:
    lights = [1, 10, 100, 200]
    site_store = {}
    gris_store = {}
    for li in lights:
        print(li)
        sens_li = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_LIGH, f'*_light{li}.nc'))
        gris_store[li] = sens_li.cum_growth.where(mar.MSK > 50).where(sens_li.cum_growth > START_POP).median(dim=('x','y')).to_pandas()
        v = sens_li.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[li] = v
        
    li_gris = pd.DataFrame(gris_store)
    li_gris.to_csv(os.path.join(RESULTS, 'sens_light_time_series_gris.csv'))
    
    li_site = pd.DataFrame(site_store)
    li_site.to_csv(os.path.join(RESULTS, 'sens_light_time_series_S6.csv'))
else:
    pass

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5, 4))


## snow depth

# S6

li_site[str(year_coldest)].plot(
    ax=axes[0,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,0].set_title('S6')

li_site[str(year_warmest)].plot(
    ax=axes[1,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)

# Ice sheet wide

li_gris[str(year_coldest)].plot(
    ax=axes[0,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,1].set_title('Ice Sheet')

li_gris[str(year_warmest)].plot(
    ax=axes[1,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)


handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(
    loc=(1.05,0.05), 
    frameon=False, 
    handlelength=1
)



for ax in axes.flatten():
    ax.set_ylim(0, 130000)
    ax.set_xlabel('')

plt.subplots_adjust(hspace=0.4)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_light.pdf'), bbox_inches='tight')

# %% [markdown]
# ### Sensitivity to temperature

# %% trusted=true
regenerate = True
if regenerate:
    temps = [0, 0.25, 0.5, 1.0]
    site_store = {}
    gris_store = {}
    for t in temps:
        print(t)
        sens_t = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_TEMP, f'*_temp{t}.nc'))
        gris_store[t] = sens_t.cum_growth.where(mar.MSK > 50).where(sens_t.cum_growth > START_POP).median(dim=('x','y')).to_pandas()
        v = sens_t.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[t] = v
        
    t_gris = pd.DataFrame(gris_store)
    t_gris.to_csv(os.path.join(RESULTS, 'sens_surft_time_series_gris.csv'))
    
    t_site = pd.DataFrame(site_store)
    t_site.to_csv(os.path.join(RESULTS, 'sens_surft_time_series_S6.csv'))
else:
    pass

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5, 4))


## temperature

# S6

t_site[str(year_coldest)].plot(
    ax=axes[0,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,0].set_title('S6')

t_site[str(year_warmest)].plot(
    ax=axes[1,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)

# Ice sheet wide

t_gris[str(year_coldest)].plot(
    ax=axes[0,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,1].set_title('Ice Sheet')

t_gris[str(year_warmest)].plot(
    ax=axes[1,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)


handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(
    loc=(1.05,0.05), 
    frameon=False, 
    handlelength=1
)



for ax in axes.flatten():
    ax.set_ylim(0, 130000)
    ax.set_xlabel('')

plt.subplots_adjust(hspace=0.4)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_temperature.pdf'), bbox_inches='tight')

# %% [markdown]
# ### Sensitivity to starting biomass term

# %% trusted=true
# Ice-sheet-wide quantile, annual values
# regenerate = False
# if regenerate:
#     ibio = [17.900000000000002, 179, 895, 1790]
#     store = {}
#     for ib in ibio:
#         print(ib)
#         sens_ibio = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_IBIO, f'*_ibio{ib}.nc'))
#         sens_ibio['x'] = mar.x
#         sens_ibio['y'] = mar.y
#         store[ib] = sens_ibio.cum_growth.where(mar.MSK > 50).where(sens_ibio.cum_growth > ib).resample(TIME='1AS').quantile(0.9, dim='TIME').median(dim=('x','y')).to_pandas()
#     ibio_quant90 = pd.DataFrame(store)
#     ibio_quant90.to_csv(os.path.join(RESULTS, 'ibio_quant90.csv'))
# else:
#     ibio_quant90 = pd.read_csv(os.path.join(RESULTS, 'ibio_quant90.csv'), index_col=0, parse_dates=True)
#     ibio_quant90.columns = [int(np.round(float(c))) for c in ibio_quant90.columns]

# %% trusted=true
# Here, do the same analysis but with the more straightforward annual median of all maximum pop sizes
# And also pull out time series of individual sites.
ibio = [17.900000000000002, 179, 895, 1790]
store = {}
site_store = {}
for ib in ibio:
    print(ib)
    sens_ibio = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_IBIO, f'*_ibio{ib}.nc'))
    sens_ibio['x'] = mar.x
    sens_ibio['y'] = mar.y
    store[ib] = sens_ibio.cum_growth.where(mar.MSK > 50).where(sens_ibio.cum_growth > ib).resample(TIME='1AS').max(dim='TIME').median(dim=('x','y')).to_pandas()
    
    v = sens_ibio.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
    site_store[ib] = v
ibio_max = pd.DataFrame(store)
ibio_s6 = pd.DataFrame(site_store)

# %% trusted=true
sens_ibio

# %% trusted=true
# Ice-sheet-wide bloom time series
ibio = [17.900000000000002, 179, 895, 1790]
store = {}
for ib in ibio:
    print(ib)
    sens_ibio = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_IBIO, f'*_ibio{ib}.nc'))
    sens_ibio['x'] = mar.x
    sens_ibio['y'] = mar.y
    store[ib] = sens_ibio.cum_growth.where(mar.MSK > 50).where(sens_ibio.cum_growth > ib).median(dim=('x','y')).to_pandas()
ibio_ts = pd.DataFrame(store)

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5, 4))


## Starting population

# S6

# Change names for legend
ibio_s6.columns = [int(np.round(float(c))) for c in ibio_s6.columns]

ibio_s6[str(year_coldest)].plot(
    ax=axes[0,0],
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)
axes[0,0].set_title('S6')

handles, labels = axes[0,0].get_legend_handles_labels()
axes[0,0].legend(
    handles[::-1], labels[::-1],     
    loc=(0.05,0.4), 
    frameon=False, 
#    title='Start pop.',
    handlelength=1
)

ibio_s6[str(year_warmest)].plot(
    ax=axes[1,0],
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)

# Ice sheet wide

ibio_ts[str(year_coldest)].plot(
    ax=axes[0,1],
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)
axes[0,1].set_title('Ice Sheet')

ibio_ts[str(year_warmest)].plot(
    ax=axes[1,1],
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)


for ax in axes.flatten():
    ax.set_ylim(0, 20000)
    ax.set_xlabel('')

plt.subplots_adjust(hspace=0.4)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_startpop.pdf'), bbox_inches='tight')

# %% [markdown] tags=[]
# ### Sensitivity to loss term

# %% trusted=true
regenerate = True
if regenerate:
    ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
    site_store = {}
    gris_store = {}
    for pl in ploss:
        print(pl)
        # sens_ploss = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_PLOS, f'*_ploss{pl}.nc'))
        # sens_ploss['x'] = mar.x
        # sens_ploss['y'] = mar.y
        sens_ploss = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_PLOS, f'*_ploss{pl}.nc'))
        gris_store[pl] = sens_ploss.cum_growth.where(mar.MSK > 50).where(sens_ploss.cum_growth > START_POP).median(dim=('x','y')).to_pandas()
        v = sens_ploss.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[pl] = v
        
    ploss_gris = pd.DataFrame(gris_store)
    ploss_gris.to_csv(os.path.join(RESULTS, 'ploss_time_series_gris.csv'))
    # Change names for legend
    ploss_gris.columns = [str(int(float(c)*100))+'%' for c in ploss_gris.columns]

    ploss_site = pd.DataFrame(site_store)
    ploss_site.to_csv(os.path.join(RESULTS, 'ploss_time_series_S6.csv'))
else:
    pass
    # ploss_quant90 = pd.read_csv(os.path.join(RESULTS, 'ploss_quant90.csv'), index_col=0, parse_dates=True)
    # percs = [str(int(float(c)*100))+'%' for c in ploss_quant90.columns]
    # ploss_quant90.columns = percs

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4.5, 4))


## Population loss

# S6

ploss_site[str(year_coldest)].plot(
    ax=axes[0,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,0].set_title('S6')

ploss_site[str(year_warmest)].plot(
    ax=axes[1,0],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)

# Ice sheet wide

ploss_gris[str(year_coldest)].plot(
    ax=axes[0,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
axes[0,1].set_title('Ice Sheet')

ploss_gris[str(year_warmest)].plot(
    ax=axes[1,1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)


handles, labels = axes[1,1].get_legend_handles_labels()
axes[1,1].legend(
    loc=(1.05,0.05), 
    frameon=False, 
    handlelength=1
)



for ax in axes.flatten():
    ax.set_ylim(0, 130000)
    ax.set_xlabel('')

plt.subplots_adjust(hspace=0.4)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_ploss.pdf'), bbox_inches='tight')

# %% [markdown]
# ---
# ## Main model outputs

# %% trusted=true
main_outputs = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_MAIN, '*.nc'))
main_outputs['x'] = mar.x
main_outputs['y'] = mar.y

# %% trusted=true
main_outputs

# %% trusted=true
# Find annual last productive day in each grid cell.
store = []
for year in range(YR_ST, YR_END+1):
    d = main_outputs.sel(TIME=str(year))
    doy = d.TIME.dt.dayofyear
    d['TIME'] = doy
    lpd = d.today_prod.cumsum(dim='TIME').idxmax(dim='TIME')
    lpd.coords['TIME'] = pd.Timestamp(year, 1, 1)
    lpd = lpd.expand_dims({'TIME':[pd.Timestamp(year, 1, 1)]})
    store.append(lpd.compute())

last_prod_doy = xr.concat(store, dim='TIME') 
# Forward-fill the end DOY to cover the full year, each year
# And we manually force this to continue to the end of 2022.
last_prod_doy = last_prod_doy.reindex(TIME=pd.date_range(last_prod_doy['TIME'].isel(TIME=0).values, '2023-01-01', freq='1D'), method='ffill')

# %% [markdown]
# ### Locations time series

# %% trusted=true
# Extract time series of population size
store = {}
for ix, row in pts_ps.iterrows():
    v = main_outputs.cum_growth.sel(x=row.geometry.x, y=row.geometry.y, method='nearest').to_pandas()
    store[ix] = v
ts = pd.DataFrame(store)

# %% trusted=true
ts

# %% trusted=true
totals.T

# %% trusted=true
pd.DataFrame(store_totals, columns=['site', 'biomass_sum'])

# %% trusted=true
pts_ps['color'].values

# %% trusted=true
#year = yr_recent_warmest
year = yr_recent_coldest

fig, axes = plt.subplots(figsize=(4,2.3), nrows=1, ncols=2, width_ratios=[0.75, 0.25])

# Time series
store_totals = []
for site in ts.columns:
    data = ts.loc[str(year)][site]
    axes[0].plot(data.index, data, c=pts_ps.loc[site]['color'], label=site, linewidth=1.2)
    doy_end = int(last_prod_doy.sel(x=pts_ps.loc[site].geometry.x, y=pts_ps.loc[site].geometry.y, method='nearest').sel(TIME=str(year)).values[0])
    date_end = dt.datetime.strptime(f'{year}-{doy_end}', '%Y-%j')
    axes[0].plot(date_end, data.loc[date_end], 'd', c=pts_ps.loc[site]['color'])
    
    store_totals.append(data.loc[:date_end].sum())

m = dates.MonthLocator() 
axes[0].xaxis.set_major_locator(m)
axes[0].xaxis.set_major_formatter(dates.DateFormatter('1 %b'))

axes[0].legend(frameon=False)
axes[0].set_ylabel('Biomass (ng DW ml$^{-1}$)')
axes[0].set_ylim(0,25000)
axes[0].set_xlabel(year)

# Totals bar chart    
totals = pd.DataFrame([ts.columns, store_totals])
totals = totals.T
totals.columns = ['site', 'biomass_sum']
totals.index = totals.site
sns.barplot(x=totals.site, y=totals.biomass_sum, palette=pts_ps['color'])
#totals.plot.bar(ax=axes[1], legend=False, color=pts_ps['color'].values)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].set_ylim(0, 2.5e6)

sns.despine()
plt.subplots_adjust(wspace=0.25)

plt.savefig(os.path.join(RESULTS, f'fig_timeseries_{year}.pdf'), bbox_inches='tight')

# %% trusted=true
totals

# %% [markdown]
# ### Map small versus large bloom years

# %% trusted=true
# Calculate metrics
valid_growth = main_outputs.cum_growth.where(main_outputs.TIME.dt.dayofyear <= last_prod_doy).where(main_outputs.cum_growth > START_POP).where(mar.MSK > 50)
annual_g_sum = valid_growth.resample(TIME='1AS').sum(dim='TIME')
annual_g_sum = annual_g_sum.where(annual_g_sum > 0).compute()
annual_g_max = valid_growth.resample(TIME='1AS').max(dim='TIME').compute()


# %% trusted=true
## To look at all years together, uncomment these lines and run cell
# norm = colors.LogNorm(vmin=179, vmax=2.5e6)
# annual_g_sum.plot(col='TIME', col_wrap=4, norm=norm)

# %% trusted=true
def plot_contours(ax):
    dem = gu.Raster(DEM_FILE)
    x,y = dem.coords()
    CS = ax.contour(x, y, 
                dem.data, 
                levels=np.arange(1000, 3500, 500),
               colors='#c2c2c2', linewidths=0.4, zorder=150, alpha=0.7)
    yc = -2.4e6
    ax.clabel(CS, CS.levels, inline=True, fontsize=6,
             manual=[(-150000, -1.25e6), (100000, -2.2e6), (200000, -2.0e6)])



# %% trusted=true
fig, ax = plt.subplots()
plot_contours(ax)

# %% trusted=true
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4*1.2,6*1.2), subplot_kw={'projection':crs})

cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

def plot_base(ax, contours=True):
    jgg_gdf.plot(ax=ax, color='#CDB380', edgecolor='none', alpha=1, zorder=1)
    gris_outline.plot(ax=ax, color='whitesmoke', edgecolor='none', alpha=1, zorder=2)
    if contours:
        plot_contours(ax)
    #surf_contours.plot(ax=ax, edgecolor='#c2c2c2', linewidth=0.7, zorder=150, alpha=0.5)
    ax.axis('off')
    
plt.subplots_adjust(wspace=0, hspace=0)

# Sums
norm_sum = colors.LogNorm(vmin=179, vmax=2.5e6)
kws_sum = dict(norm=norm_sum, cmap=cmap, rasterized=True, zorder=100, add_colorbar=False)
annual_g_sum.sel(TIME='2019').rio.clip(basins.geometry.values).plot(ax=axes[0,0], **kws_sum)
axes[0,0].set_title('2019')
label_panel(axes[0,0], 'a')

annual_g_sum.sel(TIME='2022').rio.clip(basins.geometry.values).plot(ax=axes[0,1], **kws_sum)
axes[0,1].set_title('2022')
label_panel(axes[0,1], 'b')

from matplotlib import cm
cbar_sum = fig.add_axes((0.9, 0.55, 0.04, 0.3))
cbar_sum_kws = {'label':'Total biomass (ng DW ml$^{-1}$)', 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_sum, cmap=cmap), cax=cbar_sum, **cbar_sum_kws)


# Maxes
norm_max = colors.Normalize(vmin=0, vmax=30000)
kws_max = dict(norm=norm_max, cmap=cmap, rasterized=True, zorder=100, add_colorbar=False)
annual_g_max.sel(TIME='2019').rio.clip(basins.geometry.values).plot(ax=axes[1,0], **kws_max)
axes[1,0].set_title('')
label_panel(axes[1,0], 'c')

annual_g_max.sel(TIME='2022').rio.clip(basins.geometry.values).plot(ax=axes[1,1], **kws_max)
axes[1,1].set_title('')
label_panel(axes[1,1], 'd')

cbar_max = fig.add_axes((0.9, 0.15, 0.04, 0.3))
cbar_max_kws={'label':'Max. biomass (ng DW ml$^{-1}$)', 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_max, cmap=cmap), cax=cbar_max, **cbar_max_kws)


for ax in axes.flatten():
    plot_base(ax)

plt.savefig(os.path.join(RESULTS, 'fig_map_sum_max_2019_2022.pdf'), dpi=300, bbox_inches='tight')

## To do -add scale bar

# %% [markdown]
# ### Supplementary Figure: annual bloom extent
#
# This Figure takes plotting styles directly from previous section

# %% trusted=true
# annual_g_max.plot?

# %% trusted=true
fg = annual_g_max.rio.clip(basins.geometry.values).plot(figsize=(6,9), col='TIME', col_wrap=5, subplot_kws={'projection':crs}, **kws_max)
titles = np.arange(2000, 2023, 1)
tn = 0
for ax in fg.axs.flat:
    #plot_base(ax, contours=False)
    
    ax.coastlines(color='grey', linewidth=0.5)
    ax.set_extent([-56, -31, 57, 84], crs=ccrs.PlateCarree())
    ax.axis('off')

    ax.set_title(titles[tn])
    if tn == len(titles)-1:
        break
    tn+=1
    
cbar_max = fg.fig.add_axes((0.7, 0.05, 0.03, 0.15))
cbar_max_kws={'label':'Max. biomass (ng DW ml$^{-1}$)', 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_max, cmap=cmap), cax=cbar_max, **cbar_max_kws)

plt.subplots_adjust(hspace=0.05)
# Saving is not currently working, kernel dying. Too much memory consumed by figure?
plt.savefig(os.path.join(RESULTS, 'fig_suppl_annual_bloom_max.pdf'), dpi=300, bbox_inches='tight')    


# %% [markdown]
# ### Sector-by-sector analysis
#
# Need to normalise by area. The max approach doesn't do this, because in areas like the SW with bigger blooms, the boxplots of max get 'depressed' - even though the sample size is much bigger.

# %% trusted=true

# %% trusted=true
basins

# %% trusted=true
# Calculate sums of bloom size per each elevation class in each sector of the ice sheet.
store = {}
for ix, sector in basins.iterrows():
    d = annual_g_sum.rio.clip([sector.geometry], all_touched=True, drop=True)
    sh = mar.SH.rio.clip([sector.geometry], all_touched=True, drop=True)
    sector_sum_bio_by_elev = d.groupby_bins(sh, bins=np.arange(0,2200, 400), labels=np.arange(0,2000, 400)).sum().to_pandas()
    store[sector.SUBREGION1] = sector_sum_bio_by_elev

# %% trusted=true
# GrIS context map
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
ax_gris = plt.subplot(111, projection=crs)
jgg_gdf.plot(ax=ax_gris, color='#CDB380', edgecolor='none', alpha=1, zorder=1)

gris_outline.plot(ax=ax_gris, color='whitesmoke', edgecolor='none', alpha=1, zorder=2)
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
norm = colors.LogNorm(vmin=179, vmax=2.5e6)
kws = dict(norm=norm, cmap=cmap, rasterized=True)
annual_g_sum.mean(dim='TIME').rio.clip(basins.geometry.values).plot(ax=ax_gris, zorder=10, **kws)

basins.plot(ax=ax_gris, color='None', edgecolor='Grey', linewidth=0.5, zorder=20)
ax_gris.axis('off')
for ix, basin in basins.iterrows():
    if basin.SUBREGION1 in ['SE', 'CE']:
        continue
    x, y = basin.geometry.centroid.xy
    ax_gris.text(x[0], y[0], basin.SUBREGION1, ha='center', va='center', fontsize=9, zorder=30)
plt.title('')
plt.savefig(os.path.join(RESULTS, 'fig_sectors_average_sum_map.pdf'), bbox_inches='tight')

# %% trusted=true
fig, axes = plt.subplots(figsize=(5,6), ncols=2, nrows=4)
axes = axes.flatten()

n = 1
for s in ['NO','NW','NE','CW','CE','SW','SE']:
    r = store[s]
    ax = axes[n]
    #sns.cubehelix_palette(n_colors=5, start=.7, rot=-.75, reverse=True)
    ax.stackplot(r.index, r.T, labels=r.columns, colors=sns.color_palette('flare_r', n_colors=5), edgecolor='none', linewidth=0)
    ax.set_ylim(0, 5e8)
    ax.set_title(s, y=0.9)
    ax.xaxis.set_major_locator(dates.YearLocator(5))
    n += 1
ax.legend(loc=(1,0.1), frameon=False)
sns.despine()
plt.subplots_adjust(hspace=0.5)

plt.savefig(os.path.join(RESULTS, 'fig_sectors_annual_sum.pdf'), bbox_inches='tight')

# %% [markdown]
# #### Defunct: Annual boxplots for each sector of sum or max

# %% trusted=true
store = {}
for ix, sector in basins.iterrows():
    d = annual_g_sum.rio.clip([sector.geometry], all_touched=True, drop=True)
    # Convert to a Pandas dataframe with columns (index=x,y,time), year, cum_growth
    df = d.stack(xy=('x','y')).to_pandas()
    forbox = df.T.stack().to_frame()
    forbox.columns = ['cum_growth']
    forbox['year'] = forbox.index.get_level_values(2).year
    store[sector.SUBREGION1] = forbox

# %% trusted=true
for sector in store:
    fig, ax = plt.subplots()
    sns.boxplot(ax=ax, data=store[sector], x='year', y='cum_growth', color='g')
    plt.title(sector)
    plt.ylim(0, 3e6)
    
    xticks, xlabels = plt.xticks()
    xticks = np.arange(0, 25, 5)
    xlabels = xticks + 2000
    plt.xticks(xticks, xlabels)
    plt.xlabel('')
    plt.ylabel('Sum biomass (ng DW ml$^{-1}$)')
    sns.despine()

# %% [markdown]
# ### Extent of blooms / % coverage of ice sheet by blooms and trend analysis

# %% trusted=true
#annual_biomass.where(annual_biomass > 179).mean(dim='TIME').mean().compute()

# %% trusted=true
annual_biomass = main_outputs.cum_growth.where(mar.MSK > 50).where(main_outputs.cum_growth > 179).resample(TIME='1AS').sum(dim='TIME')
as_perc = (100 / ((mar.MSK > 50).sum() * 10**2) * ((annual_biomass > 179).sum(dim=('x','y')) * 10**2)) #
as_perc.plot(label='all blooms')
big_as_perc = (100 / ((mar.MSK > 50).sum() * 10**2) * ((annual_biomass > 435364).sum(dim=('x','y')) * 10**2)) #
big_as_perc.plot(label='blooms > overall ice-sheet-wide mean')
plt.ylim(0, 25)
plt.ylabel('% of ice sheet')
sns.despine()
plt.title('')
plt.legend()
plt.savefig(os.path.join(RESULTS, 'fig_bloom_percent_cover.pdf'), bbox_inches='tight')

# %% trusted=true
annual_biomass.plot.hist()

# %% trusted=true
y = as_perc.to_pandas()
X = sm.add_constant(np.arange(0, len(y)))
m = sm.OLS(y, X)
r = m.fit()
print(r.summary())

# %% [markdown]
# ### GrIS wide min and max bloom productivity statistics

# %% trusted=true
annual_g_sum_griswide = annual_g_sum.sum(dim=('x','y'))

# %% trusted=true
annual_g_sum_griswide.idxmin()

# %% trusted=true
annual_g_sum_griswide.min()

# %% trusted=true
annual_g_sum_griswide.idxmax()

# %% trusted=true
annual_g_sum_griswide.max()

# %% [markdown]
# ### Organic carbon production potential

# %% trusted=true
# Annual total C ...
# Need to mask by last productive day?
carbon = to_carbon(annual_g_sum)
total_carbon = carbon.sum(dim=('x','y')).compute()


# %% trusted=true
total_carbon

# %% trusted=true
total_carbon.min()

# %% trusted=true
total_carbon.idxmin()

# %% trusted=true
total_carbon.max()

# %% trusted=true
total_carbon.idxmax()

# %% [markdown]
# ## UPE: Compare MAR with PROMICE

# %% trusted=true
# !curl -o UPE_U_day.nc https://dataverse.geus.dk/api/access/datafile/:persistentId?persistentId=doi:10.22008/FK2/IW73UU/3YQDQS

# %% trusted=true
upeu = xr.open_dataset('UPE_U_day.nc')

# %% trusted=true
upeu.z_boom_u.plot()

# %% trusted=true
upeu.t_u.plot()

# %% trusted=true
mar_ts = xr.open_mfdataset('/flash/tedstona/williamson/MARv3.13-ERA5/MARv3.13-10km-ERA5-*_05-09_ALGV.nc')
mar_ts = mar_ts.rename({'Y19_288':'y', 'X14_163':'x'})
mar_ts['x'] = mar.x
mar_ts['y'] = mar.y

# %% trusted=true
# %matplotlib widget
fig, ax = plt.subplots()
mar_ts.sel(TIME=slice('2009-01-01', '2022-12-31')).sel(x=pts.loc['UPE'].geometry.x, y=pts.loc['UPE'].geometry.y, method='nearest').TT.plot(ax=ax, label='MAR')
upeu.t_u.plot(ax=ax, label='AWS')

# %% [markdown]
# # Defunct analyses below here

# %% [markdown] tags=[]
# ### Plot sensitivity (early Feb '24)

# %% tags=[] trusted=true
fig, axes = plt.subplots(
    figsize=(3,5),
    nrows=2, ncols=1, 
    sharex=True
)

## Date axes
y = dates.YearLocator(5) 
axes[0].xaxis.set_major_locator(y)
axes[0].xaxis.set_major_formatter(dates.DateFormatter('%y'))
yy = dates.YearLocator(1)
axes[0].xaxis.set_minor_locator(yy)


## Starting population
ibio_quant90.plot(
    ax=axes[0],
    logy=False,
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles[::-1], labels[::-1],     
    loc=(1.1,0), 
    frameon=False, 
    title='Start pop.'
)
#axes[0].set_ylabel('Median of $P_{90}$ biomass (ng ml$^{-1}$)')
label_panel(axes[0], 'a')


## Population loss rate
ploss_quant90.plot(
    ax=axes[1],
    logy=True,
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True)
)

axes[1].legend(
    loc=(1.1, 0),
    frameon=False,
    title='Loss rate'
)
#axes[1].set_ylabel('Median of $P_{90}$ biomass (ng ml$^{-1}$)')
axes[1].set_xlabel('')
label_panel(axes[1], 'b')


## Final common commands
sns.despine(fig=fig)
fig.text(-0.1, 0.3, 'Median of $P_{90}$ biomass (ng DW ml$^{-1}$)', rotation='vertical')
plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis.pdf'), bbox_inches='tight')

# %% [markdown]
# ---

# %% [markdown]
# ### Further analysis of sensitivity to starting biomass

# %% tags=[] trusted=true
# tmp = pd.DataFrame(store)
# tmp
# tmp.plot(logy=True)

# sens_ibio_test = xr.open_mfdataset(f'/flash/tedstona/williamson/outputs/sensitivity_ibio/*_ibio179.nc')
# sens_ibio_test['x'] = mar.x
# sens_ibio_test['y'] = mar.y

# sens_ibio_test

# sens_ibio_test.cum_growth.where(mar.MSK > 50).where(sens_ibio_test.cum_growth > 179).resample(TIME='1AS').sum().plot(col='TIME', col_wrap=4)

# %% trusted=true
