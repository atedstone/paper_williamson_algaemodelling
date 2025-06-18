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
#       jupytext_version: 1.17.1
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
from copy import deepcopy

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
rcParams['font.size'] = 7

# %% trusted=true
from dask.distributed import LocalCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
client = Client(cluster)

# %% trusted=true
cluster.scale(n=6)

# %% trusted=true
client

# %% [markdown]
# ## Paths/Settings

# %% trusted=true
# OPTIONS - pick your system
# unil mac
# GIS_ROOT = '/Users/atedston/Dropbox/work/gis/'
# INPUTS_PATH = None
# WORK_ROOT = '/scratch/williamson/'

# octo
GIS_ROOT = '/work/atedstone/gis/'
INPUTS_PATH = '/work/atedstone/williamson/MARv3.13-ERA5/*.nc'
WORK_ROOT = '/work/atedstone/williamson/'

# %% trusted=true
## CREATE PATHS

# A 'reference' MAR output, mainly for the mask and georeferencing
MAR_REF = os.path.join(WORK_ROOT, 'MARv3.14.0-10km-daily-ERA5-2022.nc')
# DEM file specific to this project
DEM_FILE = os.path.join(WORK_ROOT, 'gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif')

# Generic GIS requirements
BASINS_FILE = os.path.join(GIS_ROOT, 'doi_10.7280_D1WT11__v1/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions.shp')
OUTLINE_FILE = os.path.join(GIS_ROOT, 'gris_only_outline/greenland_icesheet_fix.shp')
LAND_AREAS_FILE = os.path.join(GIS_ROOT, 'ne_10m_land/ne_10m_land.shp')
GRIS_BBOX_FILE = os.path.join(GIS_ROOT, 'greenland_area_bbox/greenland_area_bbox.shp')
SURF_CONTOURS_FILE = os.path.join(GIS_ROOT, 'gimp_contours/gimpdem_90m_v01.1_EPSG3413_grisonly_contours_2km_i500/contour.shp')

# -----------------------------------------------------------------------------------
# Algal growth model sensitivity runs (accessed by wildcard in script)
MODEL_OUTPUTS_SENS_IBIO = os.path.join(WORK_ROOT, 'outputs/sensitivity_ibio')
MODEL_OUTPUTS_SENS_PLOS = os.path.join(WORK_ROOT, 'outputs/sensitivity_ploss')
MODEL_OUTPUTS_SENS_TEMP = os.path.join(WORK_ROOT, 'outputs/sensitivity_temp')
MODEL_OUTPUTS_SENS_LIGH = os.path.join(WORK_ROOT, 'outputs/sensitivity_light')
MODEL_OUTPUTS_SENS_SNOW = os.path.join(WORK_ROOT, 'outputs/sensitivity_snowdepth')

# Main run of algal growth model
MODEL_OUTPUTS_MAIN = os.path.join(WORK_ROOT, '2025-05')
NQMC = 512

# File containing experiment design
EXPTS_DESIGN_FN = os.path.join(MODEL_OUTPUTS_MAIN, 'expt_parameters_ibio179.csv')

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

def get_qmc_ts(year, geom=None, qm_metrics=False, nqmc=NQMC):
    """ Return time series from each QMC run. 

    By default, returns a daily median pd.DataFrame computed from all pixels in the ice sheet domain, one column per QMC run.

    If geom is a geometry (Point), then subset to the nearest matching pixel instead, one column per QMC run.

    If qm_metrics=True then provide additional QMC metrics relating to MC performance.
    """
    # Time-series results
    qmc = {}
    
    if qm_metrics:
        # Bloom max of each experiment
        bmax = []
        # Running standard deviation of bmax
        sd = []
        # Running standard deviation of sd
        sd_sd = []

    for exp in range(1, nqmc+1):
        #print(exp)
        # Open the numbered experiment
        r = open_model_run(os.path.join(WORK_ROOT, '2025-05', f'model_outputs_{year}_exp{exp}.nc'))
        # If subset to Point requested then do this now
        if geom is not None:
            r = r.sel(x=geom.x, y=geom.y, method='nearest')
        # Identify pixels to include according to whether they saw any growth in the season
        incl = r.cum_growth.where(r.cum_growth > START_POP).count(dim='TIME')
        # Reduce to 1-D timeseries
        ts = r.cum_growth.where(mar.MSK > 50).where(incl > 1).median(dim=('x','y')).to_pandas() #.where(r.cum_growth > START_POP)
        if qm_metrics:
            # Append the metrics
            bmax.append(ts.max())
            sd.append(np.std(bmax))
            sd_sd.append(np.std(sd))
        # Save the time series
        qmc[exp] = ts
        qmc[exp].name = f'exp{exp}'

    t = pd.concat(qmc, axis=1)
    if qm_metrics:
        return (t, 
                bmax, 
                sd, 
                sd_sd)
    else:
        return t

def doy_to_datetime(year, doy):
    return dt.datetime(year, 1, 1) + dt.timedelta(doy - 1)
    
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
UNIT_DW = r'ng DW ml$^{-1}$'
UNIT_WW = r'cells ml$^{-1}$'

def label_panel(ax, letter, xy=(0.04,0.93)):
    ax.annotate(letter.upper(), fontweight='bold', xy=xy, xycoords='axes fraction',
           horizontalalignment='left', verticalalignment='top', fontsize=8)

import string
def letters(start=0, letters=list(string.ascii_lowercase)):
    """ Return letters of alphabet, in order, 1 by 1.
    
    :param start: integer position in letters list at which to begin.
    
    Usage:
    gen = letters()
    next(gen)
    Out: 'a'
    
    gen = letters(start=2)
    next(gen)
    Out: 'c'
    """
    ii = 0
    for i in letters:
        if ii < start:
            ii += 1
            continue
        yield(i) 


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
# ---
# ## Sensitivity analysis

# %% [markdown]
# ### Fig. 1: Phenological parameters, S6

# %% trusted=true
year = 2016
regenerate = False

fn_ibio = os.path.join(RESULTS, 'sens_analysis_s6_ibio.csv')
fn_ploss = os.path.join(RESULTS, 'sens_analysis_s6_ploss.csv')

if regenerate:
    print('Regenerating....')
    ibio = [17.900000000000002, 179, 895, 1790]
    site_store = {}
    for ib in ibio:
        sens_ibio = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_IBIO, f'model_outputs_{year}_ploss0.1_ibio{ib}.nc'))          
        v = sens_ibio.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[ib] = v
    ibio_site = pd.DataFrame(site_store)
    ibio_site.columns = [int(np.round(float(c))) for c in ibio_site.columns]
    ibio_site.to_csv(fn_ibio)

    ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
    site_store = {}
    for pl in ploss:
        sens_ploss = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_PLOS, f'model_outputs_{year}*_ploss{pl}.nc'))
        v = sens_ploss.cum_growth.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest').to_pandas()
        site_store[pl] = v  
    ploss_site = pd.DataFrame(site_store)
    ploss_site.columns = [str(int(float(c)*100))+'%' for c in ploss_site.columns]
    ploss_site.to_csv(fn_ploss)
else:
    print('INFO...using cached results (not regenerated)')
    
ibio_site = pd.read_csv(fn_ibio, index_col=0, parse_dates=True)
ploss_site = pd.read_csv(fn_ploss, index_col=0, parse_dates=True)




# ADD BLOOM DURATION with shading


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.5, 2))

# Starting population
ibio_site.plot(
    ax=axes[0],
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True),
    logy=True
)
#m = dates.MonthLocator() 
#axes[0].xaxis.set_major_locator(m)
#axes[0].xaxis.set_major_formatter(dates.DateFormatter('%b'))

#axes[0,0].set_title('S6')
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles[::-1], labels[::-1],     
    loc=(1,0.4), 
    frameon=False, 
    title='Start pop.',
    handlelength=1
)
label_panel(axes[0], 'a')
axes[0].set_ylabel('Pop. size (ng DW ml$^{-1}$)')

# Ploss
ploss_site.plot(
    ax=axes[1],
    legend=False,
    colormap=sns.color_palette('crest', as_cmap=True),
    logy=True
)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(
    handles, labels,     
    loc=(1,0.1), 
    frameon=False, 
    title='Loss %',
    handlelength=1
)
label_panel(axes[1], 'b')

for ax in axes.flatten():
    ax.set_ylim(0, 120000)
    ax.set_xlabel('')

plt.subplots_adjust(wspace=0.7)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig_phenological.pdf'), bbox_inches='tight')

# %% trusted=true
ibio_site.index

# %% [markdown]
# ### Fig. 2: Sensitivity to environmental parameters

# %% trusted=true
year = 2016
regenerate = False

fn_sd = os.path.join(RESULTS, 'sens_analysis_s6_snowdepth.csv')
fn_li = os.path.join(RESULTS, 'sens_analysis_s6_swd.csv')
fn_t = os.path.join(RESULTS, 'sens_analysis_s6_temperature.csv')


if regenerate:

    # Snow depth
    depths = [0.01, 0.05, 0.10, 0.2, 0.4, 0.8]
    site_store = {}
    last_active_bloom = {}
    for d in depths:
        sens_sd = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_SNOW, f'model_outputs_{year}_ibio179_ploss0.1_snow{d}.nc'))
        v = sens_sd.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest')
        site_store[d] = v.cum_growth.to_pandas()
        v['TIME'] = v['TIME.dayofyear']
        last_active_bloom[d] = v.today_prod.cumsum(dim='TIME').idxmax(dim='TIME').values
    sd_site = pd.DataFrame(site_store)
    sd_site.to_csv(fn_sd)
    pd.Series(last_active_bloom).to_csv(fn_sd[:-4] + '_lastbloom.csv')

    # SWd
    lights = [1, 10, 100, 200]
    site_store = {}
    last_active_bloom = {}
    for li in lights:
        sens_li = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_LIGH, f'model_outputs_{year}_ibio179_ploss0.1_light{li}.nc'))
        v = sens_li.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest')
        site_store[li] = v.cum_growth.to_pandas()
        v['TIME'] = v['TIME.dayofyear']
        last_active_bloom[li] = v.today_prod.cumsum(dim='TIME').idxmax(dim='TIME').values
    li_site = pd.DataFrame(site_store)
    li_site.to_csv(fn_li)
    pd.Series(last_active_bloom).to_csv(fn_li[:-4] + '_lastbloom.csv')

    # Temperature
    temps = [0, 0.25, 0.5, 1.0]
    site_store = {}
    last_active_bloom = {}
    for t in temps:
        sens_t = open_model_run(os.path.join(MODEL_OUTPUTS_SENS_TEMP, f'model_outputs_{year}_ibio179_ploss0.1_temp{t}.nc'))
        v = sens_t.sel(x=pts_ps.loc['S6'].geometry.x, y=pts_ps.loc['S6'].geometry.y, method='nearest')
        site_store[t] = v.cum_growth.to_pandas()
        v['TIME'] = v['TIME.dayofyear']
        last_active_bloom[t] = v.today_prod.cumsum(dim='TIME').idxmax(dim='TIME').values
    t_site = pd.DataFrame(site_store)
    t_site.to_csv(fn_t)
    pd.Series(last_active_bloom).to_csv(fn_t[:-4] + '_lastbloom.csv')

sd_site = pd.read_csv(fn_sd, index_col=0, parse_dates=True)
li_site = pd.read_csv(fn_li, index_col=0, parse_dates=True)
t_site = pd.read_csv(fn_t, index_col=0, parse_dates=True)

sd_site_end = pd.read_csv(fn_sd[:-4] + '_lastbloom.csv', index_col=0, parse_dates=True).squeeze()
sd_site_end.index = sd_site_end.index.astype(str)
li_site_end = pd.read_csv(fn_li[:-4] + '_lastbloom.csv', index_col=0, parse_dates=True).squeeze()
li_site_end.index = li_site_end.index.astype(str)
t_site_end = pd.read_csv(fn_t[:-4] + '_lastbloom.csv', index_col=0, parse_dates=True).squeeze()
t_site_end.index = t_site_end.index.astype(str)


# %% trusted=true
def bloom_start(ts, startpop=179, n=3):
    """
    Determine bloom start day of year.

    ts : time series (pd.DataFrame, may be multiple columns)
    startpop : starting/background population ng DW ml-1
    n : number of days of continuity to require when identifying start date.

    returns pd.Series of dates
    """

    def _roll(ts, n=n):
        return np.minimum(
            np.maximum(0, 
                       (ts.sum(axis=0) - n) + 1
                      ),
            1
        )

    # Work on copy of dataframe
    ts = ts.copy()
    # Convert series of pops to boolean
    ts = (ts > startpop)
    # Apply rolling while index is still datetime64
    r = ts.rolling(f'{n}D').apply(_roll)
    # Convert to DOY
    r.index = r.index.dayofyear
    return r.idxmax()


t_start = bloom_start(t_site)
sd_start = bloom_start(sd_site)
li_start = bloom_start(li_site)


t_dur = t_site_end - t_start
sd_dur = sd_site_end - sd_start
li_dur = li_site_end - li_start


# %% trusted=true
t_site_end

# %% trusted=true
from copy import deepcopy

def active_bloom_sum(df, start_dates, end_dates):
    vals = {}
    for c in df.columns:
        tmp = deepcopy(df[c])
        tmp.index = tmp.index.dayofyear
        clipped = tmp[(tmp.index >= start_dates[c] ) & (tmp.index <= end_dates[c])]
        vals[c] = clipped.sum()
    return pd.Series(vals)

t_sum = active_bloom_sum(t_site, t_start, t_site_end)
sd_sum = active_bloom_sum(sd_site, sd_start, sd_site_end)
li_sum = active_bloom_sum(li_site, li_start, li_site_end)

# %% trusted=true
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(7.5, 2))

labels = ['T', 'Sn', 'SWD']


# Ax Bloom onset 
axes[0].plot([1]*len(t_site.columns), t_start, 'o')
axes[0].plot([2]*len(sd_site.columns), sd_start, 'o')
axes[0].plot([3]*len(li_site.columns), li_start, 'o')
axes[0].xaxis.set_ticks([1,2,3], labels)
axes[0].set_ylabel('Bloom start (DOY)')

# Ax Bloom duration
axes[1].plot([1]*len(t_site.columns), t_dur, 'o')
axes[1].plot([2]*len(sd_site.columns), sd_dur, 'o')
axes[1].plot([3]*len(li_site.columns), li_dur, 'o')
axes[1].xaxis.set_ticks([1,2,3], labels)
axes[1].set_ylabel('Bloom duration (days)')

# Ax Max bloom size
axes[2].plot([1]*len(t_site.columns), t_site.max(), 'o')
axes[2].plot([2]*len(sd_site.columns), sd_site.max(), 'o')
axes[2].plot([3]*len(li_site.columns), li_site.max(), 'o')
axes[2].xaxis.set_ticks([1,2,3], labels)
axes[2].set_ylabel('Max. pop size (ng DW ml-1)')

# Sum of bloom during active period
# Can't use simple sum, needs to be sum of active bloom period.
axes[3].plot([1]*len(t_sum), t_sum, 'o')
axes[3].plot([2]*len(sd_sum), sd_sum, 'o')
axes[3].plot([3]*len(li_sum), li_sum, 'o')
axes[3].xaxis.set_ticks([1,2,3], labels)
axes[3].set_ylabel('Tot. pop size (ng DW ml-1)')

sns.despine()
plt.tight_layout()


# %% trusted=true
t_site.plot()

# %% [markdown]
# ## Load QMC experiment info

# %% trusted=true
qmc_expts = pd.read_csv(EXPTS_DESIGN_FN, index_col=0)

# %% trusted=true
qmc_expts.head()

# %% [markdown]
# ---
# ## Fig. 3: QMC analysis at Point sites

# %% trusted=true scrolled=true
regenerate = True
years = (2022, 2022)

# Filename formatting according to time range
if years[0] == years[1]:
    t = years[0]
else:
    t = '{0}_{1}'.format(years[0], years[1])

# Only recompute from netcdfs if explicitly requested
if regenerate:
    for ix, row in pts_ps.iterrows():
        print(ix)
        store = []
        for year in range(years[0], years[1]+1):
            print(year)
            ensemble = get_qmc_ts(year, geom=row.geometry)
            store.append(ensemble)
        
        pd.concat(store, axis=0).to_csv(os.path.join(RESULTS,f'qmc_gris_median_ts_{ix}_{t}.csv'))

# %% trusted=true
sites

# %% trusted=true scrolled=true
# Now reload same reudced QMC outputs from disk
sites_cold = {}
sites_warm = {}
for ix, row in pts_ps.iterrows():
    sites_cold[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_gris_median_ts_{ix}_2022.csv'), index_col=0, parse_dates=True)
    sites_warm[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_gris_median_ts_{ix}_2019.csv'), index_col=0, parse_dates=True)


# %% trusted=true
## Calculate total annual growth at each site

# First find end of active growth 
# This requires gross daily production, so have to revisit the netcdfs. 
# This is expensive so we want to do this as few times as possible.

def find_last_bloom_day_site(year, site_runs, geom, quantiles=[0.25, 0.5, 0.75]):
    """
    Returns:
    (tuple of experimment IDs, tuple of LDOY for each requested quantile)
    """
    ranked = site_runs.max().sort_values()
    expts = []
    values = []
    for q in quantiles:
        expt_id = ranked.index[np.abs(ranked - ranked.quantile(q)).argmin()]
        fn = os.path.join(WORK_ROOT, '2025-05', f'model_outputs_{year}_exp{expt_id}.nc')
        run = open_model_run(fn)
        run = run.sel(x=geom.x, y=geom.y, method='nearest')
        doy = run.TIME.dt.dayofyear
        run['TIME'] = doy
        lpd = run.today_prod.cumsum(dim='TIME').idxmax(dim='TIME')
        lpd.coords['TIME'] = pd.Timestamp(year, 1, 1)
        lpd = lpd.expand_dims({'TIME':[pd.Timestamp(year, 1, 1)]}).compute()
        
        expts.append(expt_id)
        values.append(lpd.values[0])
        
    return (expts, values)

# Retrieve the last day of year of each ensemble member that we want to plot
# - and their experiment ID.
expts_warm = {}
expts_cold = {}
ldoy_warm = {}
ldoy_cold = {}
for ix, row in pts_ps.iterrows():
    expts_warm[ix], ldoy_warm[ix] = find_last_bloom_day_site(2019, sites_warm[ix], row.geometry)
    expts_cold[ix], ldoy_cold[ix] = find_last_bloom_day_site(2022, sites_cold[ix], row.geometry)
    

# Now calculate total biomass at each quantile during the active growth period.
store = []
# This is a lookup table of how each tuple index links to the quantile.
q_lut = {0:0.25, 1:0.50, 2:0.75}

# Iterate site-by-site, adding two rows per site: one for the warm year, one for the cold year
for ix, row in pts_ps.iterrows():
    
    med = sites_cold[ix][sites_cold[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][1]), expts_cold[ix][1]].sum()
    q75 = sites_cold[ix][sites_cold[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][2]), expts_cold[ix][2]].sum() 
    q25 = sites_cold[ix][sites_cold[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][0]), expts_cold[ix][0]].sum()
    iqr = q75 - q25
    store.append(dict(
        yr='2022',
        biomass=med,
        iqr=iqr,
        q75=q75,
        q25=q25,
        site=ix
    ))

    med = sites_warm[ix][sites_warm[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][1]), expts_warm[ix][1]].sum()
    q75 = sites_warm[ix][sites_warm[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][2]), expts_warm[ix][2]].sum() 
    q25 = sites_warm[ix][sites_warm[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][0]), expts_warm[ix][0]].sum()
    iqr = q75 - q25
    store.append(dict(
        yr='2019',
        biomass=med,
        iqr=iqr,
        q75=q75,
        q25=q25,
        site=ix
    ))

df = pd.DataFrame(store)    

# %% trusted=true
letgen = letters()

fig = plt.figure(figsize=(7.5, 4))
# nrows, ncols
gs = fig.add_gridspec(2,3)

#fig, axes = plt.subplots(ncols=3, nrows=2)
col_warm = sns.set_hls_values('tab:orange', l=0.6)
col_cold = sns.set_hls_values('tab:blue', l=0.6)
def _plot_site(ax, site_name):
    
    sites_warm[site_name].plot(legend=False, color=col_warm, alpha=0.05, ax=ax, linewidth=0.5)
    sites_warm[site_name].quantile(0.5, axis=1).plot(color='tab:orange', linewidth=2, ax=ax)    
    sites_warm[site_name].quantile(0.25, axis=1).plot(color='tab:orange', ax=ax)    
    sites_warm[site_name].quantile(0.75, axis=1).plot(color='tab:orange', ax=ax)    

    # Take copy of cold site purely so that we can adjust time to match the warm year,
    # so that we can plot them on same x axis
    # n.b. this is a bit slow, would probably be better to reindex a different way...
    ts = deepcopy(sites_cold[site_name])
    ts.index = ts.index - dt.timedelta(days=365*3)
    ts.plot(legend=False, color=col_cold, alpha=0.05, ax=ax, linewidth=0.5)
    ts.quantile(0.5, axis=1).plot(color='tab:blue', linewidth=2, ax=ax)    
    ts.quantile(0.25, axis=1).plot(color='tab:blue', ax=ax)    
    ts.quantile(0.75, axis=1).plot(color='tab:blue', ax=ax)    
    
    ax.set_title(site_name)
    ax.set_ylim(0, 35000)
    m = dates.MonthLocator() 
    ax.xaxis.set_major_locator(m)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    ax.set_xlabel('')
    ax.set_ylabel(f'Pop. size ({UNIT_DW})')
    label_panel(ax, next(letgen))
    
    
ax1 = fig.add_subplot(gs[0, 0])
_plot_site(ax1, 'UPE')
ax2 = fig.add_subplot(gs[1, 0])
_plot_site(ax2, 'S6')
ax3 = fig.add_subplot(gs[0, 1])
_plot_site(ax3, 'Mittivak')
ax4 = fig.add_subplot(gs[1, 1])
_plot_site(ax4, 'South')


# Now plot total biomass by site for cold and warm years as a bar plot
# (if we used full ensemble here then we could do this as a boxplot, but would need to calculate the active bloom duration for all experiments in the ensemble)
ax5 = fig.add_subplot(gs[:, 2])

df = df.rename({'yr':'Year'}, axis=1)
g = sns.barplot(
    data=df,
    x="site", y="biomass", hue="Year",
    ax=ax5
)

# The seaborn errorbar kwarg doesn't allow to pass our desired errorbar values, so we need to drop
# down to the matplotlib interface to add these manually...:
# (we can't use the native methods unless we provide data of the full ensemble for calculation, which 
# is precisely what we need to avoid given how expensive the I/O on this is)

# Add errorbars (IQR) for cold year
xpos = 0
for ix, row in df[df.Year == '2022'].iterrows():
    ax5.plot([xpos-0.2,xpos-0.2], [row.q25, row.q75], color='k')
    xpos += 1

# Add errorbars (IQR) for warm year
xpos = 0
for ix, row in df[df.Year == '2019'].iterrows():
    ax5.plot([xpos+0.2,xpos+0.2], [row.q25, row.q75], color='k')
    xpos += 1

label_panel(ax5, next(letgen))
ax5.legend(loc='upper right')
ax5.set_ylabel(f'Total biomass ({UNIT_DW})')

sns.despine()
plt.tight_layout()

plt.savefig(os.path.join(RESULTS, 'fig3_site_ensembles.pdf'))


# %% [markdown]
# ## Fig. 4: Measured-modelled comparison

# %% trusted=true
data = pd.read_csv(os.path.join(WORK_ROOT, 'glacier_algal_biomass_datasets', 'all_data.csv'), parse_dates=['date'], dayfirst=False)

# Apply a year column so that we can easily iterate through the QMC runs by year
data['year'] = data.date.dt.year

# Convert the geometry column to proper Geometry (n.b this is basically redundant with respect to `site` column)
data = gpd.GeoDataFrame(data, geometry=gpd.GeoSeries.from_wkt(data['geom'], crs=3413))
data = data.drop(columns=['geom'])

# The provided CSV often contains lots of observed values on a single day at a given site.
# There is no need to repeatedly query for the same modelled value at this given site, so drop these duplicates away.
data = data.groupby(['study', 'site', 'date']).first()

# And, to avoid confusion, remove the columns that we are not using here at the moment.
data = data.drop(columns=['observed.cells.ml', 'modelled.biomass.dw', 'observed.cells.ml.dw'])

# Reset the index after the groupby operation produced a MultiIndex.
data = data.reset_index()

# %% trusted=true
data

# %% trusted=true
outputs = []
for year in data['year'].unique():
    print(year)
    meas_subset = data[data.year == year]
    
    for exp in range(1, NQMC+1):
        # Open the numbered experiment
        r = open_model_run(os.path.join(WORK_ROOT, '2025-05', f'model_outputs_{year}_exp{exp}.nc'))
        for ix, row in meas_subset.iterrows():
            rr = r.cum_growth.sel(x=row.geometry.x, y=row.geometry.y, method='nearest').sel(TIME=row.date)
            outputs.append({'date':row.date, 'study':row.study, 'site':row.site, 'exptid':exp, 'biomass_dw':float(rr)})
        

# %% trusted=true scrolled=true
pts_modelled = pd.DataFrame(outputs)

# %% trusted=true
# Now need to reduce the DataFrame so that we have only one row for each site-date combination
# which can then be merged back onto the df with the measured values.

q25 = pts_modelled.groupby(['study', 'date', 'site']).biomass_dw.quantile(0.25)
q25.name = 'modelled.biomass.dw.q25'
q50 = pts_modelled.groupby(['study', 'date', 'site']).biomass_dw.quantile(0.50)
q50.name = 'modelled.biomass.dw.q50'
q75 = pts_modelled.groupby(['study', 'date', 'site']).biomass_dw.quantile(0.75)
q75.name = 'modelled.biomass.dw.q75'
modelled_merge = pd.concat([q25, q50, q75], axis=1)

meas = pd.read_csv(os.path.join(WORK_ROOT, 'glacier_algal_biomass_datasets', 'all_data.csv'), parse_dates=['date'], dayfirst=False, na_values=['NA','#VALUE!'])
meas = meas.drop(columns=['modelled.biomass.dw', 'geom'])
#meas['observed.cells.ml.dw'] = pd.to_numeric(meas['observed.cells.ml.dw'])

# %% trusted=true
m = pd.merge(left=meas, right=modelled_merge, left_on=['study', 'date', 'site'], right_on=['study', 'date', 'site'], how='left')
m

# %% trusted=true
# Save to CSV files
pts_modelled.to_csv(os.path.join(RESULTS, 'qmc_modelled_at_observed_points_all_expts.csv'))
m.to_csv(os.path.join(RESULTS, 'qmc_modelled_quantiles_vs_observed.csv'))


# %% trusted=true
mm = m.groupby(['study', 'site', 'date']).mean()
sns.scatterplot(data=mm, y='observed.cells.ml.dw', x='modelled.biomass.dw.q50')

# %% trusted=true
sns.scatterplot(data=m, x='observed.cells.ml.dw', y='modelled.biomass.dw.q50')

# %% trusted=true
m['modelled.biomass.dw.q50']

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

# %% trusted=true
for year in range(2000, 2023):
    #plt.hist(qmc[year].max(axis=0))
    sns.kdeplot(qmc[year].max(axis=0), label=year, fill=True, color='tab:blue', alpha=0.1)

# %% trusted=true
for year in range(2000, 2023):
    #plt.hist(qmc[year].max(axis=0))
    sns.kdeplot(qmc[year].sum(axis=0), label=year, fill=True, color='tab:blue', alpha=0.1)

# %% trusted=true
sns.kdeplot(bmax_gris, fill=True, label='2012')
sns.kdeplot(bmax00, fill=True, label='2000')
plt.legend()

# %% trusted=true scrolled=true
qmc12_s6, bmax12_s6, sd12_s6, sd_sd12_s6 = analyse_qmc(2012, geom=pts_ps.loc['S6'].geometry, metrics=True)

# %% trusted=true scrolled=true
qmc12_s6.plot(legend=False, alpha=0.1, color='tab:blue')

# %% trusted=true

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
# ## Map small versus large bloom years

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
# ## Sector-by-sector analysis
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
# ## Extent of blooms / % coverage of ice sheet by blooms and trend analysis

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
# ## GrIS wide min and max bloom productivity statistics

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
# ## Organic carbon production potential

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

# %% trusted=true

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
mar_alg_inputs = xr.open_mfdataset(INPUTS_PATH, chunks={'TIME':365})
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

# %% [markdown]
# ### Sensitivity to starting biomass term (prior to May 2025)

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
# ### Sensitivity to loss term (prior to May 2025)

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
sns.despine()label_panel(ax[0], 'a')

plt.savefig(os.path.join(RESULTS, 'fig_sens_analysis_ploss.pdf'), bbox_inches='tight')

# %% trusted=true

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

# %% trusted=true
