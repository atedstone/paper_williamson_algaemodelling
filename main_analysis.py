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
from matplotlib import cm
import matplotlib.image as mpimg
import string
import cartopy.crs as ccrs
import seaborn as sns

sns.set_context('paper')
rcParams['font.family'] = 'Arial'
SMALL_SIZE = 8
MEDIUM_SIZE = 8
BIGGER_SIZE = 8
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# %% trusted=true scrolled=true
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
DEM_FILE = os.path.join(GIS_ROOT, 'GIMPDEM', 'gimpdem_90m_v01.1_EPSG3413_grisonly_filled_2km.tif')

# Generic GIS requirements
BASINS_FILE = os.path.join(GIS_ROOT, 'doi_10.7280_D1WT11__v1/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions.shp')
BASINS_LINES_FILE = os.path.join(GIS_ROOT, 'doi_10.7280_D1WT11__v1/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions_polyline_boundaries.gpkg.geojson')
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
MODEL_OUTPUTS_MAIN = os.path.join(WORK_ROOT, 'outputs', 'QMC')
NQMC = 512

# File containing experiment design
EXPTS_DESIGN_FN = os.path.join(MODEL_OUTPUTS_MAIN, 'expt_parameters_ibio179.csv')

# Save location for figures, CSV files
RESULTS = os.path.join(WORK_ROOT, 'results')

CRS = 'epsg:3413'

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

## SENSITIVITIES ANALYSIS 
def sensitivities_extract(fn_input, fn_output, values, geom, metric):
    """
    fn_input : path to netcdf files, with a {v} position left vacant
    fn_output : path to output csv, with {param} vacant
    values : List of parameter values to iterate
    geom : Point to extract

    Extracts time series of
    - daily population size
    - Net daily growth
    - last bloom day of year
    Saves all to CSV.
    """
    
    def _to_df(data_dict, fn, metric):   
        df = pd.DataFrame(data_dict)
        # Format the columns nicely
        if metric == 'ibio':
            df.columns = [int(np.round(float(c))) for c in df.columns]
        elif metric == 'ploss':
            df.columns = [str(int(float(c)*100))+'%' for c in df.columns]
        df.to_csv(fn)
        return df
        
    site_store_pop = {}
    site_store_Gn = {}
    last_bloom_doy = {}
    for p in values:
        sens = open_model_run(fn_input.format(v=p))          
        v = sens.sel(x=geom.x, y=geom.y, method='nearest')
        site_store_pop[p] = v.cum_growth.to_pandas()
        site_store_Gn[p] = v.today_prod_net.to_pandas()
        v['TIME'] = v['TIME.dayofyear']
        last_bloom_doy[p] = v.today_prod.cumsum(dim='TIME').idxmax(dim='TIME').values
    _ = _to_df(site_store_pop, fn_output.format(param='popsize'), metric)
    _ = _to_df(site_store_Gn, fn_output.format(param='Gn'), metric)
    pd.Series(last_bloom_doy).to_csv(fn_output.format(param='lastbloom'))
    return 

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

def active_bloom_sum(df, start_dates, end_dates):
    """ Sum the active bloom biomass, i.e. that between the start and end dates.

    Where df is 1...n columns of model outputs.
    """
    vals = {}
    for c in df.columns:
        tmp = deepcopy(df[c])
        tmp.index = tmp.index.dayofyear
        clipped = tmp[(tmp.index >= start_dates[c] ) & (tmp.index <= end_dates[c])]
        vals[c] = clipped.sum()
    return pd.Series(vals)

## ------------------------------------------------------------------------
## QMC ANALYSIS
def get_qmc_ts(year, geom=None, qm_metrics=False, nqmc=NQMC):
    """ Return time series of daily population size and daily net growth from each QMC run. 

    By default, returns a daily median pd.DataFrame computed from all pixels in the ice sheet domain, one column per QMC run.

    If geom is a geometry (Point), then subset to the nearest matching pixel instead, one column per QMC run.

    If qm_metrics=True then provide additional QMC metrics relating to MC performance.
    """
    # Time-series results
    pop = {}
    Gn = {}
    
    if qm_metrics:
        # Bloom max of each experiment
        bmax = []
        # Running standard deviation of bmax
        sd = []
        # Running standard deviation of sd
        sd_sd = []

    for exp in range(1, nqmc+1):
        print(exp)
        # Open the numbered experiment
        r = open_model_run(os.path.join(MODEL_OUTPUTS_MAIN, f'model_outputs_{year}_exp{exp}.nc'))
        # If subset to Point requested then do this now
        if geom is not None:
            r = r.sel(x=geom.x, y=geom.y, method='nearest')
        # Identify pixels to include according to whether they saw any growth in the season
        incl = r.cum_growth.where(r.cum_growth > START_POP).count(dim='TIME')
        # Reduce to 1-D timeseries
        ts = r.cum_growth.where(mar.MSK > 50).where(incl > 1).median(dim=('x','y')).to_pandas() #.where(r.cum_growth > START_POP)
        ts_net_daily_growth = r.today_prod_net.where(mar.MSK > 50).where(incl > 1).median(dim=('x','y')).to_pandas()
        if qm_metrics:
            # Append the metrics
            bmax.append(ts.max())
            sd.append(np.std(bmax))
            sd_sd.append(np.std(sd))
        # Save the time series
        pop[exp] = ts
        pop[exp].name = f'exp{exp}'
        Gn[exp] = ts_net_daily_growth
        Gn[exp].name = f'exp{exp}'

    pop = pd.concat(pop, axis=1)
    Gn = pd.concat(Gn, axis=1)
    if qm_metrics:
        return (pop, Gn,
                bmax, 
                sd, 
                sd_sd)
    else:
        return (pop, Gn)

## -----------------------------------------------------------
## GENERIC FUNCTIONS
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

def to_kgDWgrid(x, grid_km=10):
    """ Convert from ng DW ml to ng DW of algae in each grid cell. """
    #ng DW mL to ng DW L
    x = x * 1000
 
    #ng DW L to ng DW m2 (using Williamson et al 2018 measured conversion factor)
    x = x * 1.061
 
    #ng DW m2 to ng DW per km2
    x = x * 10**6
 
    #ng DW km2 to kg DW km2
    x = x * 10**-12 #10 to the power of minus 12
 
    #kg DW km2 to kg DW per pixel
    x = x * grid_km**2

    return x

def to_carbon(x, quotient, grid_km=10):
    """ 
    x : DataArray. Provide in units of cells per ml
    quotient : per cell carbon quotient, pg C per cell. e.g. 106 or 420.
    grid_km : size of grid cell in kilometres
     
    returns: kg carbon per model cell
    """

    #convert from cells per ml to pg C ml assuming provided quotient in pg C per cell 
    x = x * quotient

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

LABEL_P = '$P$ (%s)' %UNIT_DW
LABEL_PMAX = '$P_{MAX}$ (%s)' %UNIT_DW
LABEL_TOT_GN_PX = r'$\Sigma G_N$ (kg DW in 10$^2$ km)'
LABEL_TOT_GN_T = r'$\Sigma G_N$ (t DW)'

def label_panel(ax, letter, xy=(0.04,0.93)):
    ax.annotate(letter.upper(), fontweight='bold', xy=xy, xycoords='axes fraction',
           horizontalalignment='left', verticalalignment='top', fontsize=8)

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
pts_wgs84.geometry

# %% trusted=true
# Sanity-check the points
fig, ax = plt.subplots()
mar.MSK.plot(ax=ax)
pts_ps.plot(ax=ax, marker='x', color='r')

# %% trusted=true
# Get MAR elevations of each point
for ix, row in pts_ps.iterrows():
    v = mar.SH.sel(x=row.geometry.x, y=row.geometry.y, method='nearest').compute()
    #np.abs(v.x - row.geometry.x).to_pandas(), np.abs(v.y - row.geometry.y).to_pandas(),
    print(ix, v.to_pandas().astype(int))

# %% trusted=true
basins = gpd.read_file(BASINS_FILE)
basins.index = basins.SUBREGION1
basins.head()

# %% trusted=true
basin_boundaries = gpd.read_file(BASINS_LINES_FILE, layer='Greenland_Basins_PS_v1_4_2_regions_polyline_boundaries')
basin_boundaries.head()

# %% [markdown]
# ---
# ## Sensitivity analysis

# %% [markdown]
# ### Fig. 1: Phenological parameters, S6

# %% trusted=true
year = 2016
regenerate = False

fn_ibio_results = os.path.join(RESULTS, 'sens_analysis_s6_ibio_{param}.csv')
fn_ploss_results = os.path.join(RESULTS, 'sens_analysis_s6_ploss_{param}.csv')

###########################
if regenerate:
    print('Regenerating....')
    
    ibio = [17.900000000000002, 179, 895, 1790]
    fn_ibio = os.path.join(MODEL_OUTPUTS_SENS_IBIO, 'model_outputs_{year}_ploss0.1_ibio{{v}}.nc'.format(year=year))
    _ = sensitivities_extract(fn_ibio, fn_ibio_results, ibio, pts_ps.loc['S6'].geometry, 'ibio')

    ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
    fn_ploss = os.path.join(MODEL_OUTPUTS_SENS_PLOS, 'model_outputs_{year}*_ploss{{v}}.nc'.format(year=year))
    _ = sensitivities_extract(fn_ploss, fn_ploss_results, ploss, pts_ps.loc['S6'].geometry, 'ploss')
else:
    print('INFO...using cached results (not regenerated)')
###########################

ibio_site_pop = pd.read_csv(fn_ibio_results.format(param='popsize'), index_col=0, parse_dates=True)
ibio_site_Gn = pd.read_csv(fn_ibio_results.format(param='Gn'), index_col=0, parse_dates=True)
ibio_site_end = pd.read_csv(fn_ibio_results.format(param='lastbloom'), index_col=0, parse_dates=True).squeeze()
ibio_site_end.index = ibio_site_end.index.astype(str)
# Fix columns to the same as those specified as columns in sensitivities_extract()
ibio_site_end.index = [str(int(np.round(float(i)))) for i in ibio_site_end.index]

ploss_site_pop = pd.read_csv(fn_ploss_results.format(param='popsize'), index_col=0, parse_dates=True)
ploss_site_Gn = pd.read_csv(fn_ploss_results.format(param='Gn'), index_col=0, parse_dates=True)
ploss_site_end = pd.read_csv(fn_ploss_results.format(param='lastbloom'), index_col=0, parse_dates=True).squeeze()
ploss_site_end.index = ploss_site_end.index.astype(str)
# Fix columns
ploss_site_end.index = [str(int(float(c)*100))+'%' for c in ploss_site_end.index]


# Find the start of each bloom
ibio_start = bloom_start(ibio_site_pop)
ploss_start = bloom_start(ploss_site_pop)

# Use start of each bloom to calculate net growth
ibio_sum = to_kgDWgrid(active_bloom_sum(ibio_site_Gn, ibio_start, ibio_site_end))
ploss_sum = to_kgDWgrid(active_bloom_sum(ploss_site_Gn, ploss_start, ploss_site_end))

# %% trusted=true
## Start plotting
letgen = letters()
fig = plt.figure(figsize=(6,4))
# rows, cols
gs = fig.add_gridspec(2,3)

# Starting population
cmap_ibio = sns.color_palette('flare', n_colors=len(ibio_sum))
ax_ibio_pop = fig.add_subplot(gs[0, 0:2])
sns.lineplot(
    data=ibio_site_pop,
    ax=ax_ibio_pop,
    palette=cmap_ibio,
    dashes=False,
    legend=False
)
ax_ibio_pop.set_yscale('log')
label_panel(ax_ibio_pop, next(letgen))
ax_ibio_pop.set_ylabel(LABEL_P)

# Add text labels to each line
n = 0
for p in ibio_site_pop.columns:
    if p == '179':
        xloc = dt.datetime(2016,10,11)
    else:
        xloc = dt.datetime(2016,10,3)

    ax_ibio_pop.text(xloc, ibio_site_pop[p].iloc[-1], p, 
                     color=cmap_ibio[n],
                     va='center_baseline', ha='left'
                    )
    n += 1

# ----------
ax_ibio_Gn = fig.add_subplot(gs[0, 2])
ax_ibio_Gn.bar(np.arange(0, len(ibio_sum)),ibio_sum, color=cmap_ibio)
ax_ibio_Gn.set_xticks(np.arange(0, len(ibio_sum)), ibio_sum.index)
ax_ibio_Gn.set_ylabel(LABEL_TOT_GN_PX)
ax_ibio_Gn.set_xlabel('$P_{(t=0)}$ (%s)' %UNIT_DW)
ax_ibio_Gn.set_ylim(0, 2700)
label_panel(ax_ibio_Gn, next(letgen))

# --------------
# Ploss
cmap_ploss = sns.color_palette('crest', n_colors=len(ploss_sum))
ax_ploss_pop = fig.add_subplot(gs[1, 0:2])
sns.lineplot(
    data=ploss_site_pop,
    ax=ax_ploss_pop,
    palette=cmap_ploss,
    dashes=False,
    legend=False
)
ax_ploss_pop.set_yscale('log')
ax_ploss_pop.set_ylabel(LABEL_P)
label_panel(ax_ploss_pop, next(letgen))

# Add text labels to each line
n = 0
for p in ploss_site_pop.columns:
    ydelta = 0
    xloc = dt.datetime(2016,10,3)
    if p == '10%':
        ydelta = 150
        xloc = dt.datetime(2016,9,29)
    if p == '15%':
        xloc = dt.datetime(2016,9,12)
        ydelta = 150
        
    ax_ploss_pop.text(xloc, ploss_site_pop[p].iloc[-1]+ydelta, p, 
                     color=cmap_ploss[n],
                     va='center', ha='left'
                    )
    n += 1


# ------------
ax_ploss_Gn = fig.add_subplot(gs[1, 2])
ax_ploss_Gn.bar(np.arange(0, len(ploss_sum)),ploss_sum, color=cmap_ploss)
ax_ploss_Gn.set_xticks(np.arange(0, len(ploss_sum)), [int(v[:-1]) for v in ploss_sum.index])
ax_ploss_Gn.set_ylabel(LABEL_TOT_GN_PX)
ax_ploss_Gn.set_xlabel(r'$\Theta$ (%)')
label_panel(ax_ploss_Gn, next(letgen))


# ------------
for ax in [ax_ibio_pop, ax_ploss_pop]:
    ax.set_ylim(0, 120000)
    ax.set_xlabel('')

plt.tight_layout()
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig1_phenological.pdf'), bbox_inches='tight')

with pd.ExcelWriter(os.path.join(RESULTS, 'fig1.xlsx')) as w:
    ibio_site_pop.to_excel(w, sheet_name='A_ngDWml')
    ibio_sum.to_excel(w, sheet_name='B_kgDW_in_10sqkm')
    ploss_site_pop.to_excel(w, sheet_name='C_ngDWml')
    ploss_sum.to_excel(w, sheet_name='D_kgDW_in_10sqkm')

# %% [markdown]
# ### Fig. 2: Sensitivity to environmental parameters

# %% trusted=true
year = 2016
regenerate = False

fn_sd = os.path.join(RESULTS, 'sens_analysis_s6_snowdepth_{param}.csv')
fn_li = os.path.join(RESULTS, 'sens_analysis_s6_swd_{param}.csv')
fn_t = os.path.join(RESULTS, 'sens_analysis_s6_temperature_{param}.csv')

if regenerate:

    depths = [0.01, 0.05, 0.10, 0.2, 0.4, 0.8]    
    fn_snow = os.path.join(MODEL_OUTPUTS_SENS_SNOW, 'model_outputs_{year}_ibio179_ploss0.1_snow{{v}}.nc'.format(year=year))
    _ = sensitivities_extract(fn_snow, fn_sd, depths, pts_ps.loc['S6'].geometry, 'sd')

    lights = [1, 10, 100, 200]
    fn_light = os.path.join(MODEL_OUTPUTS_SENS_LIGH, 'model_outputs_{year}_ibio179_ploss0.1_light{{v}}.nc'.format(year=year))
    _ = sensitivities_extract(fn_light, fn_li, lights, pts_ps.loc['S6'].geometry, 'swd')
    
    temps = [0, 0.25, 0.5, 1.0]
    fn_temp = os.path.join(MODEL_OUTPUTS_SENS_TEMP, 'model_outputs_{year}_ibio179_ploss0.1_temp{{v}}.nc'.format(year=year))
    _ = sensitivities_extract(fn_temp, fn_t, temps, pts_ps.loc['S6'].geometry, 'T')
    
# Read in all data sets generated above.
sd_site_popsize = pd.read_csv(fn_sd.format(param='popsize'), index_col=0, parse_dates=True)
li_site_popsize = pd.read_csv(fn_li.format(param='popsize'), index_col=0, parse_dates=True)
t_site_popsize = pd.read_csv(fn_t.format(param='popsize'), index_col=0, parse_dates=True)

sd_site_Gn = pd.read_csv(fn_sd.format(param='Gn'), index_col=0, parse_dates=True)
li_site_Gn = pd.read_csv(fn_li.format(param='Gn'), index_col=0, parse_dates=True)
t_site_Gn = pd.read_csv(fn_t.format(param='Gn'), index_col=0, parse_dates=True)

sd_site_end = pd.read_csv(fn_sd.format(param='lastbloom'), index_col=0, parse_dates=True).squeeze()
sd_site_end.index = sd_site_end.index.astype(str)
li_site_end = pd.read_csv(fn_li.format(param='lastbloom'), index_col=0, parse_dates=True).squeeze()
li_site_end.index = li_site_end.index.astype(str)
t_site_end = pd.read_csv(fn_t.format(param='lastbloom'), index_col=0, parse_dates=True).squeeze()
t_site_end.index = t_site_end.index.astype(str)

# %% trusted=true
# First find the start of each bloom
t_start = bloom_start(t_site_popsize)
sd_start = bloom_start(sd_site_popsize)
li_start = bloom_start(li_site_popsize)

# Calculate the bloom duration
t_dur = t_site_end - t_start
sd_dur = sd_site_end - sd_start
li_dur = li_site_end - li_start

# Use start of each bloom to calculate net growth
t_sum = to_kgDWgrid(active_bloom_sum(t_site_Gn, t_start, t_site_end))
sd_sum = to_kgDWgrid(active_bloom_sum(sd_site_Gn, sd_start, sd_site_end))
li_sum = to_kgDWgrid(active_bloom_sum(li_site_Gn, li_start, li_site_end))


# %% trusted=true

fig = plt.figure(figsize=(4.5, 3.5))

# nrows, ncols
gs = fig.add_gridspec(2,2)

letgen = letters()

labels = ['T', 'Sn', 'SWD']
col_t = sns.color_palette(palette='Oranges', n_colors=5)[1:]
col_sd = sns.color_palette(palette='Blues', n_colors=7)[1:]
col_li = sns.color_palette(palette='Greens', n_colors=5)[1:]


# Ax Bloom onset 
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter([1]*len(t_site_popsize.columns), t_start, marker='o', c=col_t)
ax1.scatter([2]*len(sd_site_popsize.columns), sd_start, marker='o', c=col_sd)
ax1.scatter([3]*len(li_site_popsize.columns), li_start, marker='o', c=col_li)
ax1.xaxis.set_ticks([1,2,3], labels)
ax1.set_ylabel('Bloom start (DOY)')
label_panel(ax1, next(letgen), xy=(0.09,0.93))

# Ax Bloom duration
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter([1]*len(t_site_popsize.columns), t_dur, marker='o', c=col_t)
ax2.scatter([2]*len(sd_site_popsize.columns), sd_dur, marker='o', c=col_sd)
ax2.scatter([3]*len(li_site_popsize.columns), li_dur, marker='o', c=col_li)
ax2.xaxis.set_ticks([1,2,3], labels)
ax2.set_ylabel('Bloom duration (days)')
ax2.yaxis.set_ticks([80, 100, 120])
label_panel(ax2, next(letgen), xy=(0.09,0.93))

# Ax Max bloom size
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter([1]*len(t_site_popsize.columns), t_site_popsize.max(), marker='o', c=col_t)
ax3.scatter([2]*len(sd_site_popsize.columns), sd_site_popsize.max(), marker='o', c=col_sd)
ax3.scatter([3]*len(li_site_popsize.columns), li_site_popsize.max(), marker='o', c=col_li)
ax3.xaxis.set_ticks([1,2,3], labels)
ax3.set_ylabel(LABEL_PMAX)
ax3.yaxis.set_ticks([8000, 12000, 16000])
label_panel(ax3, next(letgen), xy=(0.09,0.93))

# Sum of bloom during active period
# Can't use simple sum, needs to be sum of active bloom period.
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter([1]*len(t_sum), t_sum, marker='o', c=col_t)
ax4.scatter([2]*len(sd_sum), sd_sum, marker='o', c=col_sd)
ax4.scatter([3]*len(li_sum), li_sum, marker='o', c=col_li)
ax4.xaxis.set_ticks([1,2,3], labels)
#ax4.yaxis.set_ticks([4.0e5, 6.0e5, 8.0e5, 1.0e6])
ax4.set_ylabel(LABEL_TOT_GN_PX)
label_panel(ax4, next(letgen), xy=(0.09,0.93))

sns.despine()

# Special legend building for the colour coded markers
import matplotlib.patches as mpatches
def build_legend_handles(colors, labels):
    handles = []
    # handles is a list, so append manual patch
    for c,l in zip(colors, labels):
        handles.append(mpatches.Patch(color=c, label=l))
    return handles

# plot the legend
leg_t = fig.legend(handles=build_legend_handles(col_t, t_site_popsize.columns), bbox_to_anchor=(1.17, 1.), frameon=False, title=r'T $^{o}C$')
leg_sd = fig.legend(handles=build_legend_handles(col_sd, sd_site_popsize.columns), bbox_to_anchor=(1.17, 0.71), frameon=False, title='SD m')
leg_li = fig.legend(handles=build_legend_handles(col_li, li_site_popsize.columns), bbox_to_anchor=(1.19, 0.35), frameon=False, title=r'SWD Wm$^{-2}$')

# Manually add the first legends back to the plot
ax4.add_artist(leg_t)
ax4.add_artist(leg_sd)

plt.tight_layout()

plt.savefig(os.path.join(RESULTS, 'fig2_env_sensitivity.pdf'))


# %% trusted=true
# Export to file
cols = ['start_DOY', 'dur_d', 'P_max_ngDWml', 'sigmaGn_kgDW_in_10sqkm']

df_T = pd.concat([t_start, t_dur, t_site_popsize.max(), t_sum], axis=1)
df_T.columns = cols

df_sd = pd.concat([sd_start, sd_dur, sd_site_popsize.max(), sd_sum], axis=1)
df_sd.columns = cols

df_li = pd.concat([li_start, li_dur, li_site_popsize.max(), li_sum], axis=1)
df_li.columns = cols

with pd.ExcelWriter(os.path.join(RESULTS, 'fig2.xlsx')) as w:
    df_T.to_excel(w, sheet_name='temperature')
    df_sd.to_excel(w, sheet_name='snow_depth')
    df_li.to_excel(w, sheet_name='shortwave_down')

# %% [markdown]
# ---
# ## Fig. 3: QMC analysis at Point sites

# %% trusted=true scrolled=true
# Extract time series of every single QMC experiment, exported to CSV for each Point site
regenerate = False
# To support the analysis in subsequent cells, run this cell twice, 
# once for the warm year (2019) and once for the cold year (2022)
years = (2019, 2019)

# Filename formatting according to time range
if years[0] == years[1]:
    t = years[0]
else:
    t = '{0}_{1}'.format(years[0], years[1])

# Only recompute from netcdfs if explicitly requested
if regenerate:
    for ix, row in pts_ps.iterrows():
        print(ix)
        store_pop = []
        store_Gn = []
        for year in range(years[0], years[1]+1):
            print(year)
            pop, Gn = get_qmc_ts(year, geom=row.geometry)
            store_pop.append(pop)
            store_Gn.append(Gn)
        
        pd.concat(store_pop, axis=0).to_csv(os.path.join(RESULTS,f'qmc_ts_dailypop_{ix}_{t}.csv'))
        pd.concat(store_Gn, axis=0).to_csv(os.path.join(RESULTS,f'qmc_ts_Gn_{ix}_{t}.csv'))


# Reload reduced QMC outputs from disk
sites_cold_pop = {}
sites_warm_pop = {}
sites_cold_Gn = {}
sites_warm_Gn = {}

for ix, row in pts_ps.iterrows():
    sites_cold_pop[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_ts_dailypop_{ix}_2022.csv'), index_col=0, parse_dates=True)
    sites_warm_pop[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_ts_dailypop_{ix}_2019.csv'), index_col=0, parse_dates=True)
    sites_cold_Gn[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_ts_Gn_{ix}_2022.csv'), index_col=0, parse_dates=True)
    sites_warm_Gn[ix] = pd.read_csv(os.path.join(RESULTS,f'qmc_ts_Gn_{ix}_2019.csv'), index_col=0, parse_dates=True)


# %% trusted=true
## Calculate total annual growth at each site

# First find end of active growth 
# This requires gross daily production, so have to revisit the netcdfs. 
# This is expensive so we want to do this as few times as possible, hence the quantile-based approach

def find_last_bloom_day_site(year, site_runs, geom, quantiles=[0.25, 0.5, 0.75]):
    """
    Inputs:

    - year
    - site_runs : df of one column per QMC experiment, rows are time series, for this site i nquestion.
    - geom : the
    Returns:
    (tuple of experimment IDs, tuple of LDOY for each requested quantile)
    """
    ranked = site_runs.max().sort_values()
    expts = []
    values = []
    for q in quantiles:
        expt_id = ranked.index[np.abs(ranked - ranked.quantile(q)).argmin()]
        fn = os.path.join(MODEL_OUTPUTS_MAIN, f'model_outputs_{year}_exp{expt_id}.nc')
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
    expts_warm[ix], ldoy_warm[ix] = find_last_bloom_day_site(2019, sites_warm_pop[ix], row.geometry)
    expts_cold[ix], ldoy_cold[ix] = find_last_bloom_day_site(2022, sites_cold_pop[ix], row.geometry)
    

# Now calculate total biomass at each quantile during the active growth period.
store = []
# This is a lookup table of how each tuple index links to the quantile.
q_lut = {0:0.25, 1:0.50, 2:0.75}

# Iterate site-by-site, adding two rows per site: one for the warm year, one for the cold year
for ix, row in pts_ps.iterrows():
    
    med = sites_cold_Gn[ix][sites_cold_pop[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][1]), expts_cold[ix][1]].sum()
    q75 = sites_cold_Gn[ix][sites_cold_pop[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][2]), expts_cold[ix][2]].sum() 
    q25 = sites_cold_Gn[ix][sites_cold_pop[ix] > 179].loc[:doy_to_datetime(2022, ldoy_cold[ix][0]), expts_cold[ix][0]].sum()
    iqr = q75 - q25
    store.append(dict(
        yr='2022',
        biomass=med,
        iqr=iqr,
        q75=q75,
        q25=q25,
        site=ix
    ))

    med = sites_warm_Gn[ix][sites_warm_pop[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][1]), expts_warm[ix][1]].sum()
    q75 = sites_warm_Gn[ix][sites_warm_pop[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][2]), expts_warm[ix][2]].sum() 
    q25 = sites_warm_Gn[ix][sites_warm_pop[ix] > 179].loc[:doy_to_datetime(2019, ldoy_warm[ix][0]), expts_warm[ix][0]].sum()
    iqr = q75 - q25
    store.append(dict(
        yr='2019',
        biomass=med,
        iqr=iqr,
        q75=q75,
        q25=q25,
        site=ix
    ))

Gn_sums = pd.DataFrame(store)    
# Convert from summed ng DW ml-1 to kg DW over grid
for col in ['biomass', 'iqr', 'q75', 'q25']:
    Gn_sums[col] = Gn_sums[col].apply(to_kgDWgrid)

# %% trusted=true
letgen = letters()

fig = plt.figure(figsize=(7.5, 4))
# nrows, ncols
gs = fig.add_gridspec(4,3)

col_warm = sns.set_hls_values('tab:orange', l=0.6)
col_cold = sns.set_hls_values('tab:blue', l=0.6)

## Plots of daily population size
def _plot_site(ax, site_name):

    
    ## Warm
    # Plot all model runs
    sites_warm_pop[site_name].plot(legend=False, color=col_warm, alpha=0.05, ax=ax, linewidth=0.5)
    # Plot specific metrics
    metrics = pd.concat([
        sites_warm_pop[site_name].quantile(0.5, axis=1), 
        sites_warm_pop[site_name].quantile(0.25, axis=1), 
        sites_warm_pop[site_name].quantile(0.75, axis=1)
        ], axis=1)
    metrics.columns = ['median', 'q25', 'q75']
    metrics['median'].plot(color='tab:orange', linewidth=2, ax=ax)    
    metrics['q25'].plot(color='tab:orange', ax=ax)    
    metrics['q75'].plot(color='tab:orange', ax=ax)
    # Save metrics back to df so that they can be exported to Excel below
    sites_warm_pop[site_name] = pd.concat([metrics, sites_warm_pop[site_name]], axis=1)

    ## Cold
    # Take copy of cold site purely so that we can adjust time to match the warm year,
    # so that we can plot them on same x axis
    # n.b. this is a bit slow, would probably be better to reindex a different way...
    ts = deepcopy(sites_cold_pop[site_name])
    ts.index = ts.index - dt.timedelta(days=365*3)
    # Plot all model runs
    ts.plot(legend=False, color=col_cold, alpha=0.05, ax=ax, linewidth=0.5)
    # Plot specific metrics
    metrics = pd.concat([
        ts.quantile(0.5, axis=1), 
        ts.quantile(0.25, axis=1), 
        ts.quantile(0.75, axis=1)
    ], axis=1)
    metrics.columns = ['median', 'q25', 'q75']
    metrics['median'].plot(color='tab:blue', linewidth=2, ax=ax)    
    metrics['q25'].plot(color='tab:blue', ax=ax)    
    metrics['q75'].plot(color='tab:blue', ax=ax)    
    # Re-adjust datetime and save back to main df for Excel export
    metrics.index = metrics.index + dt.timedelta(days=365*3)
    sites_cold_pop[site_name] = pd.concat([metrics, sites_cold_pop[site_name]], axis=1)
    
    ax.set_title(site_name)
    ax.set_ylim(0, 35000)
    m = dates.MonthLocator() 
    ax.xaxis.set_major_locator(m)
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    ax.set_xlabel('')
    ax.set_ylabel(f'P ({UNIT_DW})')
    label_panel(ax, next(letgen))
    
    
ax1 = fig.add_subplot(gs[0:2, 0])
_plot_site(ax1, 'UPE')
ax2 = fig.add_subplot(gs[2:4, 0])
_plot_site(ax2, 'S6')
ax3 = fig.add_subplot(gs[0:2, 1])
_plot_site(ax3, 'Mittivak')
ax4 = fig.add_subplot(gs[2:4, 1])
_plot_site(ax4, 'South')

## ------------
## Total biomass by site for cold and warm years as a bar plot
# (if we used full ensemble here then we could do this as a boxplot, but would need to calculate the active bloom duration for all experiments in the ensemble)
ax5 = fig.add_subplot(gs[2:4, 2])

Gn_sums = Gn_sums.rename({'yr':'Year'}, axis=1)
g = sns.barplot(
    data=Gn_sums,
    x="site", y="biomass", hue="Year",
    ax=ax5
)

# The seaborn errorbar kwarg doesn't allow to pass our desired errorbar values, so we need to drop
# down to the matplotlib interface to add these manually...:
# (we can't use the native methods unless we provide data of the full ensemble for calculation, which 
# is precisely what we need to avoid given how expensive the I/O on this is)

# Add errorbars (IQR) for cold year
xpos = 0
for ix, row in Gn_sums[Gn_sums.Year == '2022'].iterrows():
    ax5.plot([xpos-0.2,xpos-0.2], [row.q25, row.q75], color='k')
    xpos += 1

# Add errorbars (IQR) for warm year
xpos = 0
for ix, row in Gn_sums[Gn_sums.Year == '2019'].iterrows():
    ax5.plot([xpos+0.2,xpos+0.2], [row.q25, row.q75], color='k')
    xpos += 1

label_panel(ax5, next(letgen))
ax5.legend(bbox_to_anchor=(1, 1.2), frameon=False) #loc='upper right',
ax5.set_ylabel(LABEL_TOT_GN_PX)
ax5.set_xlabel('')


# Locations map
ax_map = fig.add_axes([0.65,0.48,0.3,0.53]) #fig.add_subplot(gs[0:2, 2])
for axis in ['top','bottom','left','right']:
    ax_map.spines[axis].set_linewidth(0)
im = mpimg.imread(os.path.join(WORK_ROOT, 'sites_context_map.png'))
ax_map.imshow(im,interpolation='none')
ax_map.set_xticks([])
ax_map.set_yticks([])
ax_map.set_facecolor('none')

sns.despine()
plt.tight_layout()

plt.savefig(os.path.join(RESULTS, 'fig3_site_ensembles.pdf'), bbox_inches='tight')


# %% trusted=true
with pd.ExcelWriter(os.path.join(RESULTS, 'fig3.xlsx')) as w:
    sites_warm_pop['UPE'].to_excel(w, sheet_name='A_UPE_2019_ngDWml')
    sites_cold_pop['UPE'].to_excel(w, sheet_name='A_UPE_2022_ngDWml')

    sites_warm_pop['S6'].to_excel(w, sheet_name='B_S6_2019_ngDWml')
    sites_cold_pop['S6'].to_excel(w, sheet_name='B_S6_2022_ngDWml')
    
    sites_warm_pop['Mittivak'].to_excel(w, sheet_name='C_MIT_2019_ngDWml')
    sites_cold_pop['UPE'].to_excel(w, sheet_name='C_MIT_2022_ngDWml')

    sites_warm_pop['South'].to_excel(w, sheet_name='D_SOUTH_2019_ngDWml')
    sites_cold_pop['South'].to_excel(w, sheet_name='D_SOUTH_2022_ngDWml')

    Gn_sums.to_excel(w, sheet_name='E_kgDW_in_10sqkm')

# %% trusted=true
# Now drop the summary columns that we added during plotting, otherwise the SFigs won't plot
cols = ['median', 'q25', 'q75']
for site in ['UPE', 'S6', 'South', 'Mittivak']:
    sites_warm_pop[site] = sites_warm_pop[site].drop(columns=cols)
    sites_cold_pop[site] = sites_cold_pop[site].drop(columns=cols)

# %% [markdown]
# ### SFigs. 1, 2: QMC parameters at S6

# %% trusted=true
qmc_expts = pd.read_csv(EXPTS_DESIGN_FN, index_col=0)

# %% trusted=true
qmc_expts.head()

# %% trusted=true
xlabels_nice = [r'$T_{th}$ ($^{o}C$)', r'${Sn}_{th}$ (m)', r'${SWD}_{th}$ (W m$^{-2}$)', r'$\Theta$']
for year in ['2019', '2022']:
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(5,6))
    
    stats = Gn_sums[(Gn_sums.Year == year) & (Gn_sums.site == 'S6')].squeeze()

    if year == '2019':
        _Gn = to_kgDWgrid(sites_warm_Gn['S6'].sum())
        _Pmax = sites_warm_pop['S6'].max()
    elif year == '2022':
        _Gn = to_kgDWgrid(sites_cold_Gn['S6'].sum())
        _Pmax = sites_cold_pop['S6'].max()
    else:
        raise ValueError
    
    _Gn.index = pd.to_numeric(_Gn.index)
    _Pmax.index = pd.to_numeric(_Pmax.index)

    corr_kws = {
        'bbox':dict(boxstyle='round,pad=0.3', facecolor='whitesmoke', edgecolor='none', alpha=0.6), 
        'ha':'right',
        'xy':(0.95, 0.8),
        'xycoords':'axes fraction'
    }
    
    n = 0
    ax_letters = ['a','c','e','g']
    for p in qmc_expts.columns:
        ax = axes[n, 0]
            
        ax.plot(qmc_expts[p], _Gn, 'o', markersize=2, alpha=0.2)
        
        ax.axhline(stats.biomass, color='k')
        ax.axhline(stats.q25, color='k', linestyle=':')
        ax.axhline(stats.q75, color='k', linestyle=':')

        sp = np.round(qmc_expts[p].corr(_Gn, method='spearman'), 2)
        ax.annotate(r'$r=' + str(sp) + '$', **corr_kws)
        ax.set_xlabel(xlabels_nice[n])
        if n == 1:
            ax.set_ylabel(r'$\Sigma G_N$ kg DW', y=-0.18)
        label_panel(ax, ax_letters[n])
        n += 1

    n = 0
    ax_letters = ['b','d','f','h']
    for p in qmc_expts.columns:
        ax = axes[n, 1]
        ax.plot(qmc_expts[p], _Pmax, 'o', alpha=0.2, markersize=2)

        sp = np.round(qmc_expts[p].corr(_Pmax, method='spearman'), 2)
        ax.annotate(r'$r=' + str(sp) + '$', **corr_kws)
        
        ax.set_xlabel(xlabels_nice[n])
        if n == 1:
            ax.set_ylabel(r'$P_{MAX}$ ng DW ml-1', y=-0.25)
        label_panel(ax, ax_letters[n])
        n += 1  
    
    plt.subplots_adjust(hspace=0.55, wspace=0.55)
    sns.despine()

    plt.savefig(os.path.join(RESULTS, f'SFig_QMC_params_{year}.pdf'))

# %% [markdown]
# ## Fig. 4: Measured-modelled comparison

# %% [markdown]
# ### Pt 1: Extract QMC data for 'offline' analysis

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
# This is expensive because we need to open all the QMC experiments in a given year
regenerate = False

if regenerate:
    outputs = []
    for year in data['year'].unique():
        print(year)
        meas_subset = data[data.year == year]
        
        for exp in range(1, NQMC+1):
            # Open the numbered experiment
            r = open_model_run(os.path.join(MODEL_OUTPUTS_MAIN, f'model_outputs_{year}_exp{exp}.nc'))
            for ix, row in meas_subset.iterrows():
                rr = r.cum_growth.sel(x=row.geometry.x, y=row.geometry.y, method='nearest').sel(TIME=row.date)
                outputs.append({'date':row.date, 'study':row.study, 'site':row.site, 'exptid':exp, 'biomass_dw':float(rr)})

    pts_modelled = pd.DataFrame(outputs)
    
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
    
    m = pd.merge(left=meas, right=modelled_merge, left_on=['study', 'date', 'site'], right_on=['study', 'date', 'site'], how='left')
    m
    
    # Save to CSV files
    pts_modelled.to_csv(os.path.join(RESULTS, 'qmc_modelled_at_observed_points_all_expts.csv'))
    m.to_csv(os.path.join(RESULTS, 'qmc_modelled_quantiles_vs_observed.csv'))


# %% [markdown]
# ### Pt 2: Load post-analysed dataset from C.W. and plot it

# %% trusted=true
# Measured-modelled data
mmdf = pd.read_csv(os.path.join(RESULTS, 'fig4_data.csv'))

# Model-derived IQR bounds around the 1:1 line, computed using least-squares by Chris
bounds = pd.read_csv(os.path.join(RESULTS, 'extra_for_ted.csv'))

# %% trusted=true
mmdf.head()

# %% trusted=true
bounds.head()

# %% trusted=true
plt.figure(figsize=(4, 2.5))

ax = plt.subplot()
plt.grid(linestyle=':')

# Plot a 1:1 line by using the modelled data plotted against itself.
origin = np.linspace(0,mmdf['modelled.50'].max(), 1000)
plt.plot(origin, origin, 'tab:cyan', marker='none', label='1:1', linewidth=1)

# Plot IQR of model values, based on linear regression by CW
plt.fill_between(bounds.x_vals, bounds.y_25, bounds.y_75, color='tab:cyan', alpha=0.5)

# Optional graphing of model IQR per-point
#combined = np.vstack((np.array(mmdf['modelled.50'] - mmdf['modelled.25']),np.array(mmdf['modelled.75'] - mmdf['modelled.50'])))
#plt.hlines(mmdf['mean'], mmdf['modelled.25'], mmdf['modelled.75'], alpha=0.5)

# Plot measured stdev
plt.vlines(mmdf['modelled.50'], mmdf['mean']+mmdf['sd'], mmdf['mean']-mmdf['sd'], color='grey', linewidth=0.5)
# Plot data points
plt.plot(mmdf['modelled.50'], mmdf['mean'], 'ok', markersize=3)

# Indicate stdev overshoots with triangle
over = mmdf[mmdf['sd'] > 30000]
plt.plot(over['modelled.50'], [39200]*len(over), marker='^', color='grey', linestyle='none', markersize=5)

# Highlight upe sample
upe = mmdf[mmdf['study'] == 'willi']
plt.plot(upe['modelled.50'], upe['mean'], 'o', markersize=3, color='gold')

plt.xlabel(f'Modelled ({UNIT_DW})')
plt.ylabel(f'Measured ({UNIT_DW})')
plt.ylim(-2000, 40000)
sns.despine()

plt.savefig(os.path.join(RESULTS, 'fig4.pdf'))

# %% trusted=true
# Organise a nicer workbook, which includes the model regression??

# %% [markdown]
# ## Ice sheet wide preparations

# %% [markdown]
# This needs to be done in two parts, both outside this notebook:
#
# 1. Run `calculate_qmc_run_summary_metrics.py`
# 2. Run `calculate_qmc_ensemble_metrics.py`

# %% trusted=true
# Set the path to the outputs provided by the second script.
fn_annual_summary = os.path.join(WORK_ROOT, RESULTS, 'model_outputs_QMCE_{year}_summary_q{q}.nc')

# %% [markdown]
# ## Fig. 5: Map small versus large bloom years

# %% trusted=true
d19 = xr.open_dataset(fn_annual_summary.format(year=2019, q=50))
d22 = xr.open_dataset(fn_annual_summary.format(year=2022, q=50))

# %% trusted=true
# Settings common to Fig. 5 and supplementary figures
crs = ccrs.NorthPolarStereo(central_longitude=-45., true_scale_latitude=70.)
cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

norm_max = colors.Normalize(vmin=0, vmax=30000)
kws_max = dict(norm=norm_max, cmap=cmap, rasterized=True, zorder=10, add_colorbar=False)

norm_sum = colors.Normalize(vmin=100, vmax=6000)
kws_sum = dict(norm=norm_sum, cmap=cmap, rasterized=True, zorder=10, add_colorbar=False)

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(4*1.2,6*1.2), subplot_kw={'projection':crs})

letgen = letters()

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

def plot_base(ax, contours=True, grid_labels=False):
    jgg_gdf.plot(ax=ax, color='#CDB380', edgecolor='none', alpha=1, zorder=1)
    basin_boundaries.plot(ax=ax, color='tab:cyan', edgecolor='tab:cyan', linewidth=0.5, alpha=0.5, zorder=20)
    # Coordinate shifts for basin name label locations, (x, y) in metres
    name_deltas = {
        'NO': (5e4, 0),
        'NE': (-4e4, 0),
        'CE': (-2e4, 0),
        'SE': (-2e4, 6e4),
        'SW': (0, 6e4),
        'CW': (0, 3e4),
        'NW': (3e4, 3e4)
    }
    for ix, basin in basins.iterrows():
        # if basin.SUBREGION1 in ['SE', 'CE']:
        #     continue
        x, y = basin.geometry.centroid.xy
        xd, yd = name_deltas[basin.SUBREGION1]
        ax.text(x[0]+xd, y[0]+yd, basin.SUBREGION1, ha='center', va='center', fontsize=7, zorder=30, color='tab:cyan', alpha=0.5)
        
    gris_outline.plot(ax=ax, color='whitesmoke', edgecolor='none', alpha=1, zorder=2)
    if contours:
        plot_contours(ax)
    ax.axis('off')

    gl = ax.gridlines(draw_labels=grid_labels, 
                 xlocs=[-55, -30], 
                 ylocs=[65, 75], 
                 x_inline=False, 
                 y_inline=False, 
                 ylim=(60, 85),
                 zorder=120, 
                xlabel_style={'zorder':200})
    return gl

    
plt.subplots_adjust(wspace=0, hspace=0)

# Sums
xr.apply_ufunc(to_kgDWgrid, d19.annual_net_growth_sum.rio.clip(basins.geometry.values)).plot(ax=axes[0,0], **kws_sum)
axes[0,0].set_title('2019')
label_panel(axes[0,0], next(letgen))

xr.apply_ufunc(to_kgDWgrid, d22.annual_net_growth_sum.rio.clip(basins.geometry.values)).plot(ax=axes[0,1], **kws_sum)
axes[0,1].set_title('2022')
label_panel(axes[0,1], next(letgen))

cbar_sum = fig.add_axes((0.9, 0.55, 0.04, 0.3))
cbar_sum_kws = {'label':LABEL_TOT_GN_PX, 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_sum, cmap=cmap), cax=cbar_sum, **cbar_sum_kws)


# Maxes
d19.annual_pop_max.rio.clip(basins.geometry.values).plot(ax=axes[1,0], **kws_max)
axes[1,0].set_title('')
label_panel(axes[1,0], next(letgen))

d22.annual_pop_max.rio.clip(basins.geometry.values).plot(ax=axes[1,1], **kws_max)
axes[1,1].set_title('')
label_panel(axes[1,1], next(letgen))

cbar_max = fig.add_axes((0.9, 0.15, 0.04, 0.3))
cbar_max_kws={'label':LABEL_PMAX, 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_max, cmap=cmap), cax=cbar_max, **cbar_max_kws)

draw_labels = {"left": "y"}
for ax in axes.flatten():
    _gl = plot_base(ax, grid_labels=draw_labels)
    draw_labels = False

fig.text(0.15, 0.505, r'55$^\circ$W')
fig.text(0.435, 0.515, r'30$^\circ$W')

plt.savefig(os.path.join(RESULTS, 'fig_map_sum_max_QMC_2019_2022.pdf'), dpi=300, bbox_inches='tight')

# %% [markdown]
# ### SFig. 3: annual bloom extents, year by year
#
# This Figure takes plotting styles directly from previous section

# %% trusted=true
annual_g_max = xr.open_mfdataset(fn_annual_summary.format(year='*', q=50), preprocess=lambda x: x.annual_pop_max).annual_pop_max

# %% trusted=true
fg = annual_g_max.rio.clip(basins.geometry.values).plot(figsize=(6,9), col='TIME', col_wrap=5, subplot_kws={'projection':crs}, **kws_max)
titles = np.arange(2000, 2023, 1)
tn = 0
for ax in fg.axs.flat:
    # We do very simple plotting otherwise the kernel dies before the figure can be finished
    #jgg_gdf.plot(ax=ax, color='#CDB380', edgecolor='none', alpha=1, zorder=1)
    #gris_outline.plot(ax=ax, color='whitesmoke', edgecolor='none', alpha=1, zorder=2)
    ax.coastlines(color='grey', linewidth=0.5)
    ax.set_extent([-56, -31, 57, 84], crs=ccrs.PlateCarree())
    ax.axis('off')

    ax.set_title(titles[tn])
    if tn == len(titles)-1:
        break
    tn+=1
    
cbar_max = fg.fig.add_axes((0.7, 0.05, 0.03, 0.15))
cbar_max_kws={'label':LABEL_PMAX, 'shrink':0.8}
plt.colorbar(mappable=cm.ScalarMappable(norm=norm_max, cmap=cmap), cax=cbar_max, **cbar_max_kws)

plt.subplots_adjust(hspace=0.05)
plt.savefig(os.path.join(RESULTS, 'fig_suppl_annual_bloom_max.pdf'), dpi=300, bbox_inches='tight')    


# %% [markdown]
# ## Fig. 6: Time-series / Sector-by-sector analysis

# %% trusted=true
qmc_med = open_model_run(os.path.join(RESULTS, fn_annual_summary.format(year='*', q=50)))
qmc_q25 = open_model_run(os.path.join(RESULTS, fn_annual_summary.format(year='*', q=25))).rio.write_crs(CRS)
qmc_q75 = open_model_run(os.path.join(RESULTS, fn_annual_summary.format(year='*', q=75))).rio.write_crs(CRS)

# %% trusted=true
KG_TO_T = 0.001

# Calculate ice-sheet-wide, convert to tonnes
gris_med = (xr.apply_ufunc(to_kgDWgrid, qmc_med.annual_net_growth_sum, dask='allowed') * (mar.MSK/100)).sum(dim=('x','y')) * KG_TO_T
gris_25 = (xr.apply_ufunc(to_kgDWgrid, qmc_q25.annual_net_growth_sum, dask='allowed') * (mar.MSK/100)).sum(dim=('x','y')) * KG_TO_T
gris_75 = (xr.apply_ufunc(to_kgDWgrid, qmc_q75.annual_net_growth_sum, dask='allowed') * (mar.MSK/100)).sum(dim=('x','y')) * KG_TO_T

# Calculate sums of bloom size per each elevation class in each sector of the ice sheet.
store = {}
store_q75 = {}
store_q25 = {}
for ix, sector in basins.iterrows():
    d = xr.apply_ufunc(to_kgDWgrid, qmc_med.annual_net_growth_sum.rio.clip([sector.geometry], all_touched=True, drop=True), dask='allowed') * (mar.MSK/100)
    sh = mar.SH.rio.clip([sector.geometry], all_touched=True, drop=True)
    sector_sum_bio_by_elev = d.groupby_bins(sh, bins=np.arange(0,2200, 400), labels=np.arange(0,2000, 400)).sum().to_pandas() * KG_TO_T
    store[sector.SUBREGION1] = sector_sum_bio_by_elev
    q75 = xr.apply_ufunc(to_kgDWgrid, qmc_q75.annual_net_growth_sum.rio.clip([sector.geometry], all_touched=True, drop=True), dask='allowed') * (mar.MSK/100)
    q25 = xr.apply_ufunc(to_kgDWgrid, qmc_q25.annual_net_growth_sum.rio.clip([sector.geometry], all_touched=True, drop=True), dask='allowed') * (mar.MSK/100)
    store_q75[sector.SUBREGION1] = q75.sum(dim=('x','y')) * KG_TO_T
    store_q25[sector.SUBREGION1] = q25.sum(dim=('x','y')) * KG_TO_T

# %% trusted=true
# Load the GEUS SMB dataset, downloaded from their dataverse.
mb = pd.read_csv(os.path.join(WORK_ROOT, 'MB_SMB_D_BMB_ann.csv'), index_col='time', parse_dates=True)
smb = mb.loc['2000-01-01':'2022-01-01'].SMB

# %% trusted=true
fig, axes = plt.subplots(figsize=(5,6), ncols=2, nrows=4)
axes = axes.flatten()

letgen = letters()

# All sectors biomass
ax = axes[0]
ax.fill_between(qmc_med['TIME'], gris_25, gris_75, color='grey', alpha=0.5, edgecolor='none')
ax.plot(qmc_med['TIME'], gris_med, color='k')
ax.xaxis.set_major_locator(dates.YearLocator(5))
# ax.annotate('All sectors', xy=(0.04, 0.78), xycoords='axes fraction', style='italic',
#            horizontalalignment='left', verticalalignment='top', fontsize=8)
ax.set_ylim(0, 10000)
label_panel(ax, next(letgen))

# All sectors SMB
axr = ax.twinx()
axr.plot(smb.index, smb, zorder=1, color='gray', linestyle='--')
axr.set_ylabel('SMB (Gt)', color='gray', y=1.2, rotation=0)
axr.spines['right'].set_color('gray')
axr.tick_params(axis='y', colors='gray')
axr.set_ylim(0, 500)
axr.set_yticks([0, 250, 500])

sns.despine(ax=ax)
sns.despine(ax=axr, right=False, top=True)

# Sector by sector biomass
n = 1
for s in ['NO','NW','NE','CW','CE','SW','SE']:
    r = store[s]
    
    ax = axes[n]
    ax.stackplot(r.index, r.T, labels=r.columns, colors=sns.color_palette('flare_r', n_colors=5), edgecolor='none', linewidth=0)
    ax.vlines(r.index, store_q25[s].T, store_q75[s].T, color='k', linewidth=1, alpha=0.8)
    ax.set_ylim(0, 1400)
    ax.xaxis.set_major_locator(dates.YearLocator(5))
    
    label_panel(ax, next(letgen))
    ax.annotate(s, xy=(0.04, 0.75), xycoords='axes fraction', style='italic',
           horizontalalignment='left', verticalalignment='top')
    sns.despine(ax=ax)
    n += 1

ax.legend(loc=(1,0.1), frameon=False, title='Elev. m', title_fontsize=SMALL_SIZE)
plt.subplots_adjust(hspace=0.5, wspace=0.4)
fig.text(0.025, 0.5, LABEL_TOT_GN_T, va='center', rotation=90)
plt.savefig(os.path.join(RESULTS, 'fig_sectors_annual_sum_qmc_med.pdf'), bbox_inches='tight')

# %% trusted=true
with pd.ExcelWriter(os.path.join(RESULTS, 'fig6.xlsx')) as w:
    main_ts = np.round(pd.concat([gris_med.to_pandas(), gris_25.to_pandas(), gris_75.to_pandas()], axis=1), 1)
    main_ts.columns = ['SigmaGn_median_tonnesDW', 'SigmaGn_q25_tonnesDW', 'SigmaGn_q75_tonnesDW']
    main_ts.to_excel(w, sheet_name='A_GrIS')

    letgen = letters(start=1)
    for s in ['NO','NW','NE','CW','CE','SW','SE']:
        reg_ts = np.round(pd.concat([store[s], store[s].sum(axis=1), store_q25[s].to_pandas(), store_q75[s].to_pandas()], axis=1), 1)
        c1 = [f'SigmaGn_median_{e}m_tonnesDW' for e in store[s].columns] 
        c1.extend(['sum_SigmaGn_median_tonnesDW', 'sum_SigmaGn_q25_tonnesDW', 'sum_SigmaGn_q75_tonnesDW'])
        reg_ts.columns = c1
        name = '{L}_{S}'.format(L=next(letgen).upper(), S=s)
        reg_ts.to_excel(w, sheet_name=name)


# %% [markdown]
# ## Fig. 7: Extent of blooms / % coverage of ice sheet by blooms and trend analysis

# %% trusted=true
def calc_bloom_extent(thresh_pop):
    bloom_extent_sqkm = ((qmc_med.annual_pop_max > thresh_pop) * (mar.MSK/100) * 10**2).sum(dim=('x','y')).to_pandas()
    bloom_extent_sqkm_q25 = ((qmc_q25.annual_pop_max > thresh_pop) * (mar.MSK/100) * 10**2).sum(dim=('x','y')).to_pandas()
    bloom_extent_sqkm_q75 = ((qmc_q75.annual_pop_max > thresh_pop) * (mar.MSK/100) * 10**2).sum(dim=('x','y')).to_pandas()
    bloom_extent_df = pd.concat([bloom_extent_sqkm, bloom_extent_sqkm_q25, bloom_extent_sqkm_q75], axis=1)
    bloom_extent_df.columns = ['med', 'q25', 'q75']
    bloom_extent_df['year'] = bloom_extent_df.index.year
    return bloom_extent_df
extent_179 = calc_bloom_extent(179)
extent_2000 = calc_bloom_extent(2000)

# %% trusted=true
fig, ax = plt.subplots(figsize=(3.5,2))

extent_179.med.plot(marker='^', markersize=3, linewidth=0.5, ax=ax, label=f'Min. P 179 {UNIT_DW}')
extent_2000.med.plot(marker='.', linewidth=0.5, ax=ax, label=f'Min. P 2000 {UNIT_DW}')
ax.fill_between(extent_2000.index, extent_2000.q25, extent_2000.q75, alpha=0.2, color='tab:orange')
plt.ylabel('Bloom extent (sq km)')
plt.ylim(0, 5e5)
plt.xlabel('')
plt.xlim('1999-01-01', '2023-01-01')
sns.despine()
plt.legend(frameon=False)

# %% trusted=true
with pd.ExcelWriter(os.path.join(RESULTS, 'fig7.xlsx')) as w:
    cols = ['area_median_sqkm', 'area_q25_sqkm', 'area_q75_sqkm']
    e179 = extent_179.drop(columns=['year'])
    e179.columns = cols
    e179.to_excel(w, sheet_name='Min. P 179 ngDWml')

    e2000 = extent_2000.drop(columns=['year'])
    e2000.columns = cols
    e2000.to_excel(w, sheet_name='Min. P 2000 ngDWml')

# %% trusted=true
# Average bloom extent
extent_179.mean()

# %% [markdown]
# ### Time regressions for text

# %% trusted=true
m = sm.OLS(extent_179.med, sm.add_constant(extent_179.year))
f = m.fit()
f.summary()

# %% trusted=true
f.params

# %% trusted=true
f.params[1]

# %% trusted=true
x = np.arange(2000, 2024)
y = f.params[1] * x  + f.params[0]
plt.plot(x, y, 'o')

# %% trusted=true
m = sm.OLS(extent_2000.med, sm.add_constant(extent_2000.year))
f = m.fit()
f.summary()

# %% [markdown]
# ## Extra statistics for text
#
# (Where not already computed in the sub-sections above)

# %% [markdown]
# ### GrIS wide min and max bloom productivity statistics

# %% trusted=true
biom_df = np.round(pd.concat([gris_med.to_pandas(), gris_25.to_pandas(), gris_75.to_pandas()], axis=1), 0)
biom_df.columns = ['med', 'q25', 'q75']
biom_df.to_csv(os.path.join(RESULTS, 'gris_wide_biomass_production_tonnes.csv'))
biom_df

# %% [markdown]
# ### Maximum P_MAX experienced

# %% trusted=true
d19 = xr.open_dataset(fn_annual_summary.format(year=2019, q=50))

# %% trusted=true
d_all = xr.open_mfdataset(fn_annual_summary.format(year='*', q=50))
d_all.annual_pop_max.max().compute()


# %% trusted=true
d_all.annual_pop_max.max(dim=('x','y')).compute().plot()


# %% [markdown]
# ### Organic carbon production potential

# %% trusted=true
def compute_carb(ds, q):
    ww = (xr.apply_ufunc(dw_to_ww, ds.annual_net_growth_sum, dask='allowed') * (mar.MSK/100)).sum(dim=('x','y'))
    carb = to_carbon(ww, q).compute().to_pandas() * KG_TO_T
    return carb

def compute_carb_at_q(q):
    carb_med = compute_carb(qmc_med, q)
    carb_q25 = compute_carb(qmc_q25, q)
    carb_q75 = compute_carb(qmc_q75, q)
    carb_df = np.round(pd.concat([carb_med, carb_q25, carb_q75], axis=1), 0)
    carb_df.columns = ['med', 'q25', 'q75']
    carb_df.to_csv(os.path.join(RESULTS, f'gris_wide_carbon_production_tonnes_{q}.csv'))
    return carb_df


# %% trusted=true
carb_q106 = compute_carb_at_q(106)
carb_q420 = compute_carb_at_q(420)

# %% trusted=true
carb_q420

# %% trusted=true
