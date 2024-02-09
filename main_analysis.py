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
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(jobs=4)
client = Client(cluster)

# %% trusted=true
client

# %% [markdown]
# ## Paths/Settings

# %% trusted=true
# A 'reference' MAR output, mainly for the mask and georeferencing
MAR_REF = '/flash/tedstona/MARv3.14-ERA5-10km/MARv3.14.0-10km-daily-ERA5-2017.nc'
# GrIS drainage basins
BASINS_FILE = '/flash/tedstona/L0data/Greenland_Basins_PS_v1_4_2_regions/Greenland_Basins_PS_v1_4_2_regions.shp'
# Growth model start and end
YR_ST = 2000
YR_END = 2022
START_POP = 179
# Algal growth model sensitivity runs (accessed by wildcard in script)
MODEL_OUTPUTS_SENS_IBIO = '/flash/tedstona/williamson/outputs/sensitivity_ibio/'
MODEL_OUTPUTS_SENS_PLOS = '/flash/tedstona/williamson/outputs/sensitivity_ploss/'
# Main run of algal growth model
MODEL_OUTPUTS_MAIN = '/flash/tedstona/williamson/outputs/main_outputs/'
# Measured biomass datasets
MEAS_BIO = '/flash/tedstona/williamson/glacier_algal_biomass_datasets/'
# Where CSV and figure files will be saved to
RESULTS = '/flash/tedstona/williamson/results/'


# %% [markdown]
# ## Analysis functions

# %% trusted=true
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
    """

    #convert from cells per ml to pg C ml assuming 106 pg C per cell
    x = x * 106

    #- pg C ml to pg C per L
    x = x * 1000

    #- pg C per l to pg C per m2 #using conversion from Williamson et al. 2018
    x = x * 1.061

    #- pg C per m2 to pg C per km2
    x = x * 10^6

    #- pg C per km2 to kg of C per km2
    x = x * 10^-15

    #total kg.C.per pixel
    #total kg of C per km2 * number of km2 per pixel 
    total_kg_C_pixel = total_kg_C_km2 * grid_km**2
    
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
gris_outline = gpd.read_file('/home/geoscience/nobackup_cassandra/L0data/gris_only_outline/greenland_icesheet_fix.shp')
gris_outline = gris_outline.to_crs(3413)

# World land areas
greenland = gpd.read_file('/home/geoscience/nobackup_cassandra/L0data/NaturalEarth/ne_10m_land/ne_10m_land.shp')
# Crop world land areas to Greenland and surrounding areas
bbox = gpd.read_file('/home/geoscience/nobackup_cassandra/L0data/greenland_region_bbox/greenland_area_bbox.shp').to_crs(3413)
just_greenland = gpd.clip(greenland.to_crs(3413), bbox)

# Manually isolate contiguous Greenland polygon from the main multi-polygon.
jg = just_greenland.filter(items=[0], axis=0)
jgg = jg.loc[0].geometry
jgg_poly = list(jgg.geoms)[9]
jgg_gdf = gpd.GeoDataFrame({'ix':[1,]}, geometry=[jgg_poly], crs=3413)

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
# ### Sensitivity to starting biomass term

# %% trusted=true
regenerate = False
if regenerate:
    ibio = [17.900000000000002, 179, 895, 1790]
    store = {}
    for ib in ibio:
        print(ib)
        sens_ibio = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_IBIO, f'*_ibio{ib}.nc'))
        sens_ibio['x'] = mar.x
        sens_ibio['y'] = mar.y
        store[ib] = sens_ibio.cum_growth.where(mar.MSK > 50).where(sens_ibio.cum_growth > ib).resample(TIME='1AS').quantile(0.9, dim='TIME').median(dim=('x','y')).to_pandas()
    ibio_quant90 = pd.DataFrame(store)
    ibio_quant90.to_csv(os.path.join(RESULTS, 'ibio_quant90.csv'))
else:
    ibio_quant90 = pd.read_csv(os.path.join(RESULTS, 'ibio_quant90.csv'), index_col=0, parse_dates=True)
    ibio_quant90.columns = [int(np.round(float(c))) for c in ibio_quant90.columns]

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
    
    v = sens_ibio.cum_growth.sel(x=pts_ps.loc['UPE'].geometry.x, y=pts_ps.loc['UPE'].geometry.y, method='nearest').to_pandas()
    site_store[ib] = v
ibio_max = pd.DataFrame(store)
ibio_upe = pd.DataFrame(site_store)

# %% trusted=true
ibio_upe.loc['2016'].plot()

# %% trusted=true
ibio_max.plot(
    logy=False,
    legend=False,
    colormap=sns.color_palette('flare', as_cmap=True)
)

# %%
store = {}
for ix, row in pts.iterrows():
    
ts = pd.DataFrame(store)

# %% [markdown] tags=[]
# ### Sensitivity to loss term

# %% trusted=true
regenerate = False
if regenerate:
    ploss = [0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.5]
    store = {}
    for pl in ploss:
        print(pl)
        sens_ploss = xr.open_mfdataset(os.path.join(MODEL_OUTPUTS_SENS_PLOS, f'*_ploss{pl}.nc'))
        sens_ploss['x'] = mar.x
        sens_ploss['y'] = mar.y
        store[pl] = sens_ploss.cum_growth.where(mar.MSK > 50).where(sens_ploss.cum_growth > 179).resample(TIME='1AS').quantile(0.9, dim='TIME').median(dim=('x','y')).to_pandas()
    ploss_quant90 = pd.DataFrame(store)
    ploss_quant90.to_csv(os.path.join(RESULTS, 'ploss_quant90.csv'))
else:
    ploss_quant90 = pd.read_csv(os.path.join(RESULTS, 'ploss_quant90.csv'), index_col=0, parse_dates=True)
    percs = [str(int(float(c)*100))+'%' for c in ploss_quant90.columns]
    ploss_quant90.columns = percs

# %% [markdown]
# ### Plot sensitivity

# %% trusted=true
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

# %% trusted=true
# tmp = pd.DataFrame(store)
# tmp
# tmp.plot(logy=True)

# sens_ibio_test = xr.open_mfdataset(f'/flash/tedstona/williamson/outputs/sensitivity_ibio/*_ibio179.nc')
# sens_ibio_test['x'] = mar.x
# sens_ibio_test['y'] = mar.y

# sens_ibio_test

# sens_ibio_test.cum_growth.where(mar.MSK > 50).where(sens_ibio_test.cum_growth > 179).resample(TIME='1AS').sum().plot(col='TIME', col_wrap=4)

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
year = 2017

plt.figure(figsize=(3.5,2.5))
for site in ts.columns:
    data = ts.loc[str(year)][site]
    plt.plot(data.index, data, c=pts.loc[site]['color'], label=site, linewidth=1.2)
    doy_end = int(last_prod_doy.sel(x=pts_ps.loc[site].geometry.x, y=pts_ps.loc[site].geometry.y, method='nearest').sel(TIME=str(year)).values[0])
    date_end = dt.datetime.strptime(f'{year}-{doy_end}', '%Y-%j')
    plt.plot(date_end, data.loc[date_end], 'd', c=pts.loc[site]['color'])
sns.despine()
plt.legend(frameon=False)
plt.ylabel('Biomass (ng DW ml$^{-1}$)')
plt.ylim(0,25000)
plt.savefig(os.path.join(RESULTS, f'fig_timeseries_{year}.pdf'), bbox_inches='tight')

# %% [markdown]
# ### Map small versus large bloom years

# %% trusted=true
# Calculate metrics
valid_growth = main_outputs.cum_growth.where(main_outputs.TIME.dt.dayofyear <= last_prod_doy).where(main_outputs.cum_growth > START_POP).where(mar.MSK > 50)
annual_g_sum = valid_growth.resample(TIME='1AS').sum(dim='TIME')
annual_g_sum = annual_g_sum.where(annual_g_sum > 0)
annual_g_max = valid_growth.resample(TIME='1AS').max(dim='TIME')

# %% trusted=true
## To look at all years together, uncomment these lines and run cell
# norm = colors.LogNorm(vmin=179, vmax=2.5e6)
# annual_g_sum.plot(col='TIME', col_wrap=4, norm=norm)

# %% trusted=true
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5,6))

cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

# Sums
norm = colors.LogNorm(vmin=179, vmax=2.5e6)
kws = dict(norm=norm, cmap=cmap, rasterized=True)
annual_g_sum.sel(TIME='2012').plot(ax=axes[0,0], **kws)
axes[0,0].set_title('2012 Sum')
annual_g_sum.sel(TIME='2017').plot(ax=axes[1,0], **kws)
axes[1,0].set_title('2017 Sum')

# Maxes
kws = dict(vmin=0, vmax=30000, cmap=cmap, rasterized=True)
annual_g_max.sel(TIME='2012').plot(ax=axes[0,1], **kws)
axes[0,1].set_title('2012 Max')
annual_g_max.sel(TIME='2017').plot(ax=axes[1,1], **kws)
axes[1,1].set_title('2017 Max')

sns.despine(left=True, bottom=True)

plt.savefig(os.path.join(RESULTS, 'fig_map_sum_max_2012_2017.pdf'), dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Sector-by-sector analysis
#
# Need to normalise by area. The max approach doesn't do this, because in areas like the SW with bigger blooms, the boxplots of max get 'depressed' - even though the sample size is much bigger.

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

# %%
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
# ### Extent of blooms / % coverage of ice sheet by blooms

# %% trusted=true
annual_biomass.plot.hist()

# %% trusted=true
annual_biomass.where(annual_biomass > 179).mean(dim='TIME').mean().compute()

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

# %% [markdown]
# ### Comparison with measured biomass population

# %% [markdown]
# #### Codify all available measurements 
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
    d = pd.read_csv(csv, parse_dates=True)
    csv_name = csv.split('/')[-1][:-4]
    meas[csv_name] = d
    
# Load Stibal separately - directly from their Excel Workbook
meas['stibal_2017'] = pd.read_excel(os.path.join(MEAS_BIO, 'grl56634-sup-0002-2017gl075958_data_si.xlsx'), sheet_name='algal cells time series data')

# %% trusted=true
## South Greenland

# chev provides date. Average all the sites on each day.
bmc = 'ia.cells.ml'
chev = meas['chevrollier_2022_2021_counts_south_gris'].groupby('date').mean()
chev = chev.reset_index()
chev['date'] = pd.to_datetime(chev['date'], dayfirst=True)
chev = chev.rename({bmc:RENAME_TO}, axis='columns')
chev['geom'] = pts_wgs84.loc['South'].geometry
chev['site'] = 'South'
chev['d_id'] = 'chev'
chev['biomass_col'] = bmc

# %% trusted=true
## South East Greenland 

# Halbach dataset
bmc = 'av.cells.ml'
halb = meas['halbach_2019_counts_se_greenland']
# Taken from Halbach et al. (2022) Table S1
halb_meta = {
    'hei':  {'geom': Point((dms_to_dd(-38,26,50), dms_to_dd(65,59,35))), 'date':dt.datetime(2019, 7, 26)},
    'mit1': {'geom': Point((dms_to_dd(-37,50, 2), dms_to_dd(65,41,39))), 'date':dt.datetime(2019, 7, 24)},
    'mit3': {'geom': Point((dms_to_dd(-37,50,25), dms_to_dd(65,41,38))), 'date':dt.datetime(2019, 7, 24)}
}
halb = halb.rename({bmc:RENAME_TO}, axis='columns')
halb = pd.merge(left=halb, right=pd.DataFrame(halb_meta).T, left_on='site', right_index=True)
halb['d_id'] = 'halb'
halb['biomass_col'] = bmc

# Lutz Mittivakkat dataset does not provide exact dates, so force the mid-date of their campaign (6-23 July 2012)
# All values were acquired within a 1 km2 area, so also okay to force a single coordinate.
bmc = 'cells.ml'
lutz = meas['lutz_2012_se_greenland']
lutz = lutz.rename({'grey_ice_samples':'site', bmc:RENAME_TO}, axis='columns')
lutz['date'] = dt.datetime(2012, 7, 6+(23-6))
# Estimate the centre point of Lutz et al. (2014) Figure 1.
lutz['geom'] = Point((-37.87, 65.685))
lutz['d_id'] = 'lutz'
lutz['site'] = 'mit'
lutz['biomass_col'] = bmc

# %%

# %% trusted=true
halb_meta

# %% trusted=true
## S6

# stib provides date. Average all the samples on each day.
# All Stibal values were acquired at S6 (see paper).
bmc = 'cells/ml'
stib = meas['stibal_2017'].filter(items=['doy 2014', bmc], axis='columns').dropna()
stib['date'] = [dt.datetime.strptime(f'2014-{int(d)}', '%Y-%j') for d in stib['doy 2014']]
stib = stib.groupby('date').mean().reset_index()
stib['geom'] = pts_wgs84.loc['S6'].geometry
stib = stib.rename({bmc:RENAME_TO}, axis='columns')
stib = stib.drop(labels=['doy 2014'], axis='columns')
stib['site'] = 'S6'
stib['biomass_col'] = bmc
stib['d_id'] = 'stib'


# williamson 2016 @ S6 provides date
bmc = 'overall.cells.per.ml'
w16s6 = meas['williamson_2016_count_biomass_all_2016'].filter(items=['date', 'habitat', bmc], axis='columns').dropna()
w16s6['date'] = [dt.datetime.strptime(d, '%d.%m.%y') for d in w16s6['date']]

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

w16s6 = w16s6.groupby('date').mean().reset_index()
w16s6['geom'] = pts_wgs84.loc['S6'].geometry
w16s6 = w16s6.rename({bmc:RENAME_TO}, axis='columns')
w16s6['d_id'] = 'w16_s6'
w16s6['site'] = 'S6'
w16s6['biomass_col'] = bmc

# %% trusted=true
stib

# %% trusted=true
stib

# %% trusted=true
## K-Transect

# Williamson space for time 2016.
# !!! check correct column is being used!
bmc = 'my.count'
w16k = meas['Williamson_dash_2016_space_for_time_counts'].filter(items=['date', 'habitat', bmc], axis='columns').dropna()
w16k['date'] = [dt.datetime.strptime(d, '%d.%m.%y') for d in w16k['date']]
# Calculate averages for each site on each date
w16k = w16k.groupby(['habitat', 'date'])[bmc].mean().reset_index()

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
## Kanger margin

# Based on Williamson et al. 2021 Frontiers: collected 8-10 August 2019, at Point 660.
bmc = 'total_cells_ml'
w21k = meas['Williamson_2021_sw_2019_algal_count'].filter(items=[bmc], axis='columns')
w21k = w21k.mean()
w21k['date'] = dt.datetime(2019, 8, 9)
# Google Maps approx coordinate a few hundred metres inland of Pt 660
w21k['geom'] = Point((-50.027204, 67.155493))
w21k['d_id'] = 'w19_660'
w21k = w21k.rename({bmc:RENAME_TO})
w21k['site'] = 'P660'
w21k['biomass_col'] = bmc
w21k = pd.DataFrame(w21k).T

# %% trusted=true
## Upernavik

# We already have coordinate. Check units? 
bmc = 'cells.per.ml'
upe = meas['williamson_2018_upe_u_cell_counts'].filter(items=[bmc], axis='columns')
upe = upe.mean()
upe['date'] = dt.datetime(2018, 7, 26)
upe['geom'] = pts_wgs84.loc['UPE'].geometry
upe = upe.rename({bmc:RENAME_TO})
upe['site'] = 'UPE'
upe['biomass_col'] = bmc
upe['d_id'] = 'upe'
upe = pd.DataFrame(upe).T

# %% trusted=true
## Join all datasets together

measurements = pd.concat([
#    chev, # removed - see note(1)
    halb,
    lutz,
    stib,
    w16s6,
    w16k,
#    w21k, # removed - see note(2)
    upe
], axis=0).reset_index()

# %% [markdown]
# *Note(1)* Chev. 2022, page 3: "The targeted surfaces were chosen to be roughly homogeneous on a wider surface in order to upscale the results for  1m2 areas, but are not representative of wider surfaces".
#
# *Note(2)* 2024-02-09: cells/ml in spreadsheet are one order of magnitude higher than all other datasets, suspect
#

# %% trusted=true
# Clean the joined dataset
measurements = measurements.drop(labels=['index'], axis='columns')
measurements['biomass'] = pd.to_numeric(measurements['biomass'])

# Convert to Polar Stereo to match MAR
measurements = gpd.GeoDataFrame(measurements, geometry='geom', crs='epsg:4236')
measurements = measurements.to_crs(3413)

# %% trusted=true
measurements

# %% trusted=true
measurements.to_file(os.path.join(RESULTS,'measurements_merged.gpkg'))

# %% trusted=true
#measurements.to_excel(os.path.join(RESULTS, 'measurements_merged.xlsx'))

# %% [markdown]
# #### Compare measurements with modelled blooms

# %% trusted=true
#measurements = pd.read_excel(os.path.join(RESULTS, 'measurements_merged.xlsx'), index_col=0)

# %% trusted=true
store = []
for ix, row in measurements.iterrows():
    v = main_outputs.cum_growth.sel(TIME=row['date']).sel(x=row['geom'].x, y=row['geom'].y, method='nearest')
    store.append(float(v.values))

# %% trusted=true
measurements['modelled_biomass_dw'] = store

# Convert measured values to DW
measurements['biomass_dw'] = ww_to_dw(measurements['biomass'])

# %% trusted=true
# %matplotlib inline
sns.scatterplot(
    x=measurements['biomass_dw'], 
    y=measurements['modelled_biomass_dw'],
    style=measurements['site'],
    hue=measurements['d_id'],
    s=50
)
plt.grid()
plt.xlim(0, 22000)
plt.ylim(0, 22000)
plt.xlabel(r'Measured biomass ng DW ml (wet$\times$0.84)')
plt.ylabel('Modelled biomass ng DW ml')
plt.legend(loc=(1.01,0))
plt.title('Means of measured biomass, except Medians for Stibal')
sns.despine()

# %% trusted=true
measurements[(measurements.site == 'S6') & (measurements.biomass_dw > 20000)]

# %% trusted=true
# %matplotlib widget
fig, ax = plt.subplots()
mar.MSK.plot(ax=ax)
m = measurements[measurements.site == 'mit3']
plt.plot(m.geom.x, m.geom.y, 'xr')

# %% [markdown]
# ### Is there a trend in annual bloom extent?

# %% trusted=true
((annual_biomass > 179).sum(dim=('x','y')) * 10**2).plot(marker='o')
plt.ylabel('km2')
plt.grid()

# %% trusted=true

y = as_perc.to_pandas()
X = sm.add_constant(np.arange(0, len(y)))
m = sm.OLS(y, X)
r = m.fit()
print(r.summary())

# %% trusted=true
150*270

# %% trusted=true
annual_biomass = main_outputs.cum_growth.where(main_outputs.TIME.dt.dayofyear <= last_prod_doy).where(main_outputs.cum_growth > 179).where(mar.MSK > 50).resample(TIME='1AS').sum(dim='TIME')
anuual_area = annual_biomass.where(annual_biomass > 179).count(dim=('x','y'))

# %% trusted=true
anuual_area.plot()

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

# %%
