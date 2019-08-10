import numpy as np
from operator import itemgetter
import GMdataImportFx as gix
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import glob


def cm2inch(value):
    return value/2.54

#########################################################################
#########################################################################
# User-defined paramters
#########################################################################

#########################################################################
#########################################################################

#Load list of stations
station = gix.loadStationDict('./station_dict.csv')
station = np.array(list(station.values()))

#Load DEM
NZI = './NZI/NZI.tif'
raster = rio.open(NZI)
trf = raster.transform
NZI = raster.read()
ext = [trf[0]-50, trf[0]+50+NZI.shape[2]*100, trf[3]-50-NZI.shape[1]*100, trf[3]+50]
NZI = NZI[0]

#Plot
cmap = plt.cm.get_cmap('bwr')
cmap_NZI = plt.cm.get_cmap('Greys_r')

rc('text', usetex=True)
rc('font', family='serif')

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(10), cm2inch(14))
ax.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)
ax.scatter(station[:, 0], station[:, 1], marker='^', s=12, c='black', zorder=2)
ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])
plt.tight_layout()

plt.savefig('./stations.pdf', dpi=600)



