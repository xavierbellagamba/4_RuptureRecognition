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

#Load waveforms
s = 800000.
wf0 = gix.loadCSV('./data/wf1.txt')
wf0 = np.array(wf0)
wf0 = wf0[:, 3:].astype(float)*s

wf1 = gix.loadCSV('./data/wf2.txt')
wf1 = np.array(wf1)
wf1 = wf1[:, 3:].astype(float)*s

wf2 = gix.loadCSV('./data/wf3.txt')
wf2 = np.array(wf2)
wf2 = wf2[:, 3:].astype(float)*s

wf3 = gix.loadCSV('./data/wf4.txt')
wf3 = np.array(wf3)
wf3 = wf3[:, 3:].astype(float)*s

wf4 = gix.loadCSV('./data/wf5.txt')
wf4 = np.array(wf4)
wf4 = wf4[:, 3:].astype(float)*s

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
ax.scatter(station[:, 0], station[:, 1], marker='^', s=12, c='black', zorder=2, facecolors=None)
ax.plot(np.arange(0, len(wf4)*10, 10)+1184000, wf4[:, 0]+5026400, c='red', linewidth=.75)
ax.plot(np.arange(0, len(wf3)*10, 10)+1282500, wf3[:, 0]+5133000, c='red', linewidth=.75)
ax.plot(np.arange(0, len(wf2)*10, 10)+1124000, wf2[:, 0]+4928500, c='red', linewidth=.75)
ax.plot(np.arange(0, len(wf1)*10, 10)+1435000, wf1[:, 0]+5270000, c='red', linewidth=.75)
ax.plot(np.arange(0, len(wf0)*10, 10)+1304000, wf0[:, 0]+4839000, c='red', linewidth=.75)
ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])
plt.tight_layout()

plt.savefig('./stations_wf.pdf', dpi=600)



