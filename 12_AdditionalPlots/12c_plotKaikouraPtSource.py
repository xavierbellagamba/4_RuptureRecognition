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


#Load list of selected faults
kaikoura = gix.loadCSV('./data/shakemapv2.csv')
GM_raster = np.array(kaikoura)
GM_raster = GM_raster[np.where(GM_raster[:, 2]>2.5)]


IM_name = 'PGV'
b_dict = {}
str_lbl = {}
b_dict['PGA'] = [0.025, 1.25]
str_lbl['PGA'] = 'PGA [$g$]'
str_lbl['PGV'] = 'PGV [$cm/s$]'
b_dict['PGV'] = [2.5, 80.]
str_file = 'shakemap_PGV.pdf'
vmin = 0.0

#Load DEM
NZI = './NZI/NZI.tif'
raster = rio.open(NZI)
trf = raster.transform
NZI = raster.read()
ext = [trf[2]-50, trf[2]+50+NZI.shape[2]*100, trf[5]-50-NZI.shape[1]*100, trf[5]+50]
NZI = NZI[0]

#Plot
cmap = plt.cm.get_cmap('hot_r')
cmap_NZI = plt.cm.get_cmap('Greys_r')

rc('text', usetex=True)
rc('font', family='serif')

im = plt.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], cmap=cmap, vmin=0.0, vmax=80.0, zorder=0)

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(10), cm2inch(14))
ax.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)

#ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=44, marker='s', cmap=cmap, alpha=0.05, zorder=2, vmin=0.0, vmax=80.0, edgecolors=None, linewidths=0.0)
#ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=11, marker='s', cmap=cmap, alpha=0.1, zorder=2, vmin=0.0, vmax=80.0, edgecolors=None, linewidths=0.0)


if not 'res' in str_file.split('_')[-1]:
    ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=144, marker=(4, 0, 20), cmap=cmap, alpha=0.1, zorder=2, vmin=vmin, vmax=b_dict[IM_name][1], edgecolors=None, linewidths=0.0)
ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=36, marker=(4, 0, 20), cmap=cmap, alpha=0.2, zorder=2, vmin=vmin, vmax=b_dict[IM_name][1], edgecolors=None, linewidths=0.0)

ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])

divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='3%', pad=0.35)
cb = plt.colorbar(im, orientation='horizontal', cax=cax, ticks=np.arange(0, 80.00001, 80./5.))
cb.ax.set_xlabel('PGV, [$cm/s$]')

plt.tight_layout()

plt.savefig('./kaikoura_pt.pdf', dpi=600)


