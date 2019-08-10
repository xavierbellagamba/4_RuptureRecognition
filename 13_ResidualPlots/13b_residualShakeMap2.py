import numpy as np
from operator import itemgetter
import GMdataImportFx as gix
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
import matplotlib
import fwk_utils as fu
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import glob


def cm2inch(value):
    return value/2.54

#########################################################################
#########################################################################
# User-defined paramters
#########################################################################

#########################################################################
#########################################################################

#Load PGA
GM_raster = gix.loadCSV('./data/shakemapv2.csv')
GM_raster = np.asarray(GM_raster)

PGV = gix.loadCSV('./data/PGV_obs.csv')
PGV = np.asarray(PGV)

base = gix.loadCSV('./data/base.csv')
base = np.asarray(base)

orig = np.copy(GM_raster)
#ind = np.where(PGV[:, 2] > 5)
#orig = orig[ind, :]

res_h = []
for i in range(len(orig[:, 0])):
    for j in range(len(base[:, 0])):
        dist = np.sqrt((base[j, 0] - orig[i, 0])**2. + (base[j, 1] - orig[i, 1])**2.)
        if dist < 1000.0:
            for k in range(len(PGV[:, 0])):
                dist = np.sqrt((PGV[k, 0] - orig[i, 0])**2. + (PGV[k, 1] - orig[i, 1])**2.)
                if dist < 1000. and PGV[k, 2] > 5:
                    res_h.append(np.log(orig[i, 2]) - np.log(PGV[k, 2]))
                    break

res_h = np.asarray(res_h)
'''
IM_name = 'PGV'
b_dict = {}
str_lbl = {}
b_dict['PGA'] = [-1., 1.]
b_dict['PGV'] = [-1., 1.]
vmin = -1.0
alpha_b = 0.4
str_lbl['PGA'] = 'ln(PGA$_{Generated}$/PGA$_{Data}$)'
str_lbl['PGV'] = 'ln(PGV$_{Generated}$/PGV$_{Data}$)'

str_file = 'onshore_obs_PGV.pdf'

#Load DEM
NZI = './NZI/NZI.tif'
raster = rio.open(NZI)
trf = raster.transform
NZI = raster.read()
ext = [trf[2]-50, trf[2]+50+NZI.shape[2]*100, trf[5]-50-NZI.shape[1]*100, trf[5]+50]
NZI = NZI[0]

#Plot
cmap = plt.cm.get_cmap('seismic')
cmap_NZI = plt.cm.get_cmap('Greys_r')

rc('text', usetex=True)
rc('font', family='serif')

im = plt.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], cmap=cmap, vmin=vmin, vmax=b_dict[IM_name][1])

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(10), cm2inch(14))
ax.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)
#ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=36, marker=(4, 0, 20), cmap=cmap, alpha=0.6, zorder=2, vmin=vmin, vmax=b_dict[IM_name][1], edgecolors=None, linewidths=0.0)
ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], s=36, marker=(4, 0, 20), cmap=cmap, alpha=0.2, zorder=2, vmin=vmin, vmax=b_dict[IM_name][1], edgecolors=None, linewidths=0.0)
#ax.scatter(GM_raster[:, 0], GM_raster[:, 1], c=GM_raster[:, 2], cmap=cmap, alpha=0.5, zorder=2, vmin=vmin, vmax=b_dict[IM_name][1], edgecolors=None, linewidths=0.0)
plt.xlim([ext[0], ext[1]])
plt.ylim([ext[2], ext[3]])
plt.xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
plt.yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
divider = make_axes_locatable(ax)
cax = divider.append_axes('bottom', size='3%', pad=0.35)
cb = plt.colorbar(im, orientation='horizontal', cax=cax, ticks=np.arange(vmin, b_dict[IM_name][1]+0.00001, b_dict[IM_name][1]/5.))
cb.ax.set_xlabel(str_lbl[IM_name])


iax_p = plt.axes([0, 0, 0, 0])
ip_p = InsetPosition(ax, [0.05, 0.66, 0.35, 0.27]) #posx, posy, width, height
iax_p.set_axes_locator(ip_p)
iax_p.hist(res_h, color='grey', bins=8)
iax_p.spines['right'].set_visible(False)
iax_p.spines['top'].set_visible(False)
iax_p.spines['left'].set_visible(False)
iax_p.set_yticks([])
iax_p.set_xlim([-3, 3])
str_metric = '$\mu$ = ' + str(round(np.mean(res_h), 2)) + ', $\sigma$ = ' + str(round(np.std(res_h), 2))
ax.text(ext[0]+25000, ext[3]-50000, str_metric)

plt.tight_layout()
plt.savefig(str_file, dpi=600)
plt.close('all')


'''






