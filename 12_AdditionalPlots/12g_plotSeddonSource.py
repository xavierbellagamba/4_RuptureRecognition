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
import glob
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def cm2inch(value):
    return value/2.54

#########################################################################
#########################################################################
# User-defined paramters
#########################################################################

#########################################################################
#########################################################################

#Load list of selected faults
fault_select = gix.loadCSV('./selectedFault_list.csv')

#Find the appropriate fault geometry
file_path = glob.glob1('./NZ_Faults_NZTM2000/', '*.csv')
fault_geometry = []
color_f = []
for i in range(len(file_path)):
    #Check if fault is selected
    isSelected = False
    for j in range(len(fault_select)):
        if fault_select[j][0] + '.' in file_path[i]:
            isSelected = True
            break
    if isSelected:
        fault_geometry_i = gix.loadCSV('./NZ_Faults_NZTM2000/' + file_path[i])
        f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in fault_geometry_i]
        fault_geometry.append(np.array(f_geo).astype(float))

found_s = gix.loadCSV('./NZ_Faults_NZTM2000/MS04.csv')
f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in found_s]
found = np.array(f_geo).astype(float)

real_s = gix.loadCSV('./NZ_Faults_NZTM2000/AwatNEVer.csv')
f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in real_s]
real = np.array(f_geo).astype(float)

real_s = gix.loadCSV('./NZ_Faults_NZTM2000/KekNeed.csv')
f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in real_s]
real2 = np.array(f_geo).astype(float)

#stations = pd.read_csv('./data/Kaikoura_obs.csv', header=None)
#stations_init = pd.read_csv('./data/Kaikoura_init.csv', header=None)
#station_dict = gix.loadStationDict('./station_dict.csv')

#st_pos = []
#st_pos_init = []
#for i in range(len(stations[0])):
#    st_pos.append(station_dict[stations[0][i]])
#for i in range(len(stations_init[0])):
#    st_pos_init.append(station_dict[stations_init[0][i]])
#st_pos = np.asarray(st_pos)
#st_pos_init = np.asarray(st_pos_init)


#Load planes
#pln = fu.loadPlaneGeometry(['kaikouraFault'])

#Load hypocenter
hypo = gix.loadCSV('./data/seddon_hypo.txt')
hypo = gix.convertWGS2NZTM2000(hypo[0][0], hypo[0][1])

#Load DEM
NZI = './NZI/NZI.tif'
raster = rio.open(NZI)
trf = raster.transform
NZI = raster.read()
ext = [trf[2]-50, trf[2]+50+NZI.shape[2]*100, trf[5]-50-NZI.shape[1]*100, trf[5]+50]
NZI = NZI[0]

#Plot
cmap = plt.cm.get_cmap('bwr')
cmap_NZI = plt.cm.get_cmap('Greys_r')

rc('text', usetex=True)
rc('font', family='serif')

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(10), cm2inch(14))
ax.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)
hyp = ax.scatter(hypo[0], hypo[1], zorder=4, s=120, marker='*', edgecolors='b', facecolors='b', label='Hypocenter')
#ax.scatter(st_pos[:, 0], st_pos[:, 1], s=12, color='red', marker='^', zorder=4)
#ax.scatter(st_pos_init[:, 0], st_pos_init[:, 1], s=12, color='green', marker='^', zorder=5)
for i in range(len(fault_geometry)):
    ax.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color=[0.6, 0.6, 0.6], linewidth=1.25, zorder=2)
fd = ax.plot(found[:, 0], found[:, 1], color='red', linewidth=1.25, zorder=3, label='ML-identified source')
cl = ax.plot(real[:, 0], real[:, 1], color='green', linewidth=1.25, zorder=3, label='Closest to hypocenter')
ax.plot(real2[:, 0], real2[:, 1], color='green', linewidth=1.25, zorder=3)
ax.plot(fault_geometry[0][:, 0], fault_geometry[0][:, 1], color=[0.6, 0.6, 0.6], linewidth=1.25, zorder=0, label='Considered sources')

#    else: 
#        ax.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color='red', linewidth=1.25, zorder=2)
#
#ax.plot(fault_geometry[1][:, 0], fault_geometry[1][:, 1], color='red', linewidth=1.25, zorder=0, label='Selected')
#ax.plot(fault_geometry[0][:, 0], fault_geometry[0][:, 1], color=[0.4, 0.4, 0.4], linewidth=1.25, zorder=0, label='Discarded')
#for j in range(int(len(pln[0])/4)):
#    ax.fill(pln[0][j*4:j*4+4, 0], pln[0][j*4:j*4+4, 1], zorder=3, alpha=0.6, c='black', linewidth=0.0)
ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
#ax.legend(title='Faults', loc=2)
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])

axins = zoomed_inset_axes(ax, 3., loc=2)
#pln = np.asarray(pln)
axins.set_xlim([1570887.9935, 1814185.9164999998])
axins.set_ylim([5265093.467, 5459948.853])
for i in range(len(fault_geometry)):
    axins.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color=[0.6, 0.6, 0.6], linewidth=1.25, zorder=2)
h = axins.scatter(hypo[0], hypo[1], s=180, zorder=5, marker='*', edgecolors='b', facecolors='b')

axins.plot(found[:, 0], found[:, 1], color='red', linewidth=1.25, zorder=3)
axins.plot(real[:, 0], real[:, 1], color='green', linewidth=1.25, zorder=3)
axins.plot(real2[:, 0], real2[:, 1], color='green', linewidth=1.25, zorder=3)

axins.grid()
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_yticks([])
axins.set_xticks([])

axins.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)

#st1 = axins.scatter(st_pos[:, 0], st_pos[:, 1], s=40, color='red', marker='^', zorder=4)
#st2 = axins.scatter(st_pos_init[:, 0], st_pos_init[:, 1], s=40, color='green', marker='^', zorder=5)

#    axins.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color=col, linewidth=1.25, zorder=5)
#for j in range(int(len(pln[0])/4)):
#    axins.fill(pln[0][j*4:j*4+4, 0], pln[0][j*4:j*4+4, 1], zorder=3, alpha=0.6, c='black', linewidth=0.0)

mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0.25", zorder=4)

rup = mpatches.Patch(color='black', alpha=0.6)
fa = mlines.Line2D([], [], color=[0.6, 0.6, 0.6])

ax.legend(loc=4)#[hyp, fd, cl], ['Hypocenter', 'ML-identified source', 'Closest faults to hypocenter'], loc=4)

plt.tight_layout()

plt.savefig('./Seddon_source.pdf', dpi=600)



