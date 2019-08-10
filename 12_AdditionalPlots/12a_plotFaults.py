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
fault_select = gix.loadCSV('./selectedFault_list.csv')

#Find the appropriate fault geometry
file_path = glob.glob1('./NZ_Faults_NZTM2000/', '*.csv')
fault_geometry = []
color_f = []
for i in range(len(file_path)):
    fault_geometry_i = gix.loadCSV('./NZ_Faults_NZTM2000/' + file_path[i])
    f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in fault_geometry_i]
    fault_geometry.append(np.array(f_geo).astype(float))
    #Check if fault is selected
    isSelected = False
    for j in range(len(fault_select)):
        if fault_select[j][0] + '.' in file_path[i]:
            isSelected = True
            break
    if isSelected:
        color_f.append(0)
    else:
        color_f.append(1)

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
for i in range(len(fault_geometry)):
    if color_f[i] == 1:
        ax.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color=[0.4, 0.4, 0.4], linewidth=1.25, zorder=2)
    else: 
        ax.plot(fault_geometry[i][:, 0], fault_geometry[i][:, 1], color='red', linewidth=1.25, zorder=2)

ax.plot(fault_geometry[1][:, 0], fault_geometry[1][:, 1], color='red', linewidth=1.25, zorder=0, label='Selected')
ax.plot(fault_geometry[0][:, 0], fault_geometry[0][:, 1], color=[0.4, 0.4, 0.4], linewidth=1.25, zorder=0, label='Discarded')

ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
ax.legend(title='Faults', loc=2)
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])
plt.tight_layout()

plt.savefig('./faults.pdf', dpi=600)



