import numpy as np
from operator import itemgetter
import GMdataImportFx as gix
import GMCluster as gml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def cm2inch(value):
    return value/2.54

#########################################################################
#########################################################################
# User-defined paramters
#########################################################################
#Fault name
fault_name = 'ConwayOS'
#########################################################################
#########################################################################
#Load hypocentre locations
data = gix.loadCSV('./rupture_info.csv', row_ignore=1)
data = np.array(data)
lbl = data[:, 0]
data = data[:, [1, 2]].astype(float)

#Convert locations
loc = gix.convertWGS2NZTM2000(data[:, 1], data[:, 0])
loc = np.array(loc)

#Load label dict
lbl_dict = gml.loadLabelDict('./label_dict.csv')

#Select hypocentres from fault name
i_rupt = []
for i in range(len(lbl)):
    if fault_name + '_' in lbl[i]:
        i_rupt.append(i)
        
#Isolate ruptures
loc = loc[:, i_rupt]
lbl = lbl[i_rupt]

#Assign label to hypocentre (and color)
cl_ID = []
for i in range(len(lbl)):
    lbl[i] = lbl_dict[lbl[i]]
    cl_ID.append(int(str(lbl[i]).split('_')[1]))

#Find the appropriate fault geometry
file_path = './NZ_Faults_NZTM2000/' + fault_name + '.csv'
fault_geometry = gix.loadCSV(file_path)
f_geo = [[x[0].split(';')[0], x[0].split(';')[1]] for x in fault_geometry]
fault_geometry = np.array(f_geo).astype(float)

#Load fault 3D
file_path = './fault3D/' + fault_name + '.csv'
plane = gix.loadCSV(file_path)
plane = [gix.convertWGS2NZTM2000(x[0], x[1]) for x in plane]
plane = np.array(plane)

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
ax.scatter(loc[0, :], loc[1, :], c=cl_ID, s=5, cmap=cmap, zorder=10)
#ax.scatter(loc[0, :], loc[1, :], c=[0.1, 0.1, 0.1], s=5, zorder=3)#Poster
ax.plot(fault_geometry[:, 0], fault_geometry[:, 1], color='black', linewidth=1.5, zorder=5)
for i in range(int(len(plane)/4)):
    ax.fill(plane[i*4:i*4+4, 0], plane[i*4:i*4+4, 1], zorder=3, alpha=0.6, c='grey', linewidth=0.0)
cl_1 = ax.scatter(loc[0, 0], loc[1, 0], c='red', s=5, cmap=cmap, zorder=0, vmin=0, vmax=1)
cl_2 = ax.scatter(loc[0, 0], loc[1, 0], c='blue', s=5, cmap=cmap, zorder=0, vmin=0, vmax=1)
ax.set_xlim([ext[0], ext[1]])
ax.set_ylim([ext[2], ext[3]])

#axins = zoomed_inset_axes(ax, 5, loc=2)#Pahaua
axins = zoomed_inset_axes(ax, 6.5, loc=2)#ConwayOS
#axins = zoomed_inset_axes(ax, 2.5, loc=2)#AlpineF2K


for i in range(int(len(plane)/4)):
    axins.fill(plane[i*4:i*4+4, 0], plane[i*4:i*4+4, 1], zorder=3, alpha=0.6, c='grey', linewidth=0.0)

x1, x2, y1, y2 = 1570000, 1700000, 5280000, 5350000 #ConwayOS
#x1, x2, y1, y2 = 1770000, 1920000, 5355000, 5430000 #Pahaua
#x1, x2, y1, y2 = 1130000, 1420000, 5000000, 5210000 #AlpineF2K
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.grid()
axins.set_xticklabels([])
axins.set_yticklabels([])
axins.set_yticks([])
axins.set_xticks([])

axins.imshow(NZI, extent=ext, zorder=1, cmap=cmap_NZI)
axins.scatter(loc[0, :], loc[1, :], c=cl_ID, s=5, cmap=cmap, zorder=10)
axins.plot(fault_geometry[:, 0], fault_geometry[:, 1], color='black', linewidth=1.5, zorder=2)
#axins.scatter(loc[0, :], loc[1, :], c=[0.1, 0.1, 0.1], s=5, zorder=3)#Poster

mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0.25", zorder=4) #ConwayOS
#mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.25", zorder=4) #Pahaua
#mark_inset(ax, axins, loc1=4, loc2=3, fc="none", ec="0.25", zorder=4) #AlpineF2K




ax.set_xticks([1205923., 1600000., 1994077.])
ax.set_xticklabels(['168.0$^\circ$E', '173.0$^\circ$E', '178.0$^\circ$E'])
ax.set_yticks([5004874., 5560252., 6115515.])
ax.set_yticklabels(['45.0$^\circ$S', '40.0$^\circ$S', '35.0$^\circ$S'])
#ax.legend([], [], title=fault_name, loc=4)#Poster
ax.legend([cl_1, cl_2], ['Cluster 1', 'Cluster 2'], title=fault_name, loc=4)
plt.tight_layout()

plt.savefig('./hypo_' + fault_name + '.pdf', dpi=600)

