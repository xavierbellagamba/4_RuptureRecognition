import numpy as np
import GMdataImportFx as gix
import GMCluster as gmc
import os
import csv
import prepDx as pdx
import matplotlib.pyplot as plt
import glob

#Name of IM to save
IM_name = ['PGA', 'PGV', 'AI', 'pSA_0.5', 'pSA_1.0', 'pSA_3.0']

#Set grid cell size
cell_size = 20000. #meters

#Load station list
station = pdx.loadVirtStationDict('virt_station_dict.csv')
data = np.asarray(list(station.values()))

#Rotate locations
#Christchurch cathedral
translation = [-1200000., -4570000.]
loc_chch = gix.convertWGS2NZTM2000(172.6376, -43.5309)
for i in range(len(data)):
    data[i] = gix.rotateCoordinate(loc_chch, data[i], 25.0)
    data[i] = gix.translateCoordinate(translation, data[i])
    
#Load label dict
lbl_dict = gmc.loadLabelDict('./label_dict.csv')

#Map station to grid loc
station_rot_dict = {}
station_name = list(station.keys())
for i in range(len(station_name)):
    station_rot_dict[station_name[i]] = [int(data[i][0]/cell_size), int(data[i][1]/cell_size)]

#Remove double loc on grid
dict_exch = {tuple(v): k for k, v in station_rot_dict.items()}
station_rot_dict2 = {v: list(k) for k, v in dict_exch.items()}

#Get grid size
x_grid_max = np.max(np.asarray(list(station_rot_dict2.values()))[:, 0])
y_grid_max = np.max(np.asarray(list(station_rot_dict2.values()))[:, 1])

#Get number of map to treat
file_name = glob.glob('./data/*.csv')
n_file = len(file_name)

#Initialize array
GM = np.ones((n_file, x_grid_max, y_grid_max, len(IM_name)))
for i in range(len(IM_name)):
    if IM_name[i] == 'PGV':
        GM[:, :, :, i] = GM[:, :, :, i]*(0.25)
    elif IM_name[i] == 'AI':
        GM[:, :, :, i] = GM[:, :, :, i]*(-2.)
    elif IM_name[i] == 'pSA_3.0':
        GM[:, :, :, i] = GM[:, :, :, i]*(-6.)
    else:
        GM[:, :, :, i] = GM[:, :, :, i]*(-5.)
GM_name = []
GM_lbl = []

#Get IM position given names
lbl_IM = pdx.loadGM_CS(file_name[i])[0]
IM_pos = []
for i in range(len(lbl_IM)):
    for j in range(len(IM_name)):
        if IM_name[j] == lbl_IM[i]:
            IM_pos.append(i)
            break

#Treat map one after another (memory efficiency)
unfoundStation = []
for i in range(1):#n_file):
    GM_i = pdx.loadGM_CS(file_name[i])
    for j in range(1, len(GM_i)):
        for k in range(len(IM_pos)):
            try:
                GM[i, station_rot_dict2[GM_i[j][0]][0], station_rot_dict2[GM_i[j][0]][1], k] = np.log(GM_i[j][IM_pos[k]])
            except: 
                unfoundStation.append(GM_i[j][0])

    GM_name.append(file_name[i][7:-4])
    GM_lbl.append(lbl_dict[file_name[i][7:-4]])
    
#Folder and file names
dir_path = './gen/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
#Save results
np.save(dir_path + 'X.npy', GM)
np.save(dir_path + 'y.npy', np.asarray(GM_lbl))
np.save(dir_path + 'X_name.npy', np.asarray(GM_name))


st = np.asarray(list(station_rot_dict2.values()))
plt.scatter(st[:, 0], st[:, 1])

