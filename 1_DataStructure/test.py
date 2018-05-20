import GMdataImportFx as gix
import numpy as np
import os
import matplotlib.pyplot as plt


#Set grid cell size
cell_size = 2000. #meters
grid_size = [450000./cell_size, 1350000./cell_size]

#Import data
#North island
NI_raw = gix.loadCSV('./data/raw/NI.csv', isInput=False, row_ignore=1, col_ignore=0)

#South island
SI_raw = gix.loadCSV('./data/raw/SI.csv', isInput=False, row_ignore=1, col_ignore=0)

data = []
for i in range(len(NI_raw)):
    data.append(NI_raw[i])

for i in range(len(SI_raw)):
    data.append(SI_raw[i])
    
del SI_raw, NI_raw
  
#Convert station location
for i in range(len(data)):
    data[i][1:3] = gix.convertWGS2NZTM2000(data[i][1], data[i][2])

#Rotate locations
#Christchurch cathedral
translation = [-1250000., -4650000.]
loc_chch = gix.convertWGS2NZTM2000(172.6376, -43.5309)
for i in range(len(data)):
    data[i][1:3] = gix.rotateCoordinate(loc_chch, data[i][1:3], 35.0)
    data[i][1:3] = gix.translateCoordinate(translation, data[i][1:3])

#Collect stations and rupture labels
data_ar = np.asarray(data).T
station_str = data_ar[0].tolist()
label_str = data_ar[8].tolist()
station_loc_x = data_ar[1].tolist()
station_loc_y = data_ar[2].tolist()
station_loc_x = [float(x) for x in station_loc_x]
station_loc_y = [float(x) for x in station_loc_y]

#Identify unique ruptures
label_u = gix.getUniqueLabel(label_str)

#Identify unique station
station = gix.findStationLocation(station_str, station_loc_x, station_loc_y)

#Map station to grid loc
for i in range(len(station)):
    station[i][3] = int(station[i][1]/cell_size)
    station[i][4] = int(station[i][2]/cell_size)

#Create result folder
folder_path = './gen'
try:
    os.makedirs(folder_path)
except:
    print('Folder already exists')

#Create rupture files containing actiated stations and labels
for i in range(len(label_u)):
    #Get ID
    ID_i = gix.getID(label_u[i], label_str)
    
    #Gather data
    data_i = []
    for j in range(len(ID_i)):
        data_i.append(data[ID_i[j]])
        
    #Map stations to grid in final data
    data_i = gix.mapStations(data_i, cell_size)
    
    #Create file
    gix.saveGMRecord(data_i, label_u[i], grid_size, cell_size)

''' 
station = np.asarray(station).T
X = station[3]
Y = station[4]

plt.figure()
plt.scatter(X, Y)
'''
    
    
    
    
    
    
    


