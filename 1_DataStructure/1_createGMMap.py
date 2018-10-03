import GMdataImportFx as gix
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

###########################################
# User-defined parameters
###########################################
#Path to the main data file
data_path = './data/DB_IM.csv'
###########################################

#Import data
print('Load data...')
data = gix.loadCSV(data_path, isInput=False, row_ignore=1, col_ignore=0)

#Convert station location
print('Convert station coordinate system...')
for i in range(len(data)):
    data[i][1:3] = gix.convertWGS2NZTM2000(data[i][1], data[i][2])
    del data[i][3]

#Collect stations and rupture labels
data_ar = np.asarray(data).T
station_str = data_ar[0].tolist()
label_str = data_ar[25].tolist()
station_loc_x = data_ar[1].tolist()
station_loc_y = data_ar[2].tolist()
station_loc_x = [float(x) for x in station_loc_x]
station_loc_y = [float(x) for x in station_loc_y]

#Identify unique ruptures
print('Get names of unique ruptures...')
label_u = gix.getUniqueLabel(label_str)

#Identify unique station
print('Identify unique stations...')
station = gix.findStationLocation(station_str, station_loc_x, station_loc_y)

#Create result folder
folder_path = './gen'
try:
    os.makedirs(folder_path)
except:
    print('Folder already exists')

#Create rupture files containing actiated stations and labels
for i in range(len(label_u)):
    print(str(i)+'/'+str(len(label_u))+': '+label_u[i])
    #Get ID
    ID_i = gix.getID(label_u[i], label_str)
    
    #Gather data
    data_i = []
    for j in range(len(ID_i)):
        data_i.append(data[ID_i[j]])
        del data_i[-1][-1]
        
    #Save GM record
    file_path = './gen/' + label_u[i] + '.csv'
    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_i)
    f.close()
    
#Save dictionnary of stations
gix.saveStationDict(station)
    
#Create IM dictionary
IM_dict = {}
data = gix.loadCSV('./data/DB_test.csv', isInput=False, row_ignore=0, col_ignore=0)
data = data[0][4:-1]
for i in range(len(data)):
    IM_dict[data[i]] = i

#Save dictionnary of IM
gix.saveIMDict(IM_dict)
