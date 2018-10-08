import numpy as np
import GMdataImportFx as gix
import GMCluster as gmc
import os
import csv
import prepDx as pdx

#Load rupture labels
rupt_dict = gmc.loadLabelDict('label_dict.csv')
rupt_lbl = list(rupt_dict.keys())

#Load cybershake station file
station_CS = pdx.loadCS_stationList('CS186_stationList.csv')

#Initialize dictionary of used stations
station_virt_dict = {}

#Folder and file names
dir_path = './data/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
#Initialize list of unfound faults
unfoundFault = []

for i in range(len(rupt_lbl)):
    #Load current GM
    str_GM = './raw_data/' + rupt_lbl[i] + '.csv'
    try:
        GM = pdx.loadGM_CS(str_GM)
        
        #Remove all non-geometric mean lines
        for j in range(len(GM)-1, 0, -1):
            if not GM[j][1] == 'geom':
                del GM[j]

        #Add station location
        GM_save = [[GM[0][0], 'easting', 'northing']]
        for j in range(len(GM[0])-2):
            GM_save[0].append(GM[0][j+2])
            
        for j in range(1, len(GM)):
            GM_save.append([GM[j][0], station_CS[GM[j][0]][0], station_CS[GM[j][0]][1]])
            for k in range(len(GM[j])-2):
                GM_save[j].append(GM[j][k+2])
                
            #Test if station already in dict
            try:
                a = station_virt_dict[GM[j][0]]
            except:
                station_virt_dict[GM[j][0]] = station_CS[GM[j][0]]
            
        #Save new file
        str_GM = './data/' + rupt_lbl[i] + '.csv'
        with open(str_GM, "w") as f:
            writer = csv.writer(f)
            writer.writerows(GM_save)
        f.close()
    except:
        unfoundFault.append([rupt_lbl[i]])
        
#Save unfound faults list
with open('./unfoundRuptures.csv', "w") as f:
    writer = csv.writer(f)
    writer.writerows(unfoundFault)
f.close()

#Save dictionary of virtual used stations
pdx.saveVirtStationDict(station_virt_dict)