import GMdataImportFx as gix
import GMCluster as gcl
import numpy as np
import matplotlib.pyplot as plt
import os


IM_considered = ['PGA', 'PGV', 'pSA_0.1', 'pSA_1.0', 'pSA_3.0', 'Ds595', 'AI']

#Import IM, station and label dict
station_dict = gix.loadStationDict('station_dict.csv')
label_dict = gcl.loadLabelDict('label_dict.csv')
IM_dict = gix.loadIMDict('IM_dict.csv')

IM_ID = [IM_dict[IM_considered[i]] for i in range(len(IM_considered))]

#Import GM
print('Load data...')
data, GM_name = gcl.loadGM_clust('*')
n_rupture = len(data)

#Create result folder
folder_path = './gen'
try:
    os.makedirs(folder_path)
except:
    print('Folder already exists')

#Assign continuous ID to stations
n_station = len(station_dict)
station_name = list(station_dict.keys())
for i in range(n_station):
    station_dict[station_name[i]] = i

#Get the mean and standard dev of each IM
print('Compute mean and standard deviation of the considered IMs...')
IM_val = [[] for i in IM_considered]
for i in range(len(data)):
    for j in range(len(data[i])):
            for k in range(len(IM_considered)):
                IM_val[k].append(data[i][j][IM_ID[k]+3])

#Extract and log IM
IM_val = np.asarray(IM_val)
n_IM = len(IM_considered)
IM_val = np.log(IM_val)
'''
#Compute dataset wide mean and std in log space
mu_IM = np.mean(IM_val, axis=(1))
std_IM = np.std(IM_val, axis=(1))

#Save mu and std
np.save('./mu_IM.npy', mu_IM)
np.save('./std_IM.npy', std_IM)
'''
#Normalize IM
print('Normalize the data...')
data_n = []
for i in range(len(data)):
    data_n.append([])
    for j in range(len(data[i])):
        data_n[i].append([])
        for k in range(len(IM_considered)):
            #data_n[i][j].append((np.log(data[i][j][IM_ID[k]+3]) - mu_IM[k])/std_IM[k])
            data_n[i][j].append(np.log(data[i][j][IM_ID[k]+3]))

#Create final matrix IM shape
IM_final = np.ones((n_rupture, n_station, n_IM))*(-8.0)

#Populate final matrix
print('Save data...')
for i in range(n_rupture):
    for j in range(len(data[i])):
        IM_final[i, j, :] = data_n[i][j][:]
    
#Transform labels
for i in range(n_rupture):
    GM_name[i] = label_dict[GM_name[i]]

GM_name = np.asarray(GM_name)

#Save final matrix and final labels
np.save('./gen/X_train.npy', IM_final)
np.save('./gen/y_train.npy', GM_name)
    
#Save IM dictionary of training data
IM_dict_train = {}
for i in range(len(IM_considered)):
    IM_dict_train[IM_considered[i]] = i

gcl.saveIMDict_trainData(IM_dict_train)
    
    
    
    
    
    
    
    

