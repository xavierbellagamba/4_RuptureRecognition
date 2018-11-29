import GMdataImportFx as gix
import GMCluster as gcl
import numpy as np
import matplotlib.pyplot as plt
import discr_utils as du
import os

###########################################
# User-defined parameters
###########################################
#Selected IMs
IM_considered = ['PGA', 'PGV', 'pSA_0.1', 'pSA_1.0', 'pSA_3.0', 'Ds595', 'AI']

#Test split
test_split = 0.2
###########################################

#Import IM, station and label dict
station_dict = gix.loadStationDict('station_dict.csv')
label_dict = gcl.loadLabelDict('label_dict.csv')
IM_dict = gix.loadIMDict('IM_dict.csv')
lbl_GM = list(label_dict.values())
lbl_GM_orig = list(label_dict.keys())

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
print('Create IM table...')
IM_val = [[] for i in IM_considered]
for i in range(len(data)):
    for j in range(len(data[i])):
            for k in range(len(IM_considered)):
                IM_val[k].append(data[i][j][IM_ID[k]+3])

#Extract and log IM
IM_val = np.asarray(IM_val)
n_IM = len(IM_considered)
IM_val = np.log(IM_val)

#Normalize IM
print('Log the data...')
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
        IM_final[i, station_dict[data[i][j][0]], :] = data_n[i][j][:]
    
#Transform labels
for i in range(n_rupture):
    GM_name[i] = label_dict[GM_name[i]]

GM_name = np.asarray(GM_name)

#Create train and test sets
N = int(1/test_split)
ind_K = du.createKFold(N, GM_name, lbl_GM)
X_test, y_test = du.createTestDataset(IM_final, GM_name, ind_K, N-1)
X_train, y_train = du.createTrainDataset(IM_final, GM_name, ind_K, N-1)

#Save final matrix and final labels
np.save('./gen/X_train.npy', X_train)
np.save('./gen/y_train.npy', y_train)
np.save('./gen/X_test.npy', X_test)
np.save('./gen/y_test.npy', y_test)
    
#Save IM dictionary of training data
IM_dict_train = {}
for i in range(len(IM_considered)):
    IM_dict_train[IM_considered[i]] = i

gcl.saveIMDict_trainData(IM_dict_train)
    
#Save real station ID
gcl.saveRealStationID(station_dict)

#Save test labels
du.saveTestGMLabel(ind_K, N-1, lbl_GM_orig)

    
    
    
    
    
    
    

