import GMdataImportFx as gix
import GMCluster as gcl
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def cm2inch(value):
	return value/2.54


###########################################
# User-defined parameters
###########################################
#Selected IM
IM_name = 'Ds595'

#Min number of exaple per label
N_min = 8

#Save map and silhouette score plots (True = yes)
savePlots = True
###########################################

#Import faultList
fault = gix.loadCSV('selectedFault_list.csv')
#fault = fault[0]

#Load station dict
station_dict = gix.loadStationDict('station_dict.csv')

#Load IM dict
IM_dict = gix.loadIMDict('./IM_dict.csv')
IM_ID = IM_dict[IM_name]

#Initialize dict label
dict_GM = {}

if savePlots:
    #Create results folder
    folder_path = './gen'
    try:
        os.makedirs(folder_path)
    except:
        print('Folder already exists')

#For each fault
for i in range(len(fault)):
    #Max number of cluster (magnitude dep.)
    Kmax = int(np.floor((10+(fault[i][1]-6.0)*20.)/10.0))
    
    #Number of cluster
    K = np.arange(2, Kmax+1)

    #Gather all the realizations and there names
    GM, GM_lbl = gcl.loadGM_clust(fault[i][0]+'_')

    if len(GM) > 0:
        print(fault[i][0])
        #Extract station coordinates and GM intensity
        station_coord = np.asarray(GM[0])[:, 1:3].astype(np.float)
        IM = np.asarray(GM)[:, :, 3:].astype(np.float)

        #For each number of cluster
        silhouette_avg = []
        for j in range(len(K)):
            GM_newLabel, IM_cl, _, _ = gcl.getGMNewLabel(K[j], station_coord, IM, IM_ID)
        
            #Compute the silhouette score of K
            silhouette_avg.append(silhouette_score(IM_cl, GM_newLabel))
        
        #Pick the K such that the silhouette score is the highest
        K_pick = K[np.argmax(silhouette_avg)]
        
        #Get final labels
        dict_fault, station_clust, station_ID = gcl.getFinalGMLabel(K_pick, station_coord, IM, IM_ID, GM_lbl, N_min)
        
        #Add fault labels to global dict
        dict_GM.update(dict_fault)
        
        if savePlots:
            #Plot results
            #Create results folder
            folder_path = './gen/' + fault[i][0]
            try:
                os.makedirs(folder_path)
            except:
                o = 1

            silhouette_str = folder_path + '/' + fault[i][0] + '_silhouette.pdf'
            map_str = folder_path + '/' + fault[i][0] + '_map.pdf'
        
            #Plot silhouette score
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
            plt.figure(1)
            plt.annotate(IM_name, (K[-1]+.004, np.max(silhouette_avg)-0.005), bbox=bbox_props, fontsize=14)
            plt.scatter(K, silhouette_avg, marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette score')
            plt.grid()
            plt.tight_layout()
            plt.savefig(silhouette_str, dpi=600)
            plt.close()
            #Plot map
            plt.figure(2)
            plt.annotate(IM_name, (np.min(station_coord[:, 0])+0.025*(np.max(station_coord[:, 0])-np.min(station_coord[:, 0])), np.min(station_coord[:, 1])+0.9*(np.max(station_coord[:, 1])-np.min(station_coord[:, 1]))), bbox=bbox_props, fontsize=14)
            plt.scatter(station_coord[:, 0], station_coord[:, 1], c=station_clust)
            plt.scatter(station_coord[station_ID, 0], station_coord[station_ID, 1], s=60, marker='s', edgecolors='black', facecolors='none')
            for i in range(len(station_ID)):
                plt.annotate(GM[0][station_ID[i]][0], (station_coord[station_ID[i], 0]+300, station_coord[station_ID[i], 1]+300))
            plt.xlabel('Easting, NZTM2000, [m]')
            plt.ylabel('Northing, NZTM2000, [m]')
            plt.grid()
            plt.tight_layout()
            plt.savefig(map_str, dpi=600)
            plt.close()

#Save the new label dictionary
gcl.saveLabelDict(dict_GM)
