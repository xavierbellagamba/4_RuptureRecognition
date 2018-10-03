import numpy as np
import csv
import glob
import GMdataImportFx as gix
from sklearn.cluster import KMeans 


##################################################################
#loadGM_clust : load the csv to relabel based on the name given in argument
##################################################################
def loadGM_clust(str_GM):
	#Get file names
	file_path_str = './data/' + str_GM + '*.csv'
	file_str = glob.glob(file_path_str)

	#Load every GM corresponding to this label
	GM = []
	GM_lbl = []
	for i in range(len(file_str)):
		GM_lbl.append(file_str[i][7:-4])
		GM.append(gix.loadCSV(file_str[i]))

	return GM, GM_lbl


##################################################################
#getGMNewLabel : get the new labels of GM based on the number of clusters
##################################################################
def getGMNewLabel(K, station_coord, IM, IM_ID):
	K_station = K + 1
	#Cluster the stations based on their relative distance using K+1 groups
	kmeans = KMeans(n_clusters=K_station)
	station_clust = kmeans.fit_predict(station_coord)

	#For each geographic cluster
	station_ID = []
	for k in range(K_station):
		std_station = []
		for l in range(len(station_clust)):
			if k == station_clust[l]:
				#Compute std of each station in the geographic cluster
				std_station.append([l, np.std(IM[:, l, IM_ID])])
            
		#Select the station with the highest std on the wished metric(s)
		std_station = np.asarray(std_station)
		station_ID.append(int(std_station[np.argmax(std_station[:, 1]), 0]))
    
	#Cluster realization in K groups based on the observed values at the selected stations
	IM_cl = IM[:, station_ID, IM_ID]
	kmeans = KMeans(n_clusters=K)
	GM_newLabel = kmeans.fit_predict(IM_cl)
        
	return GM_newLabel, IM_cl, station_clust, station_ID


##################################################################
#getFinalGMLabel : get the final labels based on optimal number of clusters and min number of example per label
##################################################################
def getFinalGMLabel(K_pick, station_coord, IM, IM_ID, GM_lbl, N_min):
	dict_GM = {}
	while True:
		if K_pick > 1:
			#Cluster realization and rename them
			GM_newLabel, _, station_clust, station_ID = getGMNewLabel(K_pick, station_coord, IM, IM_ID)

			#Add dict entries for further convertion of names
			dict_GM = {}
			for j in range(len(GM_lbl)):
				GM_final_lbl = GM_lbl[j].split('_')[0] + '_' + str(GM_newLabel[j])
				dict_GM[GM_lbl[j]] = GM_final_lbl
			#Check if at least N_min examples per label, if not redo with K_pick-1
			uniqueLabel = list(set(list(dict_GM.values())))
			allLabel = list(dict_GM.values())
			count_r = []
			for j in range(len(uniqueLabel)):
				count_r.append(allLabel.count(uniqueLabel[j]))
			if np.min(count_r) < N_min:
				K_pick = K_pick - 1
			else:
				break
		elif K_pick == 1:
			dict_GM = {}
			station_clust = [0 for i in station_coord]
			station_ID = [0]
			for j in range(len(GM_lbl)):
				GM_final_lbl = GM_lbl[j].split('_')[0] + '_0'
				dict_GM[GM_lbl[j]] = GM_final_lbl
			break
 
	return dict_GM, station_clust, station_ID


#####################################################################################
# saveLabelDict: save station dictionary as csv
#####################################################################################
def saveLabelDict(label):
	file_str = ''
	key = list(label.keys())

	#Create station file
	for i in range(len(label)):
		file_str = file_str + key[i] + ',' + label[key[i]]
		if i != len(label)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './label_dict.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadLabelDict: load station dictionary
#####################################################################################
def loadLabelDict(file_path, reverse=False):
	#Load data
	raw_data = gix.loadCSV(file_path)

	#Create dict
	label_dict = {}
	for i in range(len(raw_data)):
		label_dict.update({raw_data[i][0] : raw_data[i][1]})

	if reverse:
		label_dict = {v: k for k, v in label_dict.items()}

	return label_dict


#####################################################################################
# saveIMDict_trainData: save IM dictionary of training data as csv
#####################################################################################
def saveIMDict_trainData(label):
	file_str = ''
	key = list(label.keys())

	#Create station file
	for i in range(len(label)):
		file_str = file_str + key[i] + ',' + str(label[key[i]])
		if i != len(label)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './IM_dict_train.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadIMDict_trainData: load IM dictionary of training data
#####################################################################################
def loadIMDict_trainData(file_path, reverse=False):
	#Load data
	raw_data = gix.loadCSV(file_path)

	#Create dict
	label_dict = {}
	for i in range(len(raw_data)):
		label_dict.update({raw_data[i][0] : raw_data[i][1]})

	if reverse:
		label_dict = {v: k for k, v in label_dict.items()}

	return label_dict

























