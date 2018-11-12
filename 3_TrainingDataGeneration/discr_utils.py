import numpy as np
import csv
import prepDx as pdx
import GMdataImportFx as gix


#####################################################################################
# isDiffFromAll: determine if a is differnt from all element contained in X
#####################################################################################
def isDiffFromAll(X, a):
	isDiff = True
	for x in X:
		if x == a:
			isDiff = False
			break

	return isDiff


#####################################################################################
# createKFold: create K folder based on given data and labels, only returns indices (each label present in at least 2 folders)
#####################################################################################
def createKFold(K, y, lbl):
	y_copy = y
	y_K = [[] for i in range(K)]
	ind_K = np.ones((len(y)))*(-1)

	ind_K[0] = 0
	ind_K[1] = 1
	y_K[0].append(y[0])
	y_K[1].append(y[1])

	y = np.delete(y, 0)
	y = np.delete(y, 0)

	for i in range(len(y)-1, -1, -1):
		if isDiffFromAll(y_K[0], y[i]):
			ind_K[i] = 0
			y_K[0].append(y[i])
			y = np.delete(y, i)

	for i in range(len(y)-1, -1, -1):
		if isDiffFromAll(y_K[1], y[i]):
			ind_K[i] = 1
			y_K[1].append(y[i])
			y = np.delete(y, i)

	for i in range(len(y_copy)):
		if ind_K[i] == -1:
			ind_K[i] = np.random.randint(K)

	return ind_K


#####################################################################################
# createTrainValDataset: create the training and validation folders based on K-fold structure and index
#####################################################################################
def createTrainValDataset(X, y, ind_K, k, i_IM):
	X_val = []
	y_val = []
	X_train = []
	y_train = []

	for i in range(len(ind_K)):
		if ind_K[i] == k:
			X_val.append(X[i, :, i_IM])
			y_val.append(y[i])
		else:
			X_train.append(X[i, :, i_IM])
			y_train.append(y[i])

	return X_train, y_train, X_val, y_val


#####################################################################################
# loadGM_1D: load a ground motion composed of recorded IM from real stations
#####################################################################################
def loadGM_1D(str_rupt, realID, IM_ID):
	#Load GM
	GM = pdx.loadGM_CS(str_rupt)

	#Populate matrix
	GM_prep = np.ones((len(realID))) * (-8.0)
	for i in range(len(GM)):
		GM_prep[int(realID[GM[i][0]])] = np.log(GM[i][IM_ID])

	return GM_prep


#####################################################################################
# saveFaultDict: save fault dictionary used by the RF
#####################################################################################
def saveFaultDict(fault_dict):
	file_str = ''
	name = list(fault_dict.keys())
	ind = list(fault_dict.values())

	#Create station file
	for i in range(len(fault_dict)):
		file_str = file_str + name[i]
		for j in range(len(ind[i])):
			file_str = file_str + ',' + str(ind[i][j])
		if i != len(fault_dict)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './fault_dict.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadFaultDict: load fault dictionary
#####################################################################################
def loadFaultDict(file_path, reverse=False):
	#Load data
	raw_data = gix.loadCSV(file_path, row_ignore=0, col_ignore=0, isInput=False, isCategorical=False)

	#Create dict
	fault_dict = {}
	for i in range(len(raw_data)):
		fault_dict.update({raw_data[i][0] : [int(x) for x in raw_data[i][1:]]})

	if reverse:
		fault_dict = {v: k for k, v in fault_dict.items()}

	return fault_dict



#####################################################################################
# saveRuptureDict: save rupture dict
#####################################################################################
def saveRuptureDict(rupt_dict):
	file_str = ''
	key = list(rupt_dict.keys())

	#Create station file
	for i in range(len(rupt_dict)):
		file_str = file_str + key[i] + ',' + str(rupt_dict[key[i]])
		if i != len(rupt_dict)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './rupture_dict.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadRuptureDict: load rupture dict
#####################################################################################
def loadRuptureDict(file_path, reverse=False):
	#Load data
	raw_data = gix.loadCSV(file_path)

	#Create dict
	rupt_dict = {}
	for i in range(len(raw_data)):
		rupt_dict.update({raw_data[i][0] : int(raw_data[i][1])})

	if reverse:
		rupt_dict = {v: k for k, v in rupt_dict.items()}

	return rupt_dict


#####################################################################################
# createTestDataset: create the test dataset folders based on K-fold structure and index
#####################################################################################
def createTestDataset(X, y, ind_K, k):
	X_test = []
	y_test = []

	for i in range(len(ind_K)):
		if ind_K[i] == k:
			X_test.append(X[i, :, :])
			y_test.append(y[i])

	return X_test, y_test


#####################################################################################
# createTrainDataset: create the train dataset folders based on K-fold structure and index
#####################################################################################
def createTrainDataset(X, y, ind_K, k):
	X_train = []
	y_train = []

	for i in range(len(ind_K)):
		if not ind_K[i] == k:
			X_train.append(X[i, :, :])
			y_train.append(y[i])

	return X_train, y_train


#####################################################################################
# saveTestGMLabel: save the GM label that are selected for testing
#####################################################################################
def saveTestGMLabel(ind_K, k, GM_lbl):
	GM_select = []

	for i in range(len(ind_K)):
		if ind_K[i] == k:
			GM_select.append(GM_lbl[i])

	file_str = ''
	#Create station file
	for i in range(len(GM_select)):
		file_str = file_str + GM_select[i]
		if i != len(GM_select)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './GM_test.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()






