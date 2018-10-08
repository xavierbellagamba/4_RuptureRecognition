import numpy as np
import csv
import prepDx as pdx


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

















