import numpy as np
import csv


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
# createKFold: create K folder based on given data and labels (each label must be present in at least 2 folders)
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





	'''

	X_K = []
	y_K = []

	#Create the folders
	X_K = [[] for i in range(K)]
	y_K = [[] for i in range(K)]

	#Ensure that the 2 first folders contain at least one element of each label
	X_K[0].append(X[0])
	y_K[0].append(y[0])
	X_K[1].append(X[1])
	y_K[1].append(y[1])
	X = np.delete(X, 0)
	X = np.delete(X, 0)
	y = np.delete(y, 0)
	y = np.delete(y, 0)
	for i in range(len(y)-1, -1, -1):
		if isDiffFromAll(y_K[0], y[i]):
			X_K[0].append(X[i])
			y_K[0].append(y[i])
			X = np.delete(X, i)
			y = np.delete(y, i)
	for i in range(len(y)-1, -1, -1):
		if isDiffFromAll(y_K[1], y[i]):
			X_K[1].append(X[i])
			y_K[1].append(y[i])
			X = np.delete(X, i)
			y = np.delete(y, i)

	#Assign the rest randomly
	v_val = np.random.randint(K, size=len(y))
	for i in range(len(y)):
		X_K[v_val[i]].append(X[i])
		y_K[v_val[i]].append(y[i])

	#Reshape output in one 4D array
	X_K = [np.asarray(x) for x in X_K]
	y_K = [np.asarray(y) for y in y_K]
	X_K = np.asarray(X_K)
	y_K = np.asarray(y_K)

	return X_K, y_K
	'''


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

	'''
	X_val = X_K[k]
	y_val = y_K[k]

	t = []
	for i in range(X_K.shape[0]):
		if not i == k:
			t.append(i)

	X_train = X_K[t[0]]
	y_train = y_K[t[0]]

	for i in range(1, len(t)):
		X_train = np.concatenate((X_train, X_K[t[i]]), axis=0)
		y_train = np.concatenate((y_train, y_K[t[i]]), axis=0)

	return X_train, y_train, X_val, y_val
	'''




























