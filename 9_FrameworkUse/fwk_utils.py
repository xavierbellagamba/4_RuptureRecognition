import numpy as np
import csv
import os
from operator import itemgetter
import GMdataImportFx as gix


#####################################################################################
# getFaultProba : based on estimated rupture proba and dict of rupture->fault, estimate proba of fault
#####################################################################################
def getFaultProba(P, fault_dict):
	P_fault = np.zeros(len(fault_dict))

	ind = list(fault_dict.values())

	for i in range(len(fault_dict)):
		for j in range(len(ind[i])):
			P_fault[i] = P_fault[i] + P[ind[i][j]]

	return P_fault

#####################################################################################
# predictFault : based on the estimated proba of fault, return most proba fault
#####################################################################################
def predictFault(P_fault, fault_dict):
	ind_max = np.argmax(P_fault)
	fault_name = list(fault_dict.keys())

	return fault_name[ind_max]


#####################################################################################
# saveDiscriminatorResults : save the proba vector for fault and rupture
#####################################################################################
def saveDiscriminatorResults(P_rupture, P_fault, rupture_dict, fault_dict, data_name):
	rupture_name = list(rupture_dict.keys())
	fault_name = list(fault_dict.keys())

	#Save proba of rupture
	rupt_res = []
	for i in range(len(P_rupture)):
		rupt_res.append([rupture_name[i], P_rupture[i]])

	#Sort results
	rupt_res = sorted(rupt_res, key=itemgetter(1), reverse=True)

	#Export results
	with open('./gen/' + data_name + '/P_rupture.csv', "w") as f:
		writer = csv.writer(f)
		writer.writerows(rupt_res)
	f.close()

	#Save proba of rupture
	fault_res = []
	for i in range(len(P_fault)):
		fault_res.append([fault_name[i], P_fault[i]])

	#Sort results
	fault_res = sorted(fault_res, key=itemgetter(1), reverse=True)

	#Export results
	with open('./gen/' + data_name + '/P_fault.csv', "w") as f:
		writer = csv.writer(f)
		writer.writerows(fault_res)
	f.close()


#####################################################################################
# saveGeneratorResults : save the GM maps
#####################################################################################
def saveGeneratorResults(GM_map, data_name, IM_name):
	s = GM_map.shape

	translation = [1200000., 4570000.]
	rotation_angle = -25.0
	rotation_pivot = gix.convertWGS2NZTM2000(172.6376, -43.5309)

	grid_loc = []
	for i in range(s[0]):
		for j in range(s[1]):
			x = i*20000.
			y = j*20000.
			loc = gix.translateCoordinate(translation, [y, x])
			loc = gix.rotateCoordinate(rotation_pivot, loc, rotation_angle)
			grid_loc.append(loc)

	for i in range(s[2]):
		GM_raster = []

		for j in range(s[0]):
			for k in range(s[1]):
				GM_raster.append([grid_loc[j*s[1]+k][0], grid_loc[j*s[1]+k][1], np.exp(GM_map[j, k, i])])

		with open('./gen/' + data_name + '/' + IM_name[i][0] + '.csv', "w") as f:
			writer = csv.writer(f)
			writer.writerows(GM_raster)
		f.close()

















