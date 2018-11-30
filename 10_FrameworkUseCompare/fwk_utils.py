import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
import os
from operator import itemgetter
import GMdataImportFx as gix
import keras as k
import gene_utils as gu
import GMdataImportFx as gmc
import GMCluster as gml
import prepDx as pdx
import discr_utils as du
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#####################################################################################
# saveResultSHP : save one result (1 IM) as a shp file - NZTM2000
#####################################################################################
def saveResultSHP(GM, str_file):
	df = pd.DataFrame(GM, columns=['x','y','IM'])
	df['geometry'] = df.apply(lambda row: Point(row.x,row.y),axis=1)
	df = df.drop(['x', 'y'], axis=1)
	crs = {'init': 'epsg:2193'}
	gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
	gdf.to_file(str_file)


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
def saveGeneratorResults(GM_map, data_name, IM_name, saveSHP):
	s = GM_map.shape

	translation = [1200000., 4570000.]
	rotation_angle = -25.0
	rotation_pivot = gix.convertWGS2NZTM2000(172.6376, -43.5309)

	if saveSHP:
		#Create result folder
		dir_path = './gen/' + data_name + '/shp/'
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

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

		if saveSHP:
			str_file = './gen/' + data_name + '/shp/' + IM_name[i][0] + '_pred.shp'
			saveResultSHP(np.array(GM_raster), str_file)


#####################################################################################
#evaluateEvent: evaluate the event and produce GM map
#####################################################################################
def evaluateEvent(data_name, realID, discriminator_GM, generator_GM, fault_dict, rupture_dict, IM_name, saveSHP):
	#Create result folder
	dir_path = './gen/' + data_name
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

	#Create the IM array for discriminator
	GM = du.loadGM_1D('./data/' + data_name + '.csv', realID, 1)
	GM = GM.reshape(1, -1)

	#Get discriminator results
	print('Predict fault and rupture probabilities...\n')
	#Rupture 
	P_rupture = discriminator_GM.predict_proba(GM)[0]
	rupture_hat = discriminator_GM.predict(GM)

	#Fault
	P_fault = getFaultProba(P_rupture, fault_dict)
	fault_hat = predictFault(P_fault, fault_dict)
	print('Most probable fault: \t' + fault_hat)
	print('Most probable rupture: \t' + rupture_hat[0])

	#Export discriminator results
	print('\nSave discriminator results...')
	saveDiscriminatorResults(P_rupture, P_fault, rupture_dict, fault_dict, data_name)

	#Get results from generator
	print('Generate GM map...')
	P_rupture = P_rupture.reshape(1, 1, 1, -1)
	GM_pred = generator_GM.predict(P_rupture)

	#Export generator results
	print('Save generator results...')
	saveGeneratorResults(GM_pred[0], data_name, IM_name, saveSHP)


#####################################################################################
# saveGeneratorResultsComparison : save the GM maps
#####################################################################################
def saveGeneratorResultsComparison(GM_map, GM_obs, data_name, IM_name, saveSHP):
	s = GM_map.shape

	translation = [1200000., 4570000.]
	rotation_angle = -25.0
	rotation_pivot = gix.convertWGS2NZTM2000(172.6376, -43.5309)

	if saveSHP:
		#Create result folder
		dir_path = './gen/' + data_name + '/shp/'
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)

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

		with open('./gen/' + data_name + '/' + IM_name[i][0] + '_pred.csv', "w") as f:
			writer = csv.writer(f)
			writer.writerows(GM_raster)
		f.close()

		if saveSHP:
			str_file = './gen/' + data_name + '/shp/' + IM_name[i][0] + '_pred.shp'
			saveResultSHP(np.array(GM_raster), str_file)

	for i in range(s[2]):
		GM_raster = []

		for j in range(s[0]):
			for k in range(s[1]):
				GM_raster.append([grid_loc[j*s[1]+k][0], grid_loc[j*s[1]+k][1], np.exp(GM_obs[j, k, i])])

		with open('./gen/' + data_name + '/' + IM_name[i][0] + '_obs.csv', "w") as f:
			writer = csv.writer(f)
			writer.writerows(GM_raster)
		f.close()

		if saveSHP:
			str_file = './gen/' + data_name + '/shp/' + IM_name[i][0] + '_pred.shp'
			saveResultSHP(np.array(GM_raster), str_file)

	for i in range(s[2]):
		GM_raster = []

		for j in range(s[0]):
			for k in range(s[1]):
				GM_raster.append([grid_loc[j*s[1]+k][0], grid_loc[j*s[1]+k][1], GM_map[j, k, i] - GM_obs[j, k, i]])

		with open('./gen/' + data_name + '/' + IM_name[i][0] + '_res.csv', "w") as f:
			writer = csv.writer(f)
			writer.writerows(GM_raster)
		f.close()

		if saveSHP:
			str_file = './gen/' + data_name + '/shp/' + IM_name[i][0] + '_pred.shp'
			saveResultSHP(np.array(GM_raster), str_file)


#####################################################################################
#loadSingleGMmap: load a GM map frm simulation set (must respect layout given in 5)
#####################################################################################
def loadSingleGmMap(rupt_name):
	#Load station rot dict
	station_rot_dict = pdx.loadRotStationDict('rot_station_dict.csv')

	#Get grid size
	x_grid_max = np.max(np.asarray(list(station_rot_dict.values()))[:, 0])
	y_grid_max = np.max(np.asarray(list(station_rot_dict.values()))[:, 1])

	#Name of IM to save
	IM_name = ['PGA', 'PGV', 'AI', 'pSA_0.5', 'pSA_1.0', 'pSA_3.0']

	#Ensure IM IDs are well sorted
	lbl_IM = pdx.loadGM_CS(rupt_name)[0]
	IM_pos = []
	for i in range(len(lbl_IM)):
		for j in range(len(IM_name)):
			if IM_name[j] == lbl_IM[i]:
				IM_pos.append(i)
				break

	#Initialize array
	GM = np.ones((x_grid_max, y_grid_max, len(IM_name)))
	for i in range(len(IM_name)):
		if IM_name[i] == 'PGV':
			GM[:, :, i] = GM[:, :, i]*(0.25)
		elif IM_name[i] == 'AI':
			GM[:, :, i] = GM[:, :, i]*(-2.)
		elif IM_name[i] == 'pSA_3.0':
			GM[:, :, i] = GM[:, :, i]*(-6.)
		else:
			GM[:, :, i] = GM[:, :, i]*(-5.)

	#Load data
	GM_i = pdx.loadGM_CS(rupt_name)
	unfoundStation = []
	for j in range(1, len(GM_i)):
		for k in range(len(IM_pos)):
			try:
				GM[station_rot_dict[GM_i[j][0]][0], station_rot_dict[GM_i[j][0]][1], k] = np.log(GM_i[j][IM_pos[k]])
			except: 
				unfoundStation.append(GM_i[j][0])

	GM = np.transpose(GM, (1, 0, 2))

	return GM


#####################################################################################
#evaluateEvent: evaluate the event and produce GM map
#####################################################################################
def comparePredictObs(data_name, realID, discriminator_GM, generator_GM, fault_dict, rupture_dict, IM_name, saveSHP):
	#Create result folder
	dir_path = './gen/' + data_name
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)

	#Create the IM array for discriminator
	GM = du.loadGM_1D('./data/discriminator/' + data_name + '.csv', realID, 1)
	GM = GM.reshape(1, -1)

	#Load estimated GM map for comparison
	GM_obs = loadSingleGmMap('./data/generator/' + data_name + '.csv')

	#Get discriminator results
	print('Predict fault and rupture probabilities...\n')
	#Rupture 
	P_rupture = discriminator_GM.predict_proba(GM)[0]
	rupture_hat = discriminator_GM.predict(GM)

	#Fault
	P_fault = getFaultProba(P_rupture, fault_dict)
	fault_hat = predictFault(P_fault, fault_dict)
	print('Most probable fault: \t' + fault_hat)
	print('Most probable rupture: \t' + rupture_hat[0])

	#Export discriminator results
	print('\nSave discriminator results...')
	saveDiscriminatorResults(P_rupture, P_fault, rupture_dict, fault_dict, data_name)

	#Get results from generator
	print('Generate GM map...')
	P_rupture = P_rupture.reshape(1, 1, 1, -1)
	GM_pred = generator_GM.predict(P_rupture)

	#Export generator results
	print('Save generator results...')
	saveGeneratorResultsComparison(GM_pred[0], GM_obs, data_name, IM_name, saveSHP)





































