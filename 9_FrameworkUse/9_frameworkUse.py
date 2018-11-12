import numpy as np
import keras as k
import gene_utils as gu
from keras.models import model_from_json
import GMdataImportFx as gmc
from sklearn.ensemble import RandomForestClassifier
import GMCluster as gml
import discr_utils as du
import pickle
import fwk_utils as fu
import os
import GMdataImportFx as gix

###########################################
# User-defined parameters
###########################################
#GM data
data_name = 'test'

#Generator model name
gen_model_name = 'generator'

#Discriminator model name
discr_model_name = 'RF_discriminator'
###########################################

#Create result folder
dir_path = './gen/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
#Create result folder
dir_path = './gen/' + data_name
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
#Load data
#Load dictionaries
print('Load ' + data_name + '...')
realID = gml.loadRealStationID('./realStationID.csv')
IM_name = gix.loadCSV('./selectedIM.csv')

#Create the IM array for discriminator
GM = du.loadGM_1D('./data/' + data_name + '.csv', realID, 1)
GM = GM.reshape(1, -1)

#Load rupture fault dictionary
fault_dict = du.loadFaultDict('./fault_dict.csv')
rupture_dict = du.loadRuptureDict('./rupture_dict.csv')

#Get discriminator results
#Load discriminator
print('Load discriminator...')
rf_load = open('./Discriminator/' + discr_model_name + '.mdl', 'rb')
discriminator_GM = pickle.load(rf_load)

#Get discriminator results
print('Predict fault and rupture probabilities...\n')
#Rupture 
P_rupture = discriminator_GM.predict_proba(GM)[0]
rupture_hat = discriminator_GM.predict(GM)

#Fault
P_fault = fu.getFaultProba(P_rupture, fault_dict)
fault_hat = fu.predictFault(P_fault, fault_dict)
print('Most probable fault: \t' + fault_hat)
print('Most probable rupture: \t' + rupture_hat[0])

#Export discriminator results
print('\nSave discriminator results...')
fu.saveDiscriminatorResults(P_rupture, P_fault, rupture_dict, fault_dict, data_name)

#Get generator results
#Load generator
print('Load generator...')
json_file = open('./Generator/' + gen_model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
generator_GM = model_from_json(loaded_model_json)
generator_GM.load_weights('./Generator/' + gen_model_name + ".hdf5")

#Get results from generator
print('Generate GM map...')
P_rupture = P_rupture.reshape(1, 1, 1, -1)
GM_pred = generator_GM.predict(P_rupture)

#Export generator results
print('Save generator results...')
fu.saveGeneratorResults(GM_pred[0], data_name, IM_name)