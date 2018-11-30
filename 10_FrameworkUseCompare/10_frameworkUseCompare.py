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
data_name = 'HopeConwayOS_HYP11-36_S1344'

#Generator model name
gen_model_name = 'generator'

#Discriminator model name
discr_model_name = 'RF_discriminator'

#Save as shapefile
saveSHP = True
###########################################

#Create result folder
dir_path = './gen/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
#Load dictionaries
print('Load ' + data_name + '...')
realID = gml.loadRealStationID('./realStationID.csv')
IM_name = gix.loadCSV('./selectedIM.csv')

#Load rupture fault dictionary
fault_dict = du.loadFaultDict('./fault_dict.csv')
rupture_dict = du.loadRuptureDict('./rupture_dict.csv')

#Load discriminator
print('Load discriminator...')
rf_load = open('./Discriminator/' + discr_model_name + '.mdl', 'rb')
discriminator_GM = pickle.load(rf_load)

#Load generator
print('Load generator...')
json_file = open('./Generator/' + gen_model_name + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
generator_GM = model_from_json(loaded_model_json)
generator_GM.load_weights('./Generator/' + gen_model_name + ".hdf5")

#Evaluate results
fu.comparePredictObs(data_name, realID, discriminator_GM, generator_GM, fault_dict, rupture_dict, IM_name, saveSHP)