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
data_name = 'Seddon'#'Kaikoura_init'#'AwatNEVer_HYP28-36_S1514'#'AlpineF2K_HYP10-47_S1334'#'Kaikoura'#'AlpineF2K_HYP44-47_S1674'#'AlpineF2K_HYP22-47_S1454'#'AlpineF2K_HYP01-47_S1244'#'Manaota_HYP08-35_S1314'#

#Generator model name
gen_model_name = 'Generator_BTEncod'

#Discriminator model name
discr_model_name = 'BT_discriminator'

#Save as shapefile
saveSHP = False

#Save plots
savePlot = True
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

#GM = fu.loadSingleGmMap('./data/generator/Kaikoura.csv')


#Load discriminator
print('Load discriminator...')
rf_load = open('./Discriminator/' + discr_model_name + '.mdl', 'rb')
discriminator_GM = pickle.load(rf_load)

#Load generator
print('Load generator...')
json_file = open('./Generator/' + gen_model_name + '/generator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
generator_GM = model_from_json(loaded_model_json)
generator_GM.load_weights('./Generator/' + gen_model_name + "/generator.hdf5")

#Evaluate results
#fu.comparePredictObs(data_name, realID, discriminator_GM, generator_GM, fault_dict, rupture_dict, IM_name, saveSHP, savePlot)
fu.comparePredictObsComplex(data_name, realID, discriminator_GM, generator_GM, fault_dict, rupture_dict, IM_name, saveSHP, savePlot)

