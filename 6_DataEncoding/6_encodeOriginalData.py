import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle
import GMdataImportFx as gix
import prepDx as pdx

###########################################
# User-defined parameters
###########################################
#Model name
model_name = 'RF_discriminator.mdl'

#IM used by the discriminator
IM_name = 'AI'
###########################################

#Load generator data name
GM_name = np.load('./data/generator/X_name.npy')

#Load dictionary of real stations
realID = gml.loadRealStationID('./realStationID.csv')

#Load dictionary of IM - position in file
IM_dict = gix.loadIMDict('./IM_dict.csv')
IM_ID = IM_dict[IM_name] + 3

#Load discriminator
rf_load = open('./Discriminator/' + model_name, 'rb')
discr = pickle.load(rf_load)

#For each data name
P = []
for i in range(len(GM_name)):
    #Load GM
    str_rupt = './data/discriminator/' + GM_name[i] + '.csv'
    GM = du.loadGM_1D(str_rupt, realID, IM_ID)
    
    #Get vector of P
    P.append(discr.predict_proba([GM]))

#Save P
P = np.asarray(P)
P = P.reshape((len(GM_name), 20, 10))
dir_path = './gen/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)
    
np.save('./gen/P_X.npy', P)





























