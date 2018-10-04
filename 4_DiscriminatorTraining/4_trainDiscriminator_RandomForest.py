import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle

###########################################
# User-defined parameters
###########################################
#Selected IMs for the prediction
IM_name = ['PGV']

#Number of cross-validation folders
K = 5

#Number of trees
N_tree = np.arange(30, 41, 10)

#Model name
model_name = 'RF_disriminator.mdl'
###########################################

#Import data
print('Load data...')
X = np.load('./data/X_train.npy')
y = np.load('./data/y_train.npy')
lbl_GM = gml.loadLabelDict('label_dict.csv', reverse=True)
lbl_GM = list(lbl_GM.keys())
IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]

#Split the dataset (making sure all labels are present in training set)
print('Create ' + str(K) + ' cross-validation folders...')
ind_K = du.createKFold(K, y, lbl_GM)

#Folder and file names
dir_path = './Discriminator/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

#CV framework
bestModel = []
lowestError = 100000.0
N_tree_best = 0
IM_best = ''
print('Start the ' + str(K) + '-Fold crossvalidation...')
CV_err_mean = [[] for i in range(len(IM_name))]
CV_err_std = [[] for i in range(len(IM_name))]
for i_IM in  range(len(IM_name)):
    for i_tree in range(len(N_tree)):
        CV_err_i = []
        for i in range(K):
            print('Cross-validation folder ' + str(i+1) + '...')
    
            #Create training and validation set
            print('\t Create the training and validation dataset...')
            X_train, y_train, X_val, y_val = du.createTrainValDataset(X, y, ind_K, i, IM_ID[i_IM])
    
            #Initialize RF
            print('\t Initialize the random forest with ' + str(N_tree[i_tree]) + ' trees and ' + IM_name[i_IM] + '...')
            RF = RandomForestClassifier(n_estimators=N_tree[i_tree], max_depth=None, oob_score=True, verbose=0, bootstrap=True)

            #Fit RF
            print('\t Fit the random forest...')
            RF = RF.fit(X_train, y_train)

            #Test accuracy on test set
            print('\t Test algorithm on validation set...')
            y_val_hat = RF.predict(X_val)
            accuracy_count = 0
            for i in range(len(y_val)):
                if y_val[i] == y_val_hat[i]:
                    accuracy_count = accuracy_count + 1
            accuracy = 100.*float(accuracy_count)/float(len(y_val))
            print('\t Validation accuracy = ' + str(accuracy) + '%')
            CV_err_i.append(1.0-accuracy/100.)
            if CV_err_i[-1] < lowestError:
                lowestError = CV_err_i[-1]
                bestModel = RF
                IM_best = IM_name[i_IM]
                N_tree_best = N_tree[i_tree]

        #Estimate CV error for N_tree_i and IM_i
        CV_err_mean[i_IM].append(np.mean(CV_err_i))
        CV_err_std[i_IM].append(np.std(CV_err_i))

#Plot results        
IM_ID_norm = [x/len(IM_ID) for x in IM_ID]
cm_IM = cm.get_cmap('winter')
IM_col = cm_IM([IM_ID_norm])
fig, ax = plt.subplots()
for i in range(len(IM_name)):
    ax.scatter(N_tree, CV_err_mean[i], label=IM_name[i], color=IM_col[0, i, :])
    ax.scatter(N_tree, list(map(add, CV_err_mean[i], CV_err_std[i])), marker='_', color=IM_col[0, i, :])
    ax.scatter(N_tree, list(map(sub, CV_err_mean[i], CV_err_std[i])), marker='_', color=IM_col[0, i, :])
ax.scatter(N_tree_best, lowestError, s=60, marker='d', color='black', label='Best model (' + IM_best + ')')
ax.grid()
ax.set_axisbelow(True)
ax.legend()
ax.set_xlabel('Number of random trees')
ax.set_ylabel('Cross-validation error (inaccuracy)')
plt.tight_layout()
plt.savefig(dir_path + 'CV_RF.pdf', dpi=600)

#Save the model
model_path = dir_path + model_name
pickle.dump(bestModel, open(model_path, 'wb'))
























    
