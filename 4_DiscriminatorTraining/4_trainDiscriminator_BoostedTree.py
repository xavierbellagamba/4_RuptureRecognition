import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle
from matplotlib import rc

def cm2inch(value):
	return value/2.54
#%%
###########################################
# User-defined parameters
###########################################
#Selected IMs for the prediction
IM_name = ['PGA']#, 'PGV', 'AI']#, 'pSA0.1', 'pSA1.0', 'pSA3.0']

#Number of cross-validation folders
K = 2

#Number of trees
N_estimator = np.arange(5, 11, 5)

#Model name
model_name = 'discriminator.mdl'

#Tested features max
learning_rate = [0.5]

#Number of processes
N_proc = 2
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
dir_path = './gen/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

#Folder and file names
dir_path = './BT_Discriminator/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

#CV framework
bestModel = []
lowestError = 100000.0
N_best = 0
IM_best = ''
lr_best = -1.0
K_best = -1
print('Start the ' + str(K) + '-Fold crossvalidation...')
CV_err_mean = [[[[] for k in range(len(N_estimator))] for j in range(len(IM_name))] for i in range(len(learning_rate))]
CV_err_std = [[[[] for k in range(len(N_estimator))] for j in range(len(IM_name))] for i in range(len(learning_rate))]
for i_lr in range(len(learning_rate)):
    for i_IM in  range(len(IM_name)):
        for i_est in range(len(N_estimator)):  
            CV_err_i = []
            for i in range(K):
                print('Cross-validation folder ' + str(i+1) + '...')
    
                #Create training and validation set
                print('\t Create the training and validation dataset...')
                X_train, y_train, X_val, y_val = du.createTrainValDataset(X, y, ind_K, i, IM_ID[i_IM])
    
                #Initialize RF
                print('\t Initialize the boosted tree with ' + str(N_estimator[i_est]) + ' trees and learning rate ' + str(learning_rate[i_lr]) + IM_name[i_IM] + '...')
                BT = GradientBoostingClassifier(learning_rate=learning_rate[i_lr], 
                                                n_estimators=N_estimator[i_est], 
                                                min_samples_split=2, 
                                                max_depth=1, 
                                                max_features=None, 
                                                verbose=0, 
                                                validation_fraction=0.1, 
                                                n_iter_no_change=5)

                #Fit RF
                print('\t Fit the random forest...')
                BT = BT.fit(X_train, y_train)

                #Test accuracy on test set
                print('\t Test algorithm on validation set...')
                y_val_hat = BT.predict(X_val)
                accuracy_count = 0
                for k in range(len(y_val)):
                    if y_val[k] == y_val_hat[k]:
                        accuracy_count = accuracy_count + 1
                accuracy = 100.*float(accuracy_count)/float(len(y_val))
                print('\t Validation accuracy = ' + str(accuracy) + '%')
                CV_err_i.append(1.0-accuracy/100.)
                if CV_err_i[-1] < lowestError:
                    lowestError = CV_err_i[-1]
                    bestModel = BT
                    IM_best = IM_name[i_IM]
                    N_best = N_estimator[i_est]
                    K_best = i
                    lr_best = learning_rate[i_lr]
        
            #Estimate CV error for N_tree_i and IM_i
            CV_err_mean[i_lr][i_IM][i_est] = np.mean(CV_err_i)
            CV_err_std[i_lr][i_IM][i_est] = np.std(CV_err_i)

#Plot results
#%%
print('Plot training history and save model...')
ls = ['-', '--', '-.']        
IM_ID_norm = [x/len(IM_ID) for x in IM_ID]
cm_IM = cm.get_cmap('tab10')
IM_col = cm_IM([IM_ID_norm])

rc('text', usetex=True)
rc('font', family='serif', size=13)



for i_lr in range(len(learning_rate)):
    fig, ax = plt.subplots()
    fig.set_size_inches(cm2inch(16), cm2inch(11))
    for i_IM in range(len(IM_name)):
        lbl_str = IM_name[i_IM] + ', ' + str(learning_rate[i_lr])
        ax.plot(N_estimator, CV_err_mean[i_lr][i_IM], label=lbl_str, color=IM_col[0, i_IM, :])#, linestyle=ls[j])
        ax.fill_between(N_estimator, list(map(add, CV_err_mean[i_lr][i_IM], CV_err_std[i_lr][i_IM])), list(map(sub, CV_err_mean[i_lr][i_IM], CV_err_std[i_lr][i_IM])),  color=IM_col[0, i_IM, :], alpha=0.3)
    if learning_rate[i_lr] == lr_best:
        ax.scatter(N_best, lowestError, s=60, marker='d', color='black', label='Best model (' + IM_best + ', ' + str(lr_best) + ')')
    ax.grid()
    ax.set_axisbelow(True)
    lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2,
                    ncol=1, borderaxespad=0., fontsize=11)
    ax.set_xlabel('Number of random trees')
    ax.set_ylabel('Cross-validation error (inaccuracy)')
    plt.savefig(dir_path + 'CV_BT_' + str(i_lr) + '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')


#Save the model
model_path = dir_path + model_name
pickle.dump(bestModel, open(model_path, 'wb'))


#Save test set
print('Save test dataset...')
_, _, X_test, y_test = du.createTrainValDataset(X, y, ind_K, K_best, int(IM_dict[IM_best]))
np.save('./gen/X_test_BT.npy', X_test)
np.save('./gen/y_test_BT.npy', y_test)























    
