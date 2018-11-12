import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle
from matplotlib import rc

def cm2inch(value):
	return value/2.54

###########################################
# User-defined parameters
###########################################
#Selected IMs for the prediction
IM_name = ['PGA', 'PGV', 'AI', 'pSA_0.1', 'pSA_1.0', 'pSA_3.0']

#Number of cross-validation folders
K = 5

#Number of trees
N_tree = np.arange(25, 2001, 25)

#Model name
model_name = 'discriminator.mdl'

#Tested features max
f_max = ['sqrt']
f_str = ['$\sqrt{N}$']

#Number of processes
N_proc = 12
###########################################

#Import data
print('Load data...')
X = np.load('./data/X_train.npy')
y = np.load('./data/y_train.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')
lbl_GM = gml.loadLabelDict('label_dict.csv', reverse=True)
lbl_GM = list(lbl_GM.keys())
IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]

#Split the dataset (making sure all labels are present in training set)
print('Create ' + str(K) + ' cross-validation folders...')
ind_K = du.createKFold(K, y, lbl_GM)

#Folder and file names
dir_path = './RF_Discriminator/'
if not os.path.exists(dir_path):
	os.mkdir(dir_path)

#CV framework
n_tot = len(IM_name)*len(N_tree)*K
compl = 0
bestModel = []
lowestError = 100000.0
N_tree_best = 0
IM_best = ''
K_best = -1
f_str_best = ''
print('Start the ' + str(K) + '-Fold crossvalidation...')
CV_err_mean = [[] for i in range(len(IM_name))]
CV_err_std = [[] for i in range(len(IM_name))]
for i_IM in  range(len(IM_name)):
    for i_tree in range(len(N_tree)):
        for j in range(len(f_max)):
            CV_err_i = []
            for i in range(K):    
                #Create training and validation set
                X_train, y_train, X_val, y_val = du.createTrainValDataset(X, y, ind_K, i, IM_ID[i_IM])
    
                #Initialize RF
                RF = RandomForestClassifier(n_estimators=N_tree[i_tree], max_depth=None, oob_score=False, verbose=0, bootstrap=True, max_features=f_max[j], n_jobs=N_proc)

                #Fit RF
                RF = RF.fit(X_train, y_train)

                #Test accuracy on test set
                y_val_hat = RF.predict(X_val)
                accuracy_count = 0
                for k in range(len(y_val)):
                    if y_val[k] == y_val_hat[k]:
                        accuracy_count = accuracy_count + 1
                accuracy = 100.*float(accuracy_count)/float(len(y_val))
                CV_err_i.append(1.0-accuracy/100.)
                
                if CV_err_i[-1] < lowestError:
                    lowestError = CV_err_i[-1]
                    bestModel = RF
                    IM_best = IM_name[i_IM]
                    N_tree_best = N_tree[i_tree]
                    f_str_best = f_str[j]
                    K_best = i
                    
                compl = compl + 1
                print('\t' + str(compl) + '/' + str(n_tot))
        
            #Estimate CV error for N_tree_i and IM_i
            CV_err_mean[i_IM].append(np.mean(CV_err_i))
            CV_err_std[i_IM].append(np.std(CV_err_i))


#Train best model using selected hyperparameters
print('Fit best model...')
RF = RandomForestClassifier(n_estimators=N_tree_best, max_depth=None, oob_score=False, verbose=0, bootstrap=True, max_features=f_max[0], n_jobs=N_proc)

#Fit RF
X = X[:, :, int(IM_dict[IM_best])]
X_test = X_test[:, :, int(IM_dict[IM_best])]
RF = RF.fit(X, y)

#Test accuracy on test set
y_test_hat = RF.predict(X_test)
accuracy_count = 0
for k in range(len(y_test)):
    if y_test[k] == y_test_hat[k]:
        accuracy_count = accuracy_count + 1
accuracy_best = 100.*float(accuracy_count)/float(len(y_test))

#Plot results
print('Plot training history and save model...')      
IM_ID_norm = [x/len(IM_ID) for x in IM_ID]
cm_IM = cm.get_cmap('tab10')
IM_col = cm_IM([IM_ID_norm])

rc('text', usetex=True)
rc('font', family='serif', size=13)

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(16), cm2inch(11))
for i in range(len(IM_name)):
    for j in range(len(f_max)):
        lbl_str = IM_name[i]# + ', ' + f_str[j]
        CV_plot = []
        CV_plot_std = []
        for k in range(len(N_tree)):
            CV_plot.append(CV_err_mean[i][k*len(f_max)+j])
            CV_plot_std.append(CV_err_std[i][k*len(f_max)+j])
        ax.plot(N_tree, CV_plot, label=lbl_str, color=IM_col[0, i, :])
        ax.fill_between(N_tree, list(map(add, CV_plot, CV_plot_std)), list(map(sub, CV_plot, CV_plot_std)),  color=IM_col[0, i, :], alpha=0.3)
ax.scatter(N_tree_best, 1.-accuracy_best/100., s=60, marker='d', color='black', label='Selected model (' + IM_best + ')')
ax.grid()
ax.set_axisbelow(True)
lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2,
           ncol=1, borderaxespad=0., fontsize=11)
ax.set_xlabel('Number of random trees')
ax.set_ylabel('Cross-validation error (inaccuracy)')
#plt.tight_layout()
plt.savefig(dir_path + 'CV_RF.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

#Save the model
model_path = dir_path + model_name
pickle.dump(bestModel, open(model_path, 'wb'))


    
