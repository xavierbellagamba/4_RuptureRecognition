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
N_proc = 3
###########################################

IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]

#Folder and file names
dir_path = './RF_Discriminator/'

#Load results
IM_name = np.load(dir_path + 'im_name.npy')
N_tree = np.load(dir_path + 'N_est.npy')
CV_err_mean = np.load(dir_path + 'CV_mean.npy')
CV_err_std = np.load(dir_path + 'CV_std.npy')
N_tree_best = np.load(dir_path + 'N_best.npy')
IM_best = np.load(dir_path + 'im_best.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')
X = np.load('./data/X_train.npy')
y = np.load('./data/y_train.npy')

#Load model
#RF = pickle.load(open(dir_path + 'discriminator.mdl', 'rb'))

#Compute accuracy best
IM_best = np.array('AI')
X = X[:, :, int(IM_dict[str(IM_best)])]
X_test = X_test[:, :, int(IM_dict[str(IM_best)])]

N_tree_best = 1000
RF = RandomForestClassifier(n_estimators=N_tree_best, max_depth=None, oob_score=False, verbose=0, bootstrap=True, max_features=f_max[0], n_jobs=N_proc)
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
IM_name = ['PGA', 'PGV', 'AI', 'pSA(0.1s)', 'pSA(1.0s)', 'pSA(3.0s)' ]

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
ax.scatter(N_tree_best, 1.-accuracy_best/100., s=60, marker='d', color='black', label='Selected model (' + str(IM_best) + ')')
ax.grid()
ax.set_axisbelow(True)
lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2,
           ncol=1, borderaxespad=0., fontsize=11)
ax.set_xlabel('Number of random trees')
ax.set_ylabel('Cross-validation error (inaccuracy)')
ax.set_xlim([min(N_tree), max(N_tree)])
#plt.tight_layout()

plt.savefig(dir_path + 'CV_RF.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')

#Save the model
model_path = dir_path + model_name
pickle.dump(RF, open(model_path, 'wb'))


    
