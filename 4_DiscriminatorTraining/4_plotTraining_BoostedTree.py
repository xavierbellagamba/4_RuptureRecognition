import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
import pickle
from matplotlib import rc
from sklearn.ensemble import GradientBoostingClassifier

def cm2inch(value):
	return value/2.54

###########################################
# User-defined parameters
###########################################

###########################################



#Folder and file names
dir_path = './BT_Discriminator/'

#Load results
IM_name = np.load(dir_path + 'im_name.npy')
N_estimator = np.load(dir_path + 'N_est.npy')
CV_err_mean = np.load(dir_path + 'CV_mean.npy')
CV_err_std = np.load(dir_path + 'CV_std.npy')
N_best = np.load(dir_path + 'N_best.npy')
IM_best = np.load(dir_path + 'im_best.npy')
learning_rate = np.load(dir_path + 'lr.npy')
lr_best = np.load(dir_path + 'lr_best.npy')
error_test = np.load(dir_path + 'error_test.npy')

IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]


IM_ID_norm = [x/len(IM_ID) for x in IM_ID]
cm_IM = cm.get_cmap('tab10')
IM_col = cm_IM([IM_ID_norm])
#%%
rc('text', usetex=True)
rc('font', family='serif', size=13)

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(16), cm2inch(11))
for i_IM in range(len(IM_name)):
    lbl_str = IM_name[i_IM]# + ', ' + str(learning_rate[i_lr])
    ax.plot(learning_rate, CV_err_mean[:, i_IM, :], label=lbl_str, color=IM_col[0, i_IM, :])
    ax.fill_between(learning_rate, np.squeeze(CV_err_mean[:, i_IM] + CV_err_std[:, i_IM]), np.squeeze(CV_err_mean[:, i_IM] - CV_err_std[:, i_IM]),  color=IM_col[0, i_IM, :], alpha=0.3)
ax.scatter(lr_best, error_test, s=60, marker='d', color='black', label='Selected model (' + str(IM_best) + ')')
ax.grid()
ax.set_axisbelow(True)
ax.set_xscale('log')
ax.set_xlim([max(learning_rate), min(learning_rate)])
lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0., fontsize=11)
ax.set_xlabel('Learning rate')
ax.set_ylabel('Cross-validation error (inaccuracy)')
plt.savefig(dir_path + 'CV_BT_' + '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
