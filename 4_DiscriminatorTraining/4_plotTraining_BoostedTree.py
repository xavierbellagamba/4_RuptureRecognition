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
IM_name = ['PGA', 'PGV', 'AI', 'pSA(0.1s)', 'pSA(1.0s)', 'pSA(3.0s)' ]
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
#lgd = ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0., fontsize=11)
ax.legend(loc=1, ncol=1, fontsize=11)

ax.set_xlabel('Learning rate')
ax.set_ylabel('Cross-validation error (inaccuracy)')
plt.tight_layout()
#plt.savefig(dir_path + 'CV_BT' + '.pdf', dpi=600, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.savefig(dir_path + 'CV_BT' + '.pdf', dpi=600)
plt.close()

#Load model
BT = pickle.load(open(dir_path + 'discriminator.mdl', 'rb'))
l = BT.train_score_
#%%
lr_t = lr_best
d = 0
while lr_t < 1:
    d = d+1
    lr_t = lr_t * 10.
str_lr = str(int(lr_t)) + ' \cdot 10^{-' + str(int(d)) + '}'
#%%
fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(9), cm2inch(11))
ax.plot(np.arange(len(l)), l, c='black')
ax.set_xlim([0, len(l)])
ax.set_ylim([10, 10000])
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss (cross-entropy)')
t = ax.text(0.4*len(l), 0.55*max(l), ' Learning rate: ' + str_lr + '\n Selected IM: ' + str(IM_best), fontsize=10)
t.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='grey'))
ax.grid()
ax.set_yscale('log')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(dir_path + 'CV_BT_training' + '.pdf', dpi=600)
