import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
import GMCluster as gml
import discr_utils as du
from matplotlib import cm
from operator import add, sub
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from scipy.interpolate import UnivariateSpline
import pickle

def cm2inch(value):
    return value/2.54

###########################################
# User-defined parameters
###########################################
#Model name
model_name = 'RF_discriminator.mdl'

#IM used by the discriminator
IM_name = ['AI']

#Dropout rate
drop_rate = [0.025, 0.05]#, 0.1]#, 0.15, 0.2, 0.25, 0.3]

#Number of tests per rupture
n = 5
###########################################

#Import data
print('Load data...')
X = np.load('./data/X_test_BT.npy')
y = np.load('./data/y_test_BT.npy')
lbl_GM = gml.loadLabelDict('label_dict.csv', reverse=True)
lbl_GM = list(lbl_GM.keys())
IM_dict = gml.loadIMDict_trainData('IM_dict_train.csv')
IM_ID = [int(IM_dict[x]) for x in IM_name]
rupt_dict = {}
for i in range(len(lbl_GM)):
    rupt_dict[lbl_GM[i]] = i

#Load discriminator
rf_load = open('./Discriminator/' + model_name, 'rb')
discr = pickle.load(rf_load)

#Without dropout rate
p_pred = np.zeros((X.shape[0]))
x_pred = np.zeros((X.shape[0]))
for i in range(X.shape[0]):
    p_pred[i] = discr.predict_proba(X[i, :, IM_ID])[0][rupt_dict[y[i]]]
    pred = discr.predict(X[i, :, IM_ID])
    if pred[0] == y[i]:
        x_pred[i] = 1.0
    
acc_full_mu = np.mean(p_pred)
pred_mu = np.sum(x_pred)/float(X.shape[0])


#%%
#Dropout rate
test = np.zeros((len(drop_rate), X.shape[0], n))
test_p = np.zeros((len(drop_rate), X.shape[0], n))
acc_rupt = np.zeros((len(drop_rate), X.shape[0]))
acc_dropout_mu = np.zeros((len(drop_rate)))
p_rupt = np.zeros((len(drop_rate), X.shape[0]))
p_dropout_mu = np.zeros((len(drop_rate)))

#For each dropout
for i in range(len(drop_rate)):
    #For each existing rupture, test the dropout
    for j in range(X.shape[0]): 
        #Create list of indices to drop
        data = X[j, :, IM_ID]
        ind_nonnull = np.where(data > -8.0)[1]
        n_2null = int(len(ind_nonnull)*drop_rate[i])
        ind_2null = np.random.choice(ind_nonnull, (n, n_2null))
        
        for k in range(n):
            data_t = data
            data_t[0, ind_2null[k]] = -8.0
            
            pred = discr.predict(data_t)
            if pred[0] == y[j]:
                test[i, j, k] = 1
                
            test_p[i, j, k] = discr.predict_proba(data_t)[0][rupt_dict[y[j]]]
            
        #Compute accuracy for each rupture
        acc_rupt[i, j] = np.sum(test[i, j, :])/float(n)
        p_rupt[i, j] = np.mean(test_p[i, j, :])

    #Compute accuracy for each rupture
    acc_dropout_mu[i] = np.mean(acc_rupt[i, :])
    p_dropout_mu[i] = np.mean(p_rupt[i, :])
            
#%%
rc('text', usetex=True)
rc('font', family='serif')
#Plot results dropout dropout
fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(18), cm2inch(10))

ax.scatter(drop_rate, acc_dropout_mu, s=50, marker='D', color='black', label='\mu', zorder=10)
ax.plot(drop_rate, p_dropout_mu, markersize=10, marker='o', color='black', fillstyle='none', linestyle='none', mew=2, label='\mu', zorder=10)
ax.plot(0.0, acc_full_mu, markersize=10, marker='o', color='black', fillstyle='none', mew=2,  label='\mu', zorder=10)
ax.scatter(0.0, pred_mu, s=50, marker='D', color='black', label='\mu', zorder=10)

ax.scatter(np.zeros(X.shape[0]), p_pred, s=5, marker='o', color='darkgrey', alpha=0.5)
iax_p = plt.axes([0, 0, 0, 0])
ip_p = InsetPosition(ax, [0.055, 0, 0.08, 1.0]) #posx, posy, width, height
iax_p.set_axes_locator(ip_p)
p, x = np.histogram(p_pred, bins=np.arange(-0.05, 1.06, 0.1)) 
x = x[:-1] + (x[1] - x[0])/2
f = UnivariateSpline(x, np.asfarray(p), s=5)
iax_p.plot(f(x), x, color='k')
iax_p.set_ylim([0, 1])
iax_p.plot([0, 0], [0, 1], color='black', linewidth=0.5)
iax_p.fill_betweenx(x, f(x), 0, alpha=0.3, color='black')
iax_p.axis('off')

iax = []
ip = []
for i in range(len(drop_rate)):
    ax.scatter(np.ones(acc_rupt.shape[1])*drop_rate[i], p_rupt[i], s=5, marker='o', color='darkgrey', alpha=0.5)
    iax.append(plt.axes([0, 0, i, 1]))
    ip.append(InsetPosition(ax, [0.055+0.16*(i+1), 0, 0.08, 1.0])) #posx, posy, width, height
    iax[i].set_axes_locator(ip[i])
    p, x = np.histogram(p_rupt[i], bins=np.arange(-0.05, 1.06, 0.1)) 
    x = x[:-1] + (x[1] - x[0])/2
    f = UnivariateSpline(x, np.asfarray(p), s=100)
    iax[i].plot(f(x), x, color='black', zorder=10)
    iax[i].set_ylim([0, 1])
    iax[i].plot([0, 0], [0, 1], color='black', linewidth=0.5, zorder=10)
    iax[i].fill_betweenx(x, f(x), 0, alpha=0.3, color='black')
    iax[i].axis('off')

ax.set_xlabel('Dropout rate')
ax.set_ylabel('Accuracy \& Assigned probability')
ax.set_ylim([0, 1])
ax.set_xlim([-0.005, 0.15])
ax.set_xticks([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
ax.grid()
ax.set_axisbelow(True)
#plt.tight_layout()











