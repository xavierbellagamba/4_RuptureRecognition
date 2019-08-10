import numpy as np
import matplotlib.pyplot as plt
import GMdataImportFx as gmx
from matplotlib import rc

def cm2inch(value):
    return value/2.54

#Import final training
data_f = gmx.loadCSV('./Generator/Generator_RFEncod/trainingHistory.csv')
data_t = gmx.loadCSV('./Generator/Generator_BTEncod/trainingHistory.csv')

#Range for final training
epoch_f = np.arange(1, len(data_f[0])+1)

#Plot
rc('text', usetex=True)
rc('font', family='serif')

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(14), cm2inch(8))
#s1, = plt.plot(epoch_f, data_f[2], label='Training loss', linestyle='--', color='dimgrey')
s2, = plt.plot(data_f[0], data_f[2], label='RF encoded - Validation loss', linestyle='-', color='black')
p1, = plt.plot(data_f[0][np.argmin(data_f[2])], min(data_f[2]), 'ko', markersize=12)#, facecolor='none', label='Selected model')
p2, = plt.plot(data_f[0][np.argmin(data_f[2])], min(data_f[2]), 'wo', markersize=8)#, facecolor='none', label='Selected model')
p3, = plt.plot(data_f[0][np.argmin(data_f[2])], min(data_f[2]), 'ko', markersize=4)#, facecolor='black')
t2, = plt.plot(data_t[0], data_t[2], label='GBM encoded - Validation loss', linestyle='-', color=[0.5, 0.5, 0.5])
q1, = plt.plot(data_t[0][np.argmin(data_t[2])], min(data_t[2]), 'o', markerfacecolor=[0.5, 0.5, 0.5], markeredgecolor=[0.5, 0.5, 0.5], markersize=12)#, facecolor='none', label='Selected model')
q2, = plt.plot(data_t[0][np.argmin(data_t[2])], min(data_t[2]), 'wo', markersize=8)#, facecolor='none', label='Selected model')
q3, = plt.plot(data_t[0][np.argmin(data_t[2])], min(data_t[2]), 'o', markerfacecolor=[0.5, 0.5, 0.5], markeredgecolor=[0.5, 0.5, 0.5], markersize=4)#, facecolor='black')
ax.grid()
ax.set_axisbelow(True)
ax.legend([s2, t2, (p1, p2, p3), (q1, q2, q3)] , ['RF encoded - Validation loss', 'GBM encoded - Validation loss', 'RF - Selected model', 'GBM - Selected model'], loc=1, fontsize=10)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('MSE', fontsize=14)
ax. tick_params(axis='both', labelsize=14)
ax.set_xlim([0, 63])#max(epoch_f)+1])
ax.set_ylim([0.44, .6])
#ax.set_ylim([0, 0.4])
fig.tight_layout()
fig.savefig('./Generator/training_loss.pdf', dpi=600)





