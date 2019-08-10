import GMdataImportFx as gix
import GMCluster as gcl
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from sklearn.metrics import silhouette_score

def cm2inch(value):
	return value/2.54


###########################################
# User-defined parameters
###########################################

###########################################

#Import faultList
fault = gix.loadCSV('selectedFault_list.csv')

#Load station dict
lbl_dict = gcl.loadLabelDict('label_dict.csv')


lbl = list(set(lbl_dict.values()))
n_rupt = len(lbl)
n_fault = len(fault)
count = np.zeros(n_fault)
Mw = []

for i in range(n_fault):
    fault[i][0] = fault[i][0] + '_'
    
for i in range(n_rupt):
    lbl[i] = lbl[i][:-1]

for i in range(n_fault):
    Mw.append(fault[i][1])
    for j in range(n_rupt):
            if fault[i][0] == lbl[j]:
                count[i] = count[i] + 1
                
for i in range(n_fault):
    if Mw[i] < 7.5 and count[i] == 5:
        print(fault[i][0] + '\t' + str(fault[i][1]) + '\t' + str(count[i]))
#%%
rc('text', usetex=True)
rc('font', family='serif', size=13)

fig, ax = plt.subplots()
fig.set_size_inches(cm2inch(13), cm2inch(8))
ax = sns.swarmplot(x=count, y=Mw, color=[0.3, 0.3, 0.3])
#ax.annotate('Cw4Swedge411', (-0.35, 7.23), fontsize=9)
#ax.annotate('Pahaua', (0.55, 7.63), fontsize=9)
#ax.annotate('AlpineF2K', (1.29, 7.61), fontsize=9)
#ax.arrow(1.5, 7.55, 0.35, 0.07, head_width=0.02, head_length=0.05, fc=[0.3, 0.3, 0.3], ec=[0.3, 0.3, 0.3], linewidth=0.4)
#ax.arrow(2.1, 7.55, -0.06, 0.05, head_width=0.02, head_length=0.035, fc=[0.3, 0.3, 0.3], ec=[0.3, 0.3, 0.3], linewidth=0.4)
#ax.annotate('OpouaweUruti', (1.75, 7.5), fontsize=9)
#ax.annotate('WairarapNich', (2.22, 7.64), fontsize=9)
#ax.annotate('AwatNEVer', (3.05, 7.07), fontsize=9)
#ax.annotate('HopeConway', (3.06, 6.88), fontsize=9)
ax.grid()
ax.set_axisbelow(True)
ax.set_xlabel('Number of clusters')
ax.set_xticks([0, 1])
ax.set_xticklabels(['1', '2'])
ax.set_ylabel('M_{w}')
plt.tight_layout()
plt.savefig('swarm_cluster.pdf', dpi=600)












