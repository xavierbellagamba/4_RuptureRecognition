import numpy as np
import GMdataImportFx as gmx
import matplotlib.pyplot as plt
import csv

###########################################
# User-defined parameters
###########################################
#Min moment magnitude considered
Mw_min = 6.8
###########################################


faultList = gmx.loadCSV('./faultList.csv')

Mw = [x[1] for x in faultList]


for i in range(len(Mw)):
    if Mw[i]%0.01 > 0.0:
        Mw[i] = Mw[i]-0.2
        
#Empirical CDF
CDF = []
step = 0.05
Mw_range = np.arange(5.0, 8.0, step)
for i in range(len(Mw_range)):
    accu = 0
    for j in range(len(Mw)):
        if Mw[j] < Mw_range[i]:
            accu = accu + 1
            
    CDF.append(accu)
    

plt.figure(1)
plt.plot(Mw_range, CDF)
plt.grid()
plt.xlabel('Mw')
plt.ylabel('Cumulative number of faults')

plt.figure(2)
plt.plot(Mw_range, CDF)
plt.grid()
plt.xlabel('Mw')
plt.ylabel('Cumulative number of faults')
plt.xlim([6.4, 8.0])
plt.ylim([200, 500])

selected_faults = []
for i in range(len(Mw)):
    if Mw[i] >= Mw_min:
        selected_faults.append([faultList[i][0], Mw[i]])
        
file_path = './selectedFault_list.csv'
with open(file_path, "w") as f:
    writer = csv.writer(f)
    writer.writerows(selected_faults)
f.close()
        
        
        
        
        

