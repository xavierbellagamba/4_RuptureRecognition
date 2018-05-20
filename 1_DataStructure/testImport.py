import numpy as np
import GMdataImportFx as gix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

file_str = './gen/AlpineF2K_HYP12-21_S1574.csv'

[GM, rup, hyp] = gix.importGMRecord(file_str)

px = []
py = []
pz = []
for i in range(225):
    for j in range(675):
        px.append(float(i))
        py.append(float(j))
        pz.append(GM[i, j, 0])

px = np.asarray(px)
py = np.asarray(py)
pz = np.asarray(pz)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(px, py, pz, cmap=cm.coolwarm)
