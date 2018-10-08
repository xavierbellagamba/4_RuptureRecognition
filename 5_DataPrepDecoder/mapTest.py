import numpy as np
import prepDx as pdx
import matplotlib.pyplot as plt

station = pdx.loadVirtStationDict('virt_station_dict.csv')
station = np.asarray(list(station.values()))

plt.scatter(station[:, 0], station[:, 1])