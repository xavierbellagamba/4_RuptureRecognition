import pandas as pd

loc = pd.read_csv('virt_station_dict.csv', header=None)

station_list = pd.read_csv('rot_station_dict.csv', header=None)

dict_st = {}
for i in range(len(loc)):
    dict_st[loc[0][i]] = [loc[1][i], loc[2][i]]
    
lat = []
lon = []
for i in range(len(station_list)):
    lat.append(dict_st[station_list[0][i]][0])
    lon.append(dict_st[station_list[0][i]][1])
    
station_list = station_list.assign(lat=lat)
station_list = station_list.assign(lon=lon)

station_list.to_csv('virt_station_loc.csv')