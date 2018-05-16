import GMdataImportFx as gix

#Import data
#North island
NI_raw = gix.loadCSV('./data/raw/NI.csv', isInput=False, row_ignore=1, col_ignore=0)

#South island
SI_raw = gix.loadCSV('./data/raw/SI.csv', isInput=False, row_ignore=1, col_ignore=0)

data = []
for i in range(len(NI_raw)):
    data.append(NI_raw[i])

for i in range(len(SI_raw)):
    data.append(SI_raw[i])
    
del SI_raw, NI_raw
    
#Convert station location
for i in range(len(data)):
    data[i][1:3] = gix.convertWGS2NZTM2000(data[i][1], data[i][2])

#Rotate locations
#Christchurch cathedral
loc_chch = gix.convertWGS2NZTM2000(172.6376, -43.5309)
for i in range(len(data)):
    data[i][1:3] = gix.rotateCoordinate(loc_chch, data[i][1:3], 45.0)

#Identify unique ruptures

#Identify unique station
    
#Create uniform grid
    
#Assign station to grid cell
    
#Create rupture files containing actiated stations and labels



