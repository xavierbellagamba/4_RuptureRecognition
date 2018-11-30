import csv
import GMdataImportFx as gix

##################################################################
#loadCS_stationList: load the csv data file
##################################################################
def loadCS_stationList(data_path):
	dic_CS ={}
	with open(data_path) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=' ')

		for row in readCSV:
			if not len(row) == 1:
				#Input vector
				single_line = []
				for i in range(len(row)):
					if gix.isNumber(row[i]):
						single_line.append(float(row[i]))
					else:
						if not row[i] == '':
							single_line.append(row[i])
				dic_CS[single_line[2]] = gix.convertWGS2NZTM2000(single_line[0], single_line[1])
			else:
				el = row[0].split('\t')
				dic_CS[el[2]] = gix.convertWGS2NZTM2000(float(el[0]), float(el[1]))
	return dic_CS


##################################################################
#loadGM_CS: load the GM map data file (!!!Cybershake: shitty format!!!)
##################################################################
def loadGM_CS(data_path):
	M = []
	with open(data_path) as csvfile:
		readCSV = csv.reader(csvfile)

		for row in readCSV:
			#Input vector
			single_line = []
			for i in range(len(row)):
				if i == 0:
					single_line.append(row[i])
				else:
					if gix.isNumber(row[i]):
						single_line.append(float(row[i]))
					else:
						single_line.append(row[i])
			M.append(single_line)

	return M


#####################################################################################
# saveVirtStationDict: save station dictionary as csv
#####################################################################################
def saveVirtStationDict(station):
	file_str = ''
	key = list(station.keys())

	#Create station file
	for i in range(len(station)):
		file_str = file_str + key[i] + ',' + str(station[key[i]][0]) + ',' + str(station[key[i]][1])
		if i != len(station)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './virt_station_dict.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadVirtStationDict: load station dictionary
#####################################################################################
def loadVirtStationDict(file_path, reverse=False):
	#Load data
	raw_data = loadGM_CS(file_path)

	#Create dict
	station_dict = {}
	for i in range(len(raw_data)):
		station_dict.update({raw_data[i][0] : [float(raw_data[i][1]), float(raw_data[i][2])]})

	if reverse:
		station_dict = {v: k for k, v in station_dict.items()}

	return station_dict


#####################################################################################
# saveRotStationDict: save station dictionary as csv
#####################################################################################
def saveRotStationDict(station):
	file_str = ''
	key = list(station.keys())

	#Create station file
	for i in range(len(station)):
		file_str = file_str + key[i] + ',' + str(station[key[i]][0]) + ',' + str(station[key[i]][1])
		if i != len(station)-1:
			file_str = file_str + '\n'

	#Export file
	file_path = './rot_station_dict.csv'
	with open(file_path, "w") as output_file:
		print(file_str, file=output_file)
	output_file.close()


#####################################################################################
# loadVirtStationDict: load station dictionary
#####################################################################################
def loadRotStationDict(file_path, reverse=False):
	#Load data
	raw_data = loadGM_CS(file_path)

	#Create dict
	station_dict = {}
	for i in range(len(raw_data)):
		station_dict.update({raw_data[i][0] : [int(raw_data[i][1]), int(raw_data[i][2])]})

	if reverse:
		station_dict = {v: k for k, v in station_dict.items()}

	return station_dict











































