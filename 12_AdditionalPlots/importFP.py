import csv

data_path = './faultPlanes/Akatore.cnrs'

##################################################################
#isNumber: check if a string is a number
##################################################################
def isNumber(s):
	try:
		float(s)
		return True

	except ValueError:
		return False

data = []
with open(data_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for row in readCSV:
        single_line = []
        for i in range(len(row)):
            if isNumber(row[i]):
                single_line.append(float(row[i]))
        if len(single_line) > 1:
            data.append(single_line)
