import numpy as np
import csv
import math

##################################################################
#isNumber: check if a string is a number
##################################################################
def isNumber(s):
	try:
		float(s)
		return True

	except ValueError:
		return False


##################################################################
#loadCSV: load the csv data file
##################################################################
def loadCSV(data_path, row_ignore=0, col_ignore=0, isInput=False, isCategorical=False):
	if not isInput:
		M = []
		with open(data_path) as csvfile:
			readCSV = csv.reader(csvfile)

			#Skip header
			for i in range(row_ignore):
				next(csvfile)

			for row in readCSV:
				#Input vector
				single_line = []
				for i in range(col_ignore, len(row)):
					if isNumber(row[i]):
						single_line.append(float(row[i]))
					else:
						single_line.append(row[i])
				M.append(single_line)

		return M

	#input: last column is the output
	elif isInput:
		v_input = []
		v_output = []

		with open(data_path) as csvfile:
			readCSV = csv.reader(csvfile)

			#Skip header
			for i in range(row_ignore):
				next(csvfile)

			for row in readCSV:
				#Input vector
				single_input = []
				for i in range(col_ignore, len(row)-1):
					single_input.append(float(row[i]))
				v_input.append(single_input)
				if isCategorical == True:
					v_output.append(row[-1])
				else:
					if isNumber(row[-1]):
						v_output.append(float(row[-1]))
					else:
						v_output.append(row[-1])
		csvfile.close()

	return [v_input, v_output]


#####################################################################################
# convertWGS2NZTM2000: convert WGS1984 coordinates into projected NZTM2000 cartesian
#####################################################################################
def convertWGS2NZTM2000(longitude, latitude):
	#Constants
	latitude_0 = 0.0
	longitude_0 = np.radians(173.0)
	E_0 = 1600000.0 #X origin
	N_0 = 10000000.0 #Y origin
	k_0 = 0.9996 #Central meridian scaling factor
	a = 6378137.0 #Semi-major axis
	f = 1.0 / 298.257222101 #Inverse flattening

	#Radian
	longitude = np.radians(longitude)
	latitude = np.radians(latitude)

	#Semi-minor axis
	b = a*(1.-f)

	#Eccentricity
	e2 = 2.*f - f**2.

	#Trigo values
	sinLat = np.sin(latitude)
	cosLat = np.cos(latitude)

	#Meridian distance
	A_0 = 1. - (e2)/4. - (3.*e2**2.)/64. - (5.*e2**3.)/256.
	A_2 = 3./8. * (e2 + (e2**2.)/4. + (15.*e2**3.)/128.)
	A_4 = 15./256. * (e2**2. + (3.*e2**3.)/4.)
	A_6 = (35.*e2**3.)/3072.
	m = a * (A_0*latitude - A_2*np.sin(2.*latitude) + A_4*np.sin(4.*latitude) - A_6*np.sin(6.*latitude))
	m_0 = a * (A_0*latitude_0 - A_2*np.sin(2.*latitude_0) + A_4*np.sin(4.*latitude_0) - A_6*np.sin(6.*latitude_0))

	#Radius of curvature
	rho = (a*(1.-e2))/((1.-e2*(sinLat**2.))**1.5)
	nu = a/((1.-e2*(sinLat**2.))**0.5)
	Psi = nu / rho

	#Projection
	t = np.tan(latitude)
	omega = longitude-longitude_0

	E1 = ((omega**2.)/6.) * (cosLat**2.) * (Psi-(t**2.))
	E2 = ((omega**4.)/120.) * (cosLat**4.) * (4.*(Psi**3.)*(1.-(6.*t**2.)) + (Psi**2.)*(1.+8.*(t**2.)) - Psi*2.*(t**2.) + t**4.)
	E3 = ((omega**6.)/5040.) * (cosLat**6.) * (61. - 479.*(t**2.) + 179.*(t**4.) - t**6.)
	E_prime = k_0 * nu * omega * cosLat * (1.+E1+E2+E3)
	E = E_0 + E_prime

	N1 = ((omega**2.)/2.) * nu * sinLat * cosLat
	N2 = ((omega**4.)/24.) * nu * sinLat * (cosLat**3.) * (4.*(Psi**2.) + Psi - t**2.)
	N3 = ((omega**6.)/720.) * nu * sinLat * (cosLat**5.) * (8.*(Psi**4.)*(11.-24.*(t**2.)) - 28.*(Psi**3.)*(1.-6.*(t**2.)) + (Psi**2.)*(1.-32.*(t**2.)) - Psi*(2.*(t**2.)) + t**4.)
	N4 = ((omega**8.)/40320.) * nu * sinLat * (cosLat**7.) * (1385. - 3111.*(t**2.) + 543.*(t**4.) - t**6.)
	N_prime = k_0 * (m-m_0+N1+N2+N3+N4)
	N = N_0 + N_prime

	return [E, N]


#####################################################################################
# rotateCoordinate: rotate the coordinate around chosen point
#####################################################################################
def rotateCoordinate(pivot, coord, angle):
	#Degree to radian
	angle_rad = math.radians(angle % 360)

	#Translation
	new_coord = [coord[0] - pivot[0], coord[1] - pivot[1]]

	#Rotation
	new_coord = [new_coord[0] * math.cos(angle_rad) - new_coord[1] * math.sin(angle_rad), 
		new_coord[0] * math.sin(angle_rad) + new_coord[1] * math.cos(angle_rad)]

	#Reverse translation
	new_coord = [new_coord[0] + pivot[0], new_coord[1] + pivot[1]]

	return new_coord










