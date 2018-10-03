***********************************
Rupture recognition
***********************************

Author: X. Bellagamba
Date of last ReadMe update: 3 Oct. 2018

___________________________________
Content:
	0. Dependencies
	1. Semantics and context
	2. Framework basics
	3. How to use it
___________________________________


0. External package requirement
***********************************
python 3.6
numpy
matplotlib
sklearn
os
csv
glob
pickle
math



1. Semantics and context
***********************************
Semantics: 
Machine learning 	- 	Earthquake science
__________________________________________________
Class instance (X, y)	= 	Rupture realization (a label, a ground motion map and IM estimate at the real strong motion stations)
Multichannel 2D matrix	=	Ground motion map with multiple IMs
2D data point location	=	Station
Signal input location	=	Real strong motion station
Channel			=	Intensity measure (IM)
Label			=	Rupture name
Vector 1D		=	Collection of N_station x N_IM
Matrix 2D		= 	Ground motion map (1 IM)
Matrix 3D		= 	Ground motion map (N IM)
Ground truth data	= 	Processed recorded ground motion signal
Raw signal		=	Accelerogram

Context:
Data extracted from: 		Cybershake v18.6
Total number of faults: 	482
Considered number of faults:	116
Min considered Mw considered:	6.8
Considered number of ruptures: 	3424
Initial different label number:	3424
Available IMs:			PGA, PGV, Arias, CAV, Ds595, Ds575, MMI, 
				pSA_[0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]



2. Framework basics
***********************************
Each folder contains the code to realize 1 step of the entire process.
How to use each script is described in 3.

Step 0 - 0_WorkPrep: 
	Only used to screen faults generating less than a user-specified moment magnitude.

Step 1 - 1_DataStructure: 
	Creates 1 csv file per rupture based on the entire database. Name of the file is the Cybershake label. All IMs for all real stations of a specific rupture are contained in the file. Also creates an IM dictionary mapping IM name to int and a real ground motion station dictionary mapping the name of the station to its NZTM2000 (name of the projection) location.

Step 2 - 2_HypocentreCluster:
	Cluster the rupture from the same fault using a nested K-mean clustering algorithm. The first K-mean clustering is realized on real GM station locations. Then, within each cluster, the station showing the highest ground motion intensity variance (i.e. the most sensitive station) is picked to cluster ruptures from the same fault together. To optimize the K-mean clustering, multiple number of rupture clusters are tested. The number of station clusters is determined as: number of rupture cluster + 1. The max number of clusters tested is dependent on the magnitude (see script 2_RuptureClustering, line 42). The number of rupture clusters that obtains the lowest silhouette score is picked for relabelling, unless the number of rupture realizations of new labels are less than a user-defined value. Creates a dictionary from old (unique) labels into new labels. Can generate maps and silhouette score plots.

Step 3 - TrainingDataGeneration:
	Creates two .npy files (X.npy and y.npy) that will be used for training, based on the relabelling dictionary (Step 2) and the generated single csv files (Step 3). Allows the user to select IMs that will be stored in the X.npy file. Compute the log of the request IMs. Fill untriggered stations slots with a small constant (-8.0). Also creates a dictionary that maps used IMs into int. 

Step 4 - DiscriminatorTraining:
	Train the selected discriminator (so far a random forest) based on one user-defined IM and forest size, performing a K-fold cross-validation. The best random forest (lowest inaccuracy) is saved using pickle (not implemented yet). Also generates a plot of the K-fold cross-validation. Folders for the cross-validation are realized in a way that the training dataset will always contain all labels (i.e. at least two folders contain all the labels).

To be continued...


3. How to use it
***********************************
List all the data files that should be present and where. Gives the basic instructions to run the files.
Main script are always indicated with X_Name.py . Other *.py files are functions necessary to make it works.
Unless, structural changes are required, it is recommended to only modify the user-defined parameter box at the beginning of each main script.

Step 0 - 0_WorkPrep: 
	Required files: ./faultList.csv
	Set the minimum moment magnitude to be considered. Creates the selectedFaultList.csv .

Step 1 - 1_DataStructure: 
	Requied files: ./data/DB_IM.csv
	Set the file path if different from above. Returns as many csv files as their is different rupture realizations in ./gen . The name of the files are their unique Cybershake label. Also generates IM_dict.csv and station_dict.csv .

Step 2 - 2_HypocentreCluster:
	Required files: ./IM_dict.csv (from 1) ; ./station_dict.csv (from 1) ; selectedFaultList.csv (from 0) ; ./data/*.csv (unique rupture realization csv files from 1)
	Define the IM to be used for clustering the ruptures, the minimum number of realizations per new label, and set if figures are to be saved. Returns the old labels to new labels dictionary label_dict.csv and the map and silhouette plots if wished in ./gen .

Step 3 - TrainingDataGeneration:
	Required files: ./IM_dict.csv (from 1) ; ./station_dict.csv (from 1) ; ./label_dict.csv (from 2) ; ./data/*.csv (unique rupture realization csv files from 1)
	Provide the name of IM you wish to save in the ./gen/X.npy file. Corresponding labels are saved in the ./gen/y.npy file. Creates the ./IM_dict_train.csv file. 

Step 4 - DiscriminatorTraining:
	Required files: ./IM_dict_train.csv (from 3) ; ./data/X.npy (from 3) ; ./data/y.npy (from 3) ; ./label_dict.csv (from 2)
	Provide the number of cross-validation folder, the range defining the number of trees in the random forest and the name of the IM you wish to test (must be a saved IM from 3!).

