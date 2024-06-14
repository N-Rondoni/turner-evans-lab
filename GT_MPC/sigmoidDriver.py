import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from spikefinder_eval import _downsample


dsets = [1,2, 3, 4, 5] #dset 1, row 1 has NaN
#dsets = [5]
#dsets = [2, 4]

for dset in dsets:
    file_path = 'data/' + str(dset) + '.test.calcium.csv'
    data1 = pd.read_csv(file_path).T 
    data1 = np.array(data1)
    mDat, nDat = np.shape(data1)
    # loop through rows, each corresponds to a neuron. 
    i = 0
    while i < mDat:
        start  = time.time()
        print("Beginning solve on data set", str(dset) + ", neuron",  i)
        
        # check for NaNs
        naninds = np.isnan(data1[i,:])
        #if the below evaluates to true, there are NaNs in the dataset.
        NaNpresent = np.any(naninds)

        if NaNpresent == True:
            print("This neuron's data contains NaNs! Solving up until NaNs begin... ") 
            
        # run solver, if NaNs present main will simulate up until they begin. 
        os.system("python3 sigmoidLoopableBinnedMPCmain.py " + str(i) + " " + str(dset))
        end = time.time()
        print("previous solve for neuron", i, "completed in", (end - start)/60, "minutes")
        print("------------------------------------------------------------------------")
        i = i + 1







#for dset in dsets:
    #file_path = 'data/' + str(dset) + '.test.calcium.csv'
    #    print(dset)
#    print(m, n)
    #os.system("python3 test.py")

