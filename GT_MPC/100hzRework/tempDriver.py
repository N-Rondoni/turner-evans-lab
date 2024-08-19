import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from spikefinder_eval import _downsample


dsets = [1, 2, 3, 4, 5]
#dsets = [5]
#dsets = [2, 4]
#dsets = [4, 2, 3, 5, 1]
status = ['train'] # did 'test' already

#dsets = [9]
for stat in status:
    if stat == 'train':
        dsets = [8, 9, 10]

    for dset in dsets:
        file_path = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
        data1 = pd.read_csv(file_path).T 
        data1 = np.array(data1)
        mDat, nDat = np.shape(data1)
        # loop through rows, each corresponds to a neuron. 
        i = 0
        while i < mDat:
            if i in [5, 6, 7, 8, 12, 14, 15, 16, 17, 20]:
                if dset == 8:
                    i = i+1
                    continue
            if i in [1, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                if dset == 9:
                    i = i+1
                    continue

            start  = time.time()
            print("Beginning solve on", str(stat) ,"data set", str(dset) + ", neuron",  i)
            
            # check for NaNs
            naninds = np.isnan(data1[i,:])
            #if the below evaluates to true, there are NaNs in the dataset.
            NaNpresent = np.any(naninds)

            if NaNpresent == True:
                print("This neuron's data contains NaNs! Solving up until NaNs begin... ") 
                
            # run solver, if NaNs present main will simulate up until they begin. 
            os.system("python3 MPCmain.py " + str(i) + " " + str(dset) + " " + str(stat))
            end = time.time()
            print("previous solve for neuron", i, "completed in", (end - start)/60, "minutes")
            print("------------------------------------------------------------------------")
            i = i + 1







#for dset in dsets:
    #file_path = 'data/' + str(dset) + '.test.calcium.csv'
    #    print(dset)
#    print(m, n)
    #os.system("python3 test.py")

