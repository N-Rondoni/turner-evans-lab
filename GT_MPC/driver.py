import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from spikeCounter import spikeCounter
from spikefinder_eval import _downsample


dsets = [1, 5] #dset 1, row 1? has NaN

dsets = [5]
for dset in dsets:
    file_path = 'data/' + str(dset) + '.test.calcium.csv'
    data1 = pd.read_csv(file_path).T 
    data1 = np.array(data1)
    mDat, nDat = np.shape(data1)
    #subsetAmount = np.max(np.shape(data1[row,:])) # the way its set up, must be divisble by factor or stuff breaks.     
    #CI_Meas = data1[row, :subsetAmount]
    for i in range(mDat):
        start  = time.time()
        print("Beginning solve on data set", str(dset), "neuron",  i)
        os.system("python3 LoopableBinnedMPCmain.py " + str(i) + " " + str(dset))
        end = time.time()
        print("previous solve for neuron", i,"completed in", (end - start), "seconds")








#for dset in dsets:
    #file_path = 'data/' + str(dset) + '.test.calcium.csv'
    #    print(dset)
#    print(m, n)
    #os.system("python3 test.py")

