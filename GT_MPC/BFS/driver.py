import os
import sys
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from spikefinder_eval import _downsample




# REPRESENTATIVES:
#dset 1, neuron 0
#dset 2, neuron 0
#dset 3, neuron 1
#dset 4, neuron 1
#dset 5, neuron 1
#dsets = [1, 2, 3, 4, 5] 
#neurons = [0, 1, 1, 1, 1]

dsets = [2, 3, 4]  # axe dset 1, we're doing good there. dset 2 too large
neurons = [1, 1, 1]


# initalize parameters, step through them 

paramNum = 4 # number of choices for a particular parameter
spaceNum = 5 # number of parameters

alphaSpace = np.linspace(10, 30, paramNum)
gammaSpace = np.linspace(0.1, 2, paramNum)
kfSpace = np.linspace(0.05, 0.2, paramNum)
krSpace = np.linspace(3, 10, paramNum)
baselineSpace = np.linspace(0.5, 2, paramNum)
# append exact parameters? 

paramGrid = np.stack(np.meshgrid(*[alphaSpace, gammaSpace, kfSpace, krSpace, baselineSpace]), axis = -1).reshape(-1, spaceNum)



#print(np.shape(paramGrid))


# step through rows of paramGrid, each corresponds to a set of parameters.
for j in range(973, np.shape(paramGrid)[0]):
    alpha, gamma, kf, kr, baseLine = paramGrid[j]

    i = 0
    # step through each dset neuron pairing
    while i < len(dsets):
        dset = dsets[i]
        neuron = neurons[i]

        start  = time.time()
        print("Beginning solve on data set", str(dset) + ", neuron",  str(neuron))
            
        os.system("python3 LoopableBinnedMPCmain.py " + str(neuron) + " " + str(dset) + " " + str(alpha) + " " + str(gamma) + " " + str(kf) + " " + str(kr) + " " + str(baseLine)) # + parameters
        
        end = time.time()
        print("previous solve for neuron", i, "completed in", (end - start)/60, "minutes")
        print("------------------------------------------------------------------------")

        i = i + 1





