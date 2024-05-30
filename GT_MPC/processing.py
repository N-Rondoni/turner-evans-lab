import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time
import do_mpc
from casadi import *
import pandas as pd
from spikeCounter import spikeCounter
import seaborn as sns
from datetime import date
import spikefinder_eval as se
from spikefinder_eval import _downsample

   


if __name__=="__main__":
    # load in actual truth data
    dset = 5
    row = 1
    file_path2 = 'data/' + str(dset) + '.test.spikes.csv'
    spikeDat = pd.read_csv(file_path2).T #had to figure out why data class is flyDat from print(mat). No clue. 
    spikeDat = np.array(spikeDat)
    subsetAmount = np.max(np.shape(spikeDat[row,:]))
    spikeDat = spikeDat[:, :subsetAmount]
    mSpike,nSpike = spikeDat.shape
    spikeDatRaw = spikeDat[row, :]
    

    simSpikesRaw = np.load('data/s_node_'+ str(row) + 'dset_' + str(dset) + '.npy')
    simSpikesRaw = np.ndarray.flatten(simSpikesRaw)
        
    n = np.max(np.shape(spikeDatRaw))
    finalTime = n*(1/59.1)

    factors = np.arange(4, 32+1, 4)
    corrCoefs = np.zeros(np.shape(factors))
    print(corrCoefs)
    print(factors)
    for i in range(len(factors)):
        factor = factors[i]
        spikeDatDown = _downsample(spikeDatRaw, factor)
        simSpikeDown = _downsample(simSpikesRaw, factor)
        corrCoefs[i] = np.corrcoef(spikeDatDown, simSpikeDown)[0, 1] # toss first 200 time instants, contains bad transients.

    
    plt.plot(factors, corrCoefs)
    plt.show()
    print(corrCoefs)
    


