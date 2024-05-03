import numpy as np
import pandas as pd
import os
import sys


def spikeCounter(data, binSize):
    """
    Takes: spike data (binary time series) and computes firing rate as a function of binSize (how many timesteps). 
    This assumes raw spike data was sampled at a rate of 59.1hz (specific to GCaMP) 
    Returns: time series data listing firing rate.
    """
    n = len(data)
    tEnd = n*(1/59.1) 
    timeVec = np.linspace(0, tEnd, n)
    timeBinSize = timeVec[binSize] 
    print("Computing firing rate with ", 1000*timeBinSize, "ms time bins")
   
    firingRate = np.zeros(n)
    for i in range(binSize, n):
        firingRate[i] = np.sum(data[i-binSize:i])

    return firingRate


if __name__ == "__main__":
    # to test, load in spike dat. 
    subsetAmount = 200
    row = 1

    file_path2 = 'data/5.test.spikes.csv'
    spikeDat = pd.read_csv(file_path2).T #had to figure out why data class is flyDat from print(mat). No clue. 
    spikeDat = np.array(spikeDat)
    spikeDat = spikeDat[:, :subsetAmount]
    mSpike,nSpike = spikeDat.shape

    spikeDat = spikeDat[row, :]
    spikeCounter(spikeDat, 3)

