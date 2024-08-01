import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
"""
This file iterates through all matrices of calcium/spike data in the folder "data", 
then resamples each row (individual neuron) to an imaging rate of 100 Hz.

This new 1d array is saved into the folder "data/resampled"
"""



states = ['test', 'train']
for stat in states:
    if stat == 'test':
        dsets = [1, 2, 3, 4, 5]
    if stat == 'train':
        dsets = [1, 2, 3, 4, 5]#, 6, 7]
        
    for dset in dsets:
        # load in true spikes
        file_path1 = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
        file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv'
        
        spikeDat = pd.read_csv(file_path2).T 
        spikeDat = np.array(spikeDat)
        
        calcDat = pd.read_csv(file_path1).T 
        calcDat = np.array(calcDat)
        mDat, nDat = np.shape(calcDat)

        #subsetAmount = np.max(np.shape(spikeDat[row,:]))
        #spikeDat = spikeDat[:, :subsetAmount]
        mSpike,nSpike = spikeDat.shape

        # set imrate depending on dset
        if dset == 1:
            imRate = 1/322.5
        if dset == 2:
            imRate = 1/11.8
        if dset in [3, 5]:
            imRate = 1/59.1
        if dset == 4:
            imRate = 1/7.8
        
        resampleRate = 1/50

        # must remove NaNs before resample or whole row becomes NaN
        i = 0
        while i < mSpike:
            naninds = np.isnan(calcDat[i,:])
            NaNpresent = np.any(naninds)
            if NaNpresent == True:
                subsetAmount = ((np.where(naninds == True))[0][0]) - 1 #index of first Nan, less one. 
            else:
                subsetAmount = np.max(np.shape(calcDat))

            # with NaNs out, pull that row and save as a new file.
        
            spikeDatSingle = spikeDat[i, :subsetAmount]
            calcDatSingle = calcDat[i, :subsetAmount]
        
            
            tFinal = imRate*subsetAmount

            newLen = tFinal/(resampleRate)
            newLen = int(np.floor(newLen))
            
            spikeDatSingle = signal.resample(spikeDatSingle, newLen)
            calcDatSingle = signal.resample(calcDatSingle, newLen)

            file_path1R = 'data/resampled/node'+ str(i) + '_dset' + str(dset) + '.' + str(stat) + '.calcium'
            file_path2R = 'data/resampled/node'+ str(i) + '_dset' + str(dset) + '.' + str(stat) + '.spikes'
            np.save(file_path1R, calcDatSingle)
            np.save(file_path2R, spikeDatSingle)

            i = i + 1 




       
