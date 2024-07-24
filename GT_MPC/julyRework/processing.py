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
import seaborn as sns
from datetime import date
import spikefinder_eval as se
from spikefinder_eval import _downsample

save = False
ftype = "png"

def plotCorrelations(factors, corrCoefs, neuron, dset):
    plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting with enhancements
    ax1.plot(factors, corrCoefs, marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label='Correlation Coefficients')
    
    # Primary x-axis and y-axis labels
    ax1.set_title("Correlations between Simulated and Recorded Spikes", fontsize=20, fontweight='bold')
    ax1.set_xlabel("Downsampling Factor", fontsize=16)
    ax1.set_ylabel("Correlation Coefficient", fontsize=16)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding annotations for data points
    for i, txt in enumerate(corrCoefs):
        ax1.annotate(f'{txt:.2f}', (factors[i], corrCoefs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
    ax1.legend(fontsize=14)

    # Secondary x-axis for bin width
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Make sure the limits match
    bin_widths = [40, 83.0, 167.0, 333.0]  # Specific bin width values
    ax2.set_xticks(factors)
    ax2.set_xticklabels([f'{bw:.1f}' for bw in bin_widths])
    ax2.set_xlabel("Bin width (ms)", fontsize=20)
   
    fig.tight_layout()
    #plt.show()

    print('dset:', dset, 'neuron:', neuron, "corr:", corrCoefs[0])
    if save == True:
        filename = 'CorrCoef_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
        os.system('cp ' + filename + '.' + ftype + ' /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
        os.system('rm ' + filename + '.' + ftype)
            

def plotSignalsSubset(t, simSignal, trueSignal, sStart, sStop, neuron, dset):
    plt.figure()
    plt.plot(t[sStart:sStop], simSignal[sStart:sStop], label=r'Simulated Rate')
    plt.plot(t[sStart:sStop], trueSignal[sStart:sStop], label="Recorded Spike Rate", alpha = 0.8)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.title("Subset of Expected and Recorded Spikes, dataset " + str(dset) + " neuron " + str(neuron))
    plt.legend()
    
    if save == True:
        filename = 'Spikes_subset_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
        os.system('cp ' + filename + '.' + ftype + ' /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
        os.system('rm ' + filename + '.' + ftype)
    #os.system('cp ' + filename + '.png /mnt/c/Users/nicho/Pictures/Gt_sim')#/dset' + str(dset) +'/neuron' + str(neuron)) this line saves loose into the folder 

def plotSignals(t, simSignal, trueSignal, neuron, dset):
    plt.figure()
    plt.plot(t, simSignal, label=r'Simulated Rate')
    plt.plot(t, trueSignal, label="Recorded Spike Rate", alpha = 0.8)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.title("Expected and Recorded Spikes")#, bin size of " + str(1000*binSizeTime) + " ms")
    plt.legend()

    if save == True:
        filename = 'Spikes_fullSolve_dset'+ str(dset) + "_neuron" + str(neuron)
        plt.savefig(filename + '.' + ftype, format = ftype)
        os.system('cp ' + filename + '.' + ftype + ' /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
        os.system('rm ' + filename + '.' + ftype)
    #


def NaNChecker(dset, row, stat):
    # check for NaNs in calcium dataset
    file_path = 'data/' + str(dset) + '.' + str(stat) + '.calcium.csv'
    #file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv'

    data1 = pd.read_csv(file_path).T 
    data1 = np.array(data1)
    mDat, nDat = np.shape(data1)
    # loop through rows, each corresponds to a neuron. 
    
    # check for NaNs
    naninds = np.isnan(data1[row,:])
    #if the below evaluates to true, there are NaNs in the dataset.
    NaNpresent = np.any(naninds)

    return NaNpresent 



if __name__=="__main__":
   
    # load in actual truth data
    states = ['test', 'train']
    tempSum = 0
    counter = 0
    downsampledCorScor = []
    for stat in states:
        dsets = [1, 2, 3, 4, 5]
        #tempSum = 0
        #counter = 0
        for dset in dsets:
            # load in true spikes
            file_path2 = 'data/' + str(dset) + '.' + str(stat) + '.spikes.csv'
            spikeDat = pd.read_csv(file_path2).T 
            spikeDat = np.array(spikeDat)

            #subsetAmount = np.max(np.shape(spikeDat[row,:]))
            #spikeDat = spikeDat[:, :subsetAmount]
            mSpike,nSpike = spikeDat.shape

            # set imrate depending on dset
            #if dset == 1:
            #    imRate = 1/322.5
            #if dset == 2:
            #    imRate = 1/11.8
            #if dset in [3, 5]:
            #    imRate = 1/59.1
            #if dset == 4:
            #    imRate = 1/7.8
            imRate = 1/59.1
        
            i = 0
            while i < mSpike:
                # if NaNs in calcium dataset, ignore and step to the next.
                if NaNChecker(dset, i, stat) == True:
                    naninds = np.isnan(spikeDat[i,:])
                    #if dset == 1:
                        #if stat == 'train': # This is the new shit that is tossing hella errors
                            #print(spikeDat[i,:])
                            #i = i+1
                            #continue
                    subsetAmount = ((np.where(naninds == True))[0][0]) - 1 #index of first Nan, less one. 
                else:
                    subsetAmount = np.max(np.shape(spikeDat[i,:]))

                spikeDatRaw = spikeDat[i, :subsetAmount]

                simSpikesRaw = np.load('data/'+ str(stat) + '_s_node_'+ str(i) + 'dset_' + str(dset) + '.npy')
                simSpikesRaw = np.ndarray.flatten(simSpikesRaw)        
                n = np.max(np.shape(spikeDatRaw))
                finalTime = n*(imRate)
                

                # scale firing rate down so we can see what is happening. Avoid transients, hard to see.
                simSpikesRaw = (np.max(spikeDatRaw[200:])/np.max(simSpikesRaw[200:]))*simSpikesRaw # correlation coeff. invariant wrt scaling.     
                
                #simSpikesRaw = np.round(simSpikesRaw)
               

                # create corr coeff
                factors = [4, 8, 16, 32]
                corrCoefs = np.zeros(np.shape(factors))
                for j in range(len(factors)):
                    factor = factors[j]
                    spikeDatDown = _downsample(spikeDatRaw, factor)
                    simSpikeDown = _downsample(simSpikesRaw, factor)
                    corrCoefs[j] = np.corrcoef(spikeDatDown, simSpikeDown)[0, 1] # toss first 200 time instants, contains bad transients.
                    #corCoefSub = print(dset, i, np.corrcoef(spikeDatDown[200:400], simSpikeDown[200:400])[0, 1] )
                    if j == 0:
                        tempSum = tempSum + corrCoefs[0]
                        counter = counter + 1   
                # set up time to match, note final time is still computed with undownsampled n. Only use this time Vec for testing to be safe.
                n1 = min([len(spikeDatDown), len(simSpikeDown)])
                t_down = np.linspace(0, finalTime, n1)
                timeVec = np.linspace(0, finalTime, n)
                neuron = i
               
                

                # finally call plot functions
                plotCorrelations(factors, corrCoefs, neuron, dset) 
                downsampledCorScor = np.append(downsampledCorScor, corrCoefs[0])

                #plotSignals(t_down[50:], simSpikeDown[50:], spikeDatDown[50:], neuron, dset) # THESE ARE DOWNSAMPLES VALUES
                #subStart, subStop = 200, 400
                #plotSignalsSubset(t_f, simSpikeDown, spikeDatDown, subStart, subStop, neuron, dset) # UNCOMMENT TO PLOT DOWNSAMPLED VALUES
                subStart, subStop = 2000, 4000
                
                plotSignals(timeVec, simSpikesRaw, spikeDatRaw, neuron, dset)
                #plotSignalsSubset(timeVec, simSpikesRaw, spikeDatRaw, subStart, subStop, neuron, dset)

                #print(np.shape(t_f[subStart:subStop]), np.shape(simSpikeDown[subStart:subStop]), np.shape(spikeDatDown[subStart:subStop]))
                #print(np.shape(t_f), np.shape(simSpikeDown), np.shape(spikeDatDown))

                i = i + 1 

        print("average:", tempSum/counter)
    print(downsampledCorScor)
    np.save("allScores", downsampledCorScor)

    #plt.show()

