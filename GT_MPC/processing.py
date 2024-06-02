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



def plot_correlations(factors, corrCoefs):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting with enhancements
    ax1.plot(factors, corrCoefs, marker='o', linestyle='-', color='b', markersize=8, linewidth=2, label='Correlation Coefficients')
    
    # Primary x-axis and y-axis labels
    ax1.set_title("Correlations between Simulated and Recorded Spikes", fontsize=20, fontweight='bold')
    ax1.set_xlabel("Downsampling Factor", fontsize=16)
    ax1.set_ylabel("Correlation Coefficient", fontsize=16)
    
    # Adding a grid
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding annotations for data points
    for i, txt in enumerate(corrCoefs):
        ax1.annotate(f'{txt:.2f}', (factors[i], corrCoefs[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)
    
    # Adding a legend
    ax1.legend(fontsize=14)

    # Secondary x-axis for bin width
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())  # Make sure the limits match
    bin_widths = [40, 83.0, 167.0, 333.0]  # Specific bin width values
    ax2.set_xticks(factors)
    ax2.set_xticklabels([f'{bw:.1f}' for bw in bin_widths])
    ax2.set_xlabel("Bin width (ms)", fontsize=20)
   
    # Adjusting layout to fit elements
    fig.tight_layout()
    
    # Showing the plot
    plt.show()
    
    # Printing the correlation coefficients
    print(corrCoefs)


if __name__=="__main__":
    # load in actual truth data
    dset = 1
    row = 0
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

    factors = [4, 8, 16, 32]
    corrCoefs = np.zeros(np.shape(factors))
    print(corrCoefs)
    print(factors)
    for i in range(len(factors)):
        factor = factors[i]
        spikeDatDown = _downsample(spikeDatRaw, factor)
        simSpikeDown = _downsample(simSpikesRaw, factor)
        corrCoefs[i] = np.corrcoef(spikeDatDown, simSpikeDown)[0, 1] # toss first 200 time instants, contains bad transients.
        
    plot_correlations(factors, corrCoefs)
    

    #plt.plot(factors, corrCoefs)
    #plt.title("Correlations between simulated and recorded spikes", fontsize = 18)
    #plt.xlabel("Downsampling factor", fontsize = 14)
    #plt.ylabel("Correlation coefficient", fontsize = 14)
    #plt.tight_layout()
    #plt.show()
    #print(corrCoefs)

