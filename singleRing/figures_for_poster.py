import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import scipy.io
import pingouin as pg
import os

# Tired of the runtime to generate firingRates, data was saved
# and is now plotted here. Run updatedVelocity.py to generate.



def matVis(A):
    fig = plt.figure()
    figMat = plt.imshow(A, aspect='auto', extent = [0, firingTimes[-1], -np.pi, np.pi])
    plt.title('Heatmap of Firing Rate', fontsize = 22)
    plt.xlabel(r'Time (s)', fontsize = 16)
    plt.ylabel(r'Hd Cell $\theta$', fontsize = 16)
    plt.colorbar(figMat)



if __name__ == "__main__":

    # plot firing rates as heatmap
    firingRate = np.load("data/firingRates.npy")
    firingTimes = np.load("data/firingTimes.npy")
    print(len(firingTimes))
    matVis(firingRate)

    print(np.shape(firingRate))
    N = np.shape(firingRate)[0]
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    # plot cross sections
    for i in [50, 500, 1000, 1500]:
        print(i)
        plt.figure(i)
        plt.plot(x, firingRate[:,i])

    plt.show()
