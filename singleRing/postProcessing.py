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
    figMat = plt.imshow(A, extent = [0, firingTimes[-1], -np.pi, np.pi])
    plt.title('Heatmap of Firing Rate', fontsize = 20)
    plt.xlabel(r'Time (s)', fontsize = 14)
    plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
    plt.colorbar(shrink=.20)#location="bottom")
    #the below places ticks at linspace locations, then labels. Extent above determines width.
    plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    
    #plt.xticks(np.arange() 


if __name__ == "__main__":

    # plot firing rates as heatmap
    firingRate = np.load("data/firingRates.npy") 
    firingTimes = np.load("data/firingTimes.npy") # is sol.t in updatedVelocity.py
    #print(len(firingTimes))
    print(np.shape(firingRate))

    matVis(firingRate)

    #print(np.shape(firingRate))
    N = np.shape(firingRate)[0]
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
    # plot cross section
    plt.figure(2)
    times = [2000]
    colors = ['blue', 'green']
    j=0
    for i in times:
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'Firing Rate $f$ (Hz)', fontsize = 14)
        plt.title(r'Snapshot of Bump Profile', fontsize = 20) 
        plt.plot(x, firingRate[:,i], alpha=0.7, color = colors[j], label='t= ' +  str(firingTimes[i])[:5] + 's') 
        plt.xticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
        j = j + 1    
    plt.legend()


    # plot multiple cross sections
    plt.figure(3)
    times = [2000, 2400, 2600]
    colors = ['blue', 'green', 'orange']
    j = 0
    for i in times:
        plt.xlabel(r'$\theta$', fontsize = 14)
        plt.ylabel(r'Firing Rate $f$ (Hz)', fontsize = 14)
        plt.title(r'Snapshots of Bump Profile', fontsize = 20) 
        if j == 0:
            plt.plot(x, firingRate[:,i], alpha=0.7, color = colors[j], label='t= ' +  str(firingTimes[i])[:5] + 's')
        else:
            plt.plot(x, firingRate[:,i], linestyle='--', alpha=0.7, color = colors[j], label='t= ' +  str(firingTimes[i])[:5] + 's') 
        plt.xticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
        j = j + 1    
    plt.legend()


       
    # check velocity
    velSim = np.load("data/velSim.npy")
        
    file_path = "data/vRot_cond1_allFly1_stripe2.mat"
    mat = scipy.io.loadmat(file_path)
    temp = mat['vel'] # print mat to see what options
    (m, n) = temp.shape #flipped from other data
    realVel = np.zeros(m)
    realVel = temp[:, 0]
    
    file_path2 = "data/time_cond1_allFly1_stripe2.mat"
    mat = scipy.io.loadmat(file_path2)
    temp = mat['time']
    (m1, n1) = temp.shape
    timeVec = np.zeros(m1)
    timeVec = temp[:-1, 0] # chop off last time for same number of vel
    tEnd = timeVec[-1]     # this also reshapes

    # compare velocities, simulated and real, at specific time instants.
    velInterp = np.interp(timeVec, firingTimes[:-1], velSim)
    
    error =(1/len(velInterp))*(velInterp - realVel)**2
    
    print("Mean Square Error:", np.sum(error)) # velocities a bit off. 


    #  check location of bump (where firing rate is maximal)



    plt.show()
