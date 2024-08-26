# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time


def sine(Amp, w, t):
    """
    Defines signal for which we will examine frequency response

    Arguments: 
        Amp : Amplitude
        w   : Period
        t   : time
    """
    out = Amp*np.sin(w*t) + 11
    return out



def CRN(t, A):
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI^* respectively. 
                A = [x, z]
        t : time
        p : vector of the parameters:
                p = [kf, kr, amp, w]
        s :  must be one of the impulse functions above, e.g., impulseSquare(t)
             remember to change this in plotting functions below as well. 
    """
    x, z = A
    kf, kr, amp, L, wNow = p # omega looped through in main
    
    s = sine(amp, wNow, t) + 11 # shift upwards
    

    # define chemical master equation 
    # this is the RHS of the system     
 
    du = [s - x + kr*z - kf*x*(L-z), # Ca^{2+}
        kf*x*(L-z) - kr*z]           # Ci*
    
    return du


if __name__=="__main__":

    n = 2000
    tEnd = 100
    timeVec = np.linspace(0, tEnd, n)
    tsolve = [0, tEnd]
    transientTime = 40 #for small w, transients end after ~40s. 

    # define initial conditions
    X = 0   #Ca^{2+}
    Z = 30  #CI^*

    # define constants/reaction parameters, kr << kf
    kf = 1  #0.0513514
    kr = 10 #7.6
    amp = 300    
    # total sum of calcium indicator
    L = 30
    
    # pack up parameters and ICs, looping through periods
    u0 = [X, Z]

    omegas = np.arange(0.1, 60, 0.1)
    accumed_X = np.zeros((len(omegas), len(timeVec)))
    accumed_Z = np.zeros((len(omegas), len(timeVec)))
    accumed_S = np.zeros((len(omegas), len(timeVec)))
    
    start = time.time()
    
    # simulate, stepping over w, while saving soln to other mat
    for i in range(len(omegas)):
        p = [kf, kr, amp, L, omegas[i]]
        sol = solve_ivp(CRN, tsolve, u0, t_eval=timeVec)
        accumed_X[i, :] = sol.y[0, :]
        accumed_Z[i, :] = sol.y[1, :]
        accumed_S[i, :] = sine(amp, omegas[i], timeVec)

    accumed_Y = np.zeros(np.shape(accumed_X))
    accumed_Y = L - accumed_Z

    end = time.time()
    print('Total solve time ', end - start, "(s) for ", len(omegas), "different frequencies.")
 
    # want to compute amplitude after ~ 10 seconds, as then transients have died down
    transientIndex = len(timeVec) - len(np.where(timeVec > transientTime)[0])
    # the above ensures timeVec[transientIndex] > 10
    print(timeVec[transientIndex])
    
    # for each w, compute amplitude
    ampX = np.zeros(len(omegas))
    ampZ = np.zeros(len(omegas))
    ampS = np.zeros(len(omegas)) # not need but good sanity check, should always be A_in
    ampY = np.zeros(len(omegas))

    for i in range(len(omegas)):
        maxValX = np.max(accumed_X[i, transientIndex:])
        minValX = np.min(accumed_X[i, transientIndex:])
        ampX[i] = (maxValX - minValX)/2
        
        maxValZ = np.max(accumed_Z[i, transientIndex:])
        minValZ = np.min(accumed_Z[i, transientIndex:])
        ampZ[i] = (maxValZ - minValZ)/2
        
        maxValS = np.max(accumed_S[i, transientIndex:])
        minValS = np.min(accumed_S[i, transientIndex:])
        ampS[i] = (maxValS - minValS)/2

        maxValY = np.max(accumed_Y[i, transientIndex:])
        minValY = np.min(accumed_Y[i, transientIndex:])
        ampY[i] = (maxValY - minValY)/2
        
 
    # compute mean of signal
    meanX = np.zeros(len(omegas))
    meanZ = np.zeros(len(omegas))
    for i in range(len(omegas)):
        meanX[i] = np.mean(accumed_X[i, transientIndex:])
        meanZ[i] = np.mean(accumed_Z[i, transientIndex:])


    # plot signal, response in both state variables
    fig, axs = plt.subplots(1, 4)
    omegas = np.log(omegas)
   
    # plot ratios of amplitudes
    axs[0].plot(omegas, np.log(ampX/ampS), label = r'$\frac{A(ca^{2+})}{A(s)}$')
    #axs[0].plot(omegas, np.log(ampS), label = r'A(s)')
    axs[0].set_xlabel(r'$\omega$', fontsize = 16)
    axs[0].set_ylabel(r'Log ratio of amplitudes', fontsize=14)
    axs[0].title.set_text("Calcium Ion")#, fontsize = 18)
    axs[0].legend()

    axs[1].plot(omegas, np.log(ampZ/ampS), label = r'$\frac{A(CI^*)}{A(s)}$')
    axs[1].set_xlabel(r'$\omega$', fontsize = 16)
    #axs[1].set_ylabel(r'$\frac{A(CI^*)}{A(s)}$', fontsize=16)
    axs[1].title.set_text("Calcium Indicator")#, fontsize = 18)
    axs[1].legend()
 
    axs[2].plot(omegas, np.log((ampY + ampZ)/ampS), label = r'$\frac{A(CI^*)+ A(CI)}{A(s)}$')
    axs[2].set_xlabel(r'$\omega$', fontsize = 16)
    #axs[2].set_ylabel(r'$\frac{A(CI^*)}{A(s)}$', fontsize=16)
    axs[2].title.set_text("Sum of calcium indicators")
    axs[2].legend() 

    axs[3].plot(omegas, np.log(ampX/ampS), label = r'$\frac{A(ca^{2+})}{A(s)}$')
    axs[3].plot(omegas, np.log(ampZ/ampS), label = r'$\frac{A(CI^*)}{A(s)}$')
    axs[3].plot(omegas, np.log((ampZ + ampY)/ampS), label = r'$\frac{A(CI^*)+ A(CI)}{A(s)}$')
    axs[3].plot(omegas, np.log((ampX + ampZ)/ampS), label = r'$\frac{A(Ca^{2+}) + A(CI^*)}{A(s)}$')
    axs[3].title.set_text("All Overlaid")
    axs[3].set_xlabel(r'$\omega$', fontsize = 16)
    axs[3].legend() 
    plt.tight_layout()
    
    plt.figure(2)
    #omegas = np.exp(omegas)
    plt.plot(omegas, np.log((ampX + ampZ)/ampS), label = r'$\frac{A(Ca^{2+}) + A(CI^*)}{A(s)}$')
    #plt.plot(omegas, np.log((ampZ + ampY)/ampS), label = r'$\frac{A(CI^*)+ A(CI)}{A(s)}$')
    plt.plot(omegas, np.log(ampX/ampS), label = r'$\frac{A(ca^{2+})}{A(s)}$')
    plt.plot(omegas, np.log(ampZ/ampS), label = r'$\frac{A(CI^*)}{A(s)}$')
    plt.xlabel(r'$\omega$', fontsize = 16)
    plt.ylabel(r'Ratio of amplitudes', fontsize=16)
    plt.title("Log-Log Plot of Frequency Response", fontsize = 20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize = 12)
    plt.tight_layout


    plt.figure(3)
    #omegas = np.exp(omegas)
    #plt.plot(omegas, np.log(ampX/ampS), label = r'$\frac{A(ca^{2+})}{A(s)}$')
    #plt.plot(omegas, np.log(ampZ/ampS), label = r'$\frac{A(CI^*)}{A(s)}$')
    #plt.plot(omegas, ampY/ampS, label = r'$\frac{A(CI)}{A(s)}$')
    #plt.plot(omegas, np.log((ampZ + ampY)/ampS), label = r'$\frac{A(CI^*)+ A(CI)}{A(s)}$')
    plt.plot(omegas, np.log((ampX + ampZ)/ampS), label = r'$\frac{A(Ca^{2+}) + A(CI^*)}{A(s)}$')
    plt.title("Log-Log Plot of Frequency Response")
    plt.xlabel(r'$\omega$', fontsize = 16)
    plt.ylabel(r'Ratio of amplitudes', fontsize=16)
    plt.legend() 
    plt.tight_layout()
    


    plt.show()


