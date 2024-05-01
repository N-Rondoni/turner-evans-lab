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
 
    du = [s - x + kr*z - kf*x*(L-z), # + beta 
        kf*x*(L-z) - kr*z] 
    
    return du


if __name__=="__main__":

    n = 2000
    tEnd = 100
    timeVec = np.linspace(0, tEnd, n)
    tsolve = [0, tEnd]
    transientTime = 49 #for small w, transients end after ~40s. 

    # define initial conditions
    X = 0 #Ca^{2+}
    Z = 50  #CI^* #was previously 0, real data readouts start with some concentration.

    # define constants/reaction parameters, kr << kf
    kf = 0.0513514
    kr = 7.6
    amp = 10    
    # total sum of calcium indicator
    L = 100
    
    # pack up parameters and ICs, looping through periods
    u0 = [X, Z]

    omegas = np.arange(0, 100, 0.1)
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


    end = time.time()
    print('Total solve time ', end - start, "(s) for ", len(omegas), "different frequencies.")
 
    # want to compute amplitude after ~ 10 seconds, as then transients have died down
    transientIndex = len(timeVec) - len(np.where(timeVec > transientTime)[0])
    # the above ensures timeVec[transientIndex] > 10

    # for each w, compute amplitude
    ampX = np.zeros(len(omegas))
    ampZ = np.zeros(len(omegas))
    ampS = np.zeros(len(omegas)) # not need but good sanity check, should always be A_in


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
 
   
    # plot signal, response in both state variables
    fig, axs = plt.subplots(1, 3)
    

#    for i in range(len(sol.t)):
#        pulse[i] = impulseRamp(sol.t[i])
#    axs[0].plot(sol.t, pulse)


    # plot ratios of amplitudes
#    axs[0].plot(omegas, ampS/ampX)
#    axs[0].set_xlabel(r'Period $w$', fontsize = 16)
#    axs[0].set_ylabel(r'$\frac{A(s)}{A(ca^{2+})}$', fontsize=16)
    
#    axs[1].plot(omegas, ampS/ampZ)
#    axs[1].set_xlabel(r'Period $w$', fontsize = 16)
#    axs[1].set_ylabel(r'$\frac{A(s)}{A(CI^*)}$', fontsize=16)
#    plt.title("Frequency Response")
    
    # plot ratios of amplitudes
    axs[0].plot(omegas, ampX/10)
    axs[0].set_xlabel(r'$w$', fontsize = 16)
    axs[0].set_ylabel(r'$\frac{A(ca^{2+})}{A(s)}$', fontsize=16)
    axs[0].title.set_text("Calcium Ion")#, fontsize = 18)

    axs[1].plot(omegas, ampZ/10)
    axs[1].set_xlabel(r'$w$', fontsize = 16)
    axs[1].set_ylabel(r'$\frac{A(CI^*)}{A(s)}$', fontsize=16)
    axs[1].title.set_text("Calcium Indicator")#, fontsize = 18)

    axs[2].plot(omegas, ampZ/10, label = r'$\frac{A(ca^{2+})}{A(s)}$')
    axs[2].plot(omegas, ampX/10, label = r'$\frac{A(CI^*)}{A(s)}$')
    axs[2].legend()
    axs[2].title.set_text("Both Plots Overlaid")
    axs[2].set_xlabel(r'$w$', fontsize = 16)

    plt.show()


