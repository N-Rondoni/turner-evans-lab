# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time

def impulseSine(t):
    T = 3
    A = 10
    signal = A*np.abs(np.sin(np.pi*t/T))
    return signal


def impulseSaw(t):
    # can only be called on a single element
    A = 1
    T = 5
    if t % T < T/2:        
        signal = 2*A*t/T
    if t % T >= T/2:
        signal = A - 2*A*t/T
    return signal

#def impulseSquare(t):
#    T = 5


def CRN(t, A):
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI^* respectively. 
                A = [x, z]
        t : time
        p : vector of the parameters:
                p = [kf, kr, alpha, gamma]
    """
    x, z = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, L = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    s = impulseSine(t)

    du = [alpha*s - gamma*x + kr*z - kf*x*(L-z), # + beta 
        kf*x*(L-z) - kr*z] 
    
    return du


if __name__=="__main__":

    n = 1000
    imRate = 11.4
    tEnd = n*(1/imRate)
    tEnd = 20
    timeVec = np.linspace(0, tEnd, n)
    tsolve = [0, tEnd]

    print(tEnd)

    # define initial conditions
    X = 0 #Ca^{2+}
    Z = 50  #CI^* #was previously 0, real data readouts start with some concentration.

    # define constants/reaction parameters, kr << kf
    kf = 0.0513514
    kr = 7.6
    alpha = 1
    gamma = 1    
    # total sum of calcium indicator
    L = 100
    
    # pack up parameters and ICs
    p = [kf, kr, alpha, gamma, L]
    u0 = [X, Z]

    start = time.time()
    sol = solve_ivp(CRN, tsolve, u0, t_eval=timeVec)
    end = time.time()

    print('Total solve time ', end - start)
    
    print(sol.y.shape)
  
    # plot signal, response in both state variables
    fig, axs = plt.subplots(1, 3)
    # plots sawtooth impulse, use this format for things called on individal time moments.
    #pulse = np.zeros(len(sol.t))
    #for i in range(len(sol.t)):
    #    pulse[i] = impulseSaw(sol.t[i])
    #axs[0].plot(sol.t, pulse)
    axs[0].plot(sol.t, impulseSine(sol.t)) # plots sine impulse
    axs[0].set_xlabel(r'$t$', fontsize = 16)
    axs[0].set_ylabel(r'Impulse', fontsize=16)
    
    axs[1].plot(sol.t, sol.y[0, :])
    axs[1].set_xlabel(r'$t$', fontsize = 16)
    axs[1].set_ylabel(r'$Ca^{2+}$', fontsize=16)
    
    axs[2].plot(sol.t, sol.y[1, :])
    axs[2].set_xlabel(r'$t$', fontsize = 16)
    axs[2].set_ylabel(r'$CI^*$', fontsize=16)
    



    plt.show()


