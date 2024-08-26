# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out fluorescence given firing rate. (created by rateCreate.py, which is singleRing.py that writes out). 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def firing(t):
    ##
    # Idealized firing rate of the neuron in question. Returns hz at given time.
    ##
    peakHz = 150
    rate = peakHz*np.exp(-5*(t-.1)**2) 
    return rate

def CRN(t, A):
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI, CI^* respectively. 
                A = [x, y, z]
        t : time
        p : vector of the parameters:
                p = [kf, kr, alpha, gamma]
    """
    x, y, z = A
    kf, kr, alpha, gamma = p
    
    # define chemical master equation 
    # this is the RHS of the system
    
    s = firing(t)
    #print(s)
    #s = 1

    du = [alpha*s - gamma*x + kr*z - kf*y*x,
        kr*z - kf*y*x, 
        kf*y*x - kr*z] 
    return du

def plotThreeLines(x, y1, y2, y3):
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$')
    plt.plot(x, y3, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    plt.show() 


# define initial conditions
X = 0
Y = 100
Z = 0

# define constants/reaction parameters, kr << kf
kf = 100
kr = 50
alpha = 100  #was 10 with Marcella. Uped alpha to increase production rate of Ca^{2+}.
gamma = 1000



# pack up parameters and ICs
p = [kf, kr, alpha, gamma]
u0 = [X, Y, Z]


# set up time
tsolve = [0, 1]

sol = solve_ivp(CRN, tsolve, u0)#, t_eval=tEvals)

plotThreeLines(sol.t, sol.y[0,:], sol.y[1,:], sol.y[2,:])


#### Testing print statements below. 

# import s and time, created by singleRing.py
#tEvals = np.load("timesEvaled.npy")
#S = np.load("FiringRates.npy")
###### Using idealized s instead ######

#print(sol.y)
#print(tEvals.shape, S.shape)
#print(tEvals)
#print((sol.y).shape)
