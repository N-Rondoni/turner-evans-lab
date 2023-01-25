# Single ring solution using Zhang's equations,
# Now solved with Fourier Spectral method. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os

def tuningCurve(x):
    A = 1.72
    k = 5.29 #same k as in weight function or different? 
    B = 94.8/np.exp(k)
    x0 = 0
    out = A + B*np.exp(k*np.cos(x - x0))
    return out  


def weightFunc(x):
    # params
    b = .35    # amplitude
    k = 3      # period
    vs = .1    # vertical shift
    gamma = -0.063 # used as odd coeff

    # define even portion, close to sinc function
    evenOut = np.zeros(len(x))
    for i in range(len(x)):
        if np.isclose(x[i], 0): #if x[i] == 0:
            evenOut[i] = b - vs    
        else: 
            evenOut[i] = b*np.sin(b*k*np.pi*x[i])/(np.pi*x[i]) - vs
    # create odd portion, is derivative of even wrt x
    oddOut = np.zeros(len(x))
    for i in range(len(x)):
       oddOut[i] = gamma*((1/x[i])*(b**2)*k*np.cos(b*k*np.pi*x[i]) - (1/(np.pi * x[i]**2))*b*np.sin(b*k*np.pi*x[i]))
    totalOut = evenOut + oddOut
    return evenOut, oddOut, totalOut


def plotWeights(x, y1, y2, y3):
    plt.plot(x, y1, '--', linewidth = 1, label = 'Even Component')
    plt.plot(x, y2, '--', linewidth = 1, label = 'Odd Component')
    plt.plot(x, y3, linewidth = 2, label = 'Total (Odd + Even)')
    plt.title('Weight Distributions', fontsize = 18)
    plt.xlabel(r'$\theta$', fontsize = 14)
    plt.ylabel(r'$w(\theta, t)$', fontsize = 14)
    plt.legend()
    title = 'weight_functions.png'
    plt.savefig(title)
    os.system('cp '+ title +  ' /mnt/c/Users/nicho/Pictures/singleRing')
    plt.show() 


if __name__=="__main__":
    x = np.linspace(-np.pi, np.pi, 100)
    
    y1, y2, y3 = weightFunc(x)

    plotWeights(x, y1, y2, y3)

    print(tuningCurve(1))
    print("soup")


