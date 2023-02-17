# Single ring solution using Zhang's equations,
# Now solved with Fourier Spectral method. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os

def tuningCurve(x):
    A = 2.53
    k = 8.08 
    B = 34.8/np.exp(k)
    x0 = 0
    out = A + B*np.exp(k*np.cos(x - x0))
    return out  


def weightFunc(x):
    '''
        creates outputs for plotting identical to weightFunc2, works on arrays. 
    '''
    # params
    b = .35    # amplitude
    k = 3      # period
    vs = .1    # vertical shift
    gamma = -0.063 # used as odd coeff
    alpha = 0.00201

    # define even portion, close to sinc function
    evenOut = np.zeros(len(x))
    oddOut = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == 0:
            evenOut[i] = b - vs
            oddOut = 0
        else: 
            evenOut[i] = b*np.sin(b*k*np.pi*x[i])/(np.pi*x[i]) - vs
            oddOut[i] = gamma*((1/x[i])*(b**2)*k*np.cos(b*k*np.pi*x[i]) - (1/(np.pi * x[i]**2))*b*np.sin(b*k*np.pi*x[i]))
    # create odd portion, is derivative of even wrt x  
    #oddOut = gamma*np.sin(x)

    totalOut = evenOut + oddOut
    return evenOut, oddOut, totalOut


def weightFunc2(x):
    '''
        creates weight functions to be used in filling of weight matrix. Identical values as weightFunc, 
        but on individual element not list.
    '''
    # params
    b = .35    # amplitude
    k = 3      # period
    vs = .1    # vertical shift
    gamma = -0.063 # used as odd coeff
    alpha = 0.00201

    # define even/odd portions, close to sinc function and its derivative respectively
    if x == 0:
        evenOut = b - vs
        oddOut =  0
    else: 
        evenOut = b*np.sin(b*k*np.pi*x)/(np.pi*x) - vs
        oddOut = gamma*((1/x)*(b**2)*k*np.cos(b*k*np.pi*x) - (1/(np.pi * x**2))*b*np.sin(b*k*np.pi*x))
    # Another odd option
    #oddOut = gamma*np.sin(x)
    totalOut = evenOut + oddOut
    return totalOut

def weightMat(x):
    W = np.zeros((len(x), len(x)))    
    for i in range(len(x)):
        for j in range(len(x)):
            #if ((x[i] - x[j]) >= np.pi): #doesn't work for negatives
            #    temp = (x[i] - x[j]) % (2*np.pi)
                #print(temp)
            #else: 
            #    temp = x[i] - x[j]
            #W[i,j] = temp
            #######################################################
            if ((x[i] - x[j]) >= np.pi):
                temp = x[i] - x[j] - 2*np.pi
            elif (x[i] - x[j] <= -np.pi):
                temp = x[i] - x[j] + 2*np.pi
            else:
                temp = x[i] - x[j] 
            W[i, j] = weightFunc2(temp)
            #W[i, j] = temp
    return W
    

def sigma(s):
    a = 6.34
    beta = 0.8
    b = 10
    c = 0.5
   
    #a = 1
    #beta = 1
    #b = 1.1
    #c = 1
    
    out = np.zeros(len(s))
    for i in range(len(s)):
        out[i] = a*np.log(1 + np.exp(b*(s[i] + c)))**beta
    return out

def sys(t, u):
    '''
    RHS of system to be used in solver. Parameter tau kept here.
    '''
    Tau = 10

    thetaSpace = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
    W = weightMat(thetaSpace)
    #f = forcingVec(thetaSpace) creates constant forcing
    f = sigma(u)

    du = (1/Tau)*(-u + (1/len(u)) * np.matmul(W, f))
   
    #print(du[-1], du[0]) #This does not make the solution periodic. i

    #du[0] = du[-1]
    #du[1] = du[-2]
    #du[-1] = du[0]
    return du


def initialCondRand(x):
    IC = np.zeros(len(x))
    IC[0] = 1
    for i in range(1, len(x)):
        IC[i] = IC[i-1]#/i + np.random.rand()
    return IC


def initialCondBump(x):
    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = tuningCurve(x[i])
    return IC



## Plotting functions ##  -------------------------------------
def plot3d(x, y, z):
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Single Ring")
    ax.set_ylabel('Time')
    ax.set_xlabel(r'$\theta$')
    ax.set_zlabel(r'$u$')
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    filename = 'SR.png'
    fig1.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/singleRing') # only run with this line uncommented if you are Nick
    plt.show()

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
    #plt.show() 

## End Plotting Functions ## -------------------------------------


if __name__=="__main__":
    # set number of spatial discretizations
    N = 50

    # set up theta space
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    #y1, y2, y3 = weightFunc(x)
    #print(x)

    # plot weight function
    #plotWeights(x, y1, y2, y3)
   
    # create initial condiitions
    u0 = initialCondRand(x)

    sol = solve_ivp(sys, [0, 1200], u0)
  

    timeGrid, thetaGrid = np.meshgrid(sol.t, x)
    
    # Recall sol is energy levels, not actual firing rates. 
    # step through each time slice, rectify accordingly (pass thru sig) 
  
    (m, n) = sol.y.shape
    firingRate = np.zeros((m, n))
    for i in range(n):
        firingRate[:, i] = sigma(sol.y[:, i])

     
    plot3d(thetaGrid, timeGrid, firingRate)
    #plot3d(thetaGrid, timeGrid, sol.y)

    w = weightMat(x)
    print(w)
    plt.matshow(w)
    plt.show()
