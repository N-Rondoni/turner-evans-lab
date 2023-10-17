# Single ring solution using Zhang's equations,
# Now solved with Fourier Spectral method. 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
plt.rcParams['svg.fonttype'] = 'none'
from scipy.integrate import solve_ivp
import scipy.io
import pingouin as pg
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
            evenOut[i] = (b**2)*k - vs
            oddOut[i] = 0
        else: 
            evenOut[i] = b*np.sin(b*k*np.pi*x[i])/(np.pi*x[i]) - vs
            oddOut[i] = gamma*((1/x[i])*(b**2)*k*np.cos(b*k*np.pi*x[i]) - (1/(np.pi * x[i]**2))*b*np.sin(b*k*np.pi*x[i]))
    # create odd portion, is derivative of even wrt x  
   

    totalOut = evenOut + oddOut
    return evenOut, oddOut, totalOut


def weightFunc2(x, t):
    '''
        creates weight functions to be used in filling of weight matrix. Identical values as weightFunc, 
        but on individual element not list. Requires realVel, timeVec to be defined before use.
    '''
    # params
    b = .35    # amplitude
    k = 3      # period
    vs = .1    # vertical shift
    gamma = -0.063 # used as coeff in odd part


    # define even/odd portions, close to sinc function and its derivative respectively
    if x == 0:
        evenOut = (b**2)*k - vs
        oddOut =  0
    else: 
        evenOut = b*np.sin(b*k*np.pi*x)/(np.pi*x) - vs
        oddOut = gamma*((1/x)*(b**2)*k*np.cos(b*k*np.pi*x) - (1/(np.pi * x**2))*b*np.sin(b*k*np.pi*x))

    # interpolate velocities to match times
    velInterp = np.interp(t, timeVec, realVel)
    
    # update odd portion with velocity. Found by noting each scalar multiple of odd out scales vel accordingly.
    # Eg., want 1 odd out to yield vel of 6.2 
    velAn = -gamma / Tau # from paper, analytic expression for angular vel. In rad/ms currently. 
    velAn = velAn * 1000 # contert to seconds
    scalarMult = (1/velAn)*velInterp 
    #(1/6.294898346736869)*velInterp

#    print(velInterp)
#    print(velAn)

    totalOut = evenOut + scalarMult*oddOut
    return totalOut


def weightMat(x, t):
    W = np.zeros((len(x), len(x)))    
    for i in range(len(x)):
        for j in range(len(x)):
            if ((x[i] - x[j]) >= np.pi):
                temp = x[i] - x[j] - 2*np.pi
            elif (x[i] - x[j] <= -np.pi):
                temp = x[i] - x[j] + 2*np.pi
            else:
                temp = x[i] - x[j] 
            W[i, j] = weightFunc2(temp, t)
    return W
    

def sigma(s):
    # this is what the authors use to go from energy -> hz
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


def linRect(s):
    # here is a more modern correction function. Graphs are the same essentially
    #     if soln is passed in at end. Cannot be used in sys tho. 
    out = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] > 0:
            out[i] = s[i]
        else:
            out[i] = 0 
    return out


def sys(t, u):
    '''
    RHS of system to be used in solver. Parameter tau kept here.
    '''

    thetaSpace = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
    W = weightMat(thetaSpace, t)
    #f = forcingVec(thetaSpace) creates constant forcing
    f = sigma(u) #creates the travelling bump
    #f = linRect(u) #is not good
    
    
    du = (1/Tau)*(-u + (1/len(u)) * np.matmul(W, f))
    du = 1000*du # convert from ms to s. 
    return du


def initialCondRand(x):
    IC = np.zeros(len(x))
    IC[0] = 1
    for i in range(1, len(x)):
        IC[i] = IC[i-1]/i + np.random.rand()
    return IC


def initialCondBump(x):
    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = tuningCurve(x[i])
    return IC


def initialCondFlat(x):
    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = 1
    return IC


def initialCondBias(x):
    # could do better here
    # instead of if, find closes and set to 1
    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = 1
        # constant comes from matlab exploration
        # examaine real world data for bump loc
        if np.isclose(x[i], -2.0328, atol = 1E-1):
            IC[i] = 2
    return IC

## Plotting functions ##  -------------------------------------
def plot3d(x, y, z):
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Single Ring", fontsize = 22)
    ax.set_ylabel('Time (s)', fontsize = 16)
    ax.set_xlabel(r'HD Cell $\theta$', fontsize = 16)
    ax.set_zlabel(r'Firing Rate $f$ (Hz)', fontsize = 16)
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    plt.show()

def plotWeights(x, y1, y2, y3):
    plt.plot(x, y1, '--', linewidth = 1, label = 'Even Component')
    plt.plot(x, y2, '--', linewidth = 1, label = 'Odd Component')
    plt.plot(x, y3, linewidth = 2, label = 'Total (Odd + Even)')
    plt.title('Weight Function', fontsize = 18)
    plt.xlabel(r'$\theta$', fontsize = 14)
    plt.ylabel(r'$w(\theta)$', fontsize = 14)
    plt.legend()
    plt.show() 


def matVis(A):
    vmax = np.max(A)
    vmin = np.min(A)  
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter = 0, vmax = vmax) # centers colorbar around 0
    plt.matshow(A, extent = [-np.pi, np.pi, np.pi, -np.pi], cmap='bwr', norm=norm)
    plt.xticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.xlabel(r'$\theta$', fontsize = 14)
    plt.ylabel(r'$\theta$', fontsize = 14)
    plt.colorbar()
    plt.title("Visualization of Weight Matrix", fontsize = 20)
    plt.show()
    

## End Plotting Functions ## -------------------------------------


if __name__=="__main__":
    # set number of spatial discretizations
    N = 48
    # set up theta space
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    y1, y2, y3 = weightFunc(x)
    #print(x)

    # plot weight function
    plotWeights(x, y1, y2, y3)
   
    # create initial condiitions
    #u0 = initialCondRand(x)
    #u0 = initialCondFlat(x)
    u0 = initialCondBias(x)

    # define time constant
    Tau = 10
    
    ''' -------------------------------
    End changable parameters. 
    '''
        # load in velocities from the data,
    # reshape it so it plays well with others.
    file_path = "data/vRot_cond1_allFly1_stripe2.mat"
    mat = scipy.io.loadmat(file_path)
    temp = mat['vel'] # print mat to see what options
    (m, n) = temp.shape #flipped from other data
    realVel = np.zeros(m)
    realVel = temp[:, 0]
    print(m,n)

    # instead of using imaging rate, load in time data.
    file_path2 = "data/time_cond1_allFly1_stripe2.mat"
    mat = scipy.io.loadmat(file_path2)
    temp = mat['time']
    (m1, n1) = temp.shape
    timeVec = np.zeros(m1)
    timeVec = temp[:-1, 0] # chop off last time for same number of vel
    tEnd = timeVec[-1]     # this also reshapes
    #tEnd = 3.5             # uncomment for faster testing

    # temp plotting
    w = weightMat(x, 0)
    matVis(w)
    
    w = weightMat(x, 100)
    matVis(w)
    ##

    sol = solve_ivp(sys, [0, tEnd], u0)


    timeGrid, thetaGrid = np.meshgrid(sol.t, x)
    
    # Recall sol is energy, not actual firing rates. 
    # step through each time slice, rectify accordingly (pass thru sig) 
  
    (m, n) = sol.y.shape
    firingRate = np.zeros((m, n))
    for i in range(n):
        #firingRate[:, i] = 15*linRect(sol.y[:, i])  # more modern correction function.
        firingRate[:, i] = sigma(sol.y[:, i])      # what the author's use.

    # finally plot solution.
    plot3d(thetaGrid, timeGrid, firingRate)


    # compute velocities to back out how odd component impacts vel.
    vel = np.zeros(len(sol.t) - 1)
    for i in range(1, len(sol.t)-1):
        m1 = pg.circ_mean(x, firingRate[:,i])
        m2 = pg.circ_mean(x, firingRate[:,i+1])
        tempVel = (m1 - m2) / (sol.t[i] - sol.t[i+1])
        vel[i] = tempVel


    np.save("data/firingRates.npy", firingRate)
    np.save("data/firingTimes.npy", sol.t)
    np.save("data/velSim.npy", vel)
    

    matVis(firingRate) 
    

    
