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
            evenOut[i] = (b**2)*k - vs
            oddOut[i] = 0
        else: 
            evenOut[i] = b*np.sin(b*k*np.pi*x[i])/(np.pi*x[i]) - vs
            oddOut[i] = gamma*((1/x[i])*(b**2)*k*np.cos(b*k*np.pi*x[i]) - (1/(np.pi * x[i]**2))*b*np.sin(b*k*np.pi*x[i]))
    # create odd portion, is derivative of even wrt x  
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
        evenOut = (b**2)*k - vs
        oddOut =  0
    else: 
        evenOut = b*np.sin(b*k*np.pi*x)/(np.pi*x) - vs
        oddOut = gamma*((1/x)*(b**2)*k*np.cos(b*k*np.pi*x) - (1/(np.pi * x**2))*b*np.sin(b*k*np.pi*x))

    totalOut = evenOut + oddOut
    return totalOut

def weightMat(x):
    W = np.zeros((len(x), len(x)))    
    for i in range(len(x)):
        for j in range(len(x)):
            if ((x[i] - x[j]) >= np.pi):
                temp = x[i] - x[j] - 2*np.pi
            elif (x[i] - x[j] <= -np.pi):
                temp = x[i] - x[j] + 2*np.pi
            else:
                temp = x[i] - x[j] 
            W[i, j] = weightFunc2(temp)
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
    Tau = 10

    thetaSpace = np.linspace(-np.pi, np.pi, N, endpoint=False)
    
    W = weightMat(thetaSpace)
    #f = forcingVec(thetaSpace) creates constant forcing
    f = sigma(u) #creates the travelling bump
    #f = linRect(u) #is not good
    
    
    du = (1/Tau)*(-u + (1/len(u)) * np.matmul(W, f))
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


## Plotting functions ##  -------------------------------------
def plot3d(x, y, z):
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Single Ring", fontsize = 18)
    ax.set_ylabel('Time (ms)')
    ax.set_xlabel(r'$\theta$')
    ax.set_zlabel(r'$\sigma(u)$ (Hz)')
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
    plt.matshow(A)
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
    u0 = initialCondRand(x)
    #u0 = initialCondFlat(x)
   
    ''' -------------------------------
    End changable parameters. 
    '''

    sol = solve_ivp(sys, [0, 1200], u0)
  

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
    
        
    # velocity is constant so it may be computed using any time instants (after bump is formed)
    # in light of this, compute vel between time step 20 and 21. 
    bumpVal1 = np.max(firingRate[:, 40]) # returns max value at any node, 20th time step
    id1 = np.where(firingRate[:, 40] == bumpVal1)[0] # returns index of max theta value.
    #print(bumpVal1, firingRate[id1, 40])
    
    bumpVal2 = np.max(firingRate[:, 41]) # returns max value at any node, 21st time step
    id2 = np.where(firingRate[:, 41] == bumpVal2)[0] # index of max theta val. 
    #print(bumpVal2, firingRate[id2, 41])

    bumpVal3 = np.max(firingRate[:, 30]) # returns max value at any node, 20th time step
    id3 = np.where(firingRate[:, 30] == bumpVal3)[0] # returns index of max theta value.
    #print(bumpVal3, firingRate[id3, 30])

    bumpVal4 = np.max(firingRate[:, 31]) # returns max value at any node, 21st time step
    id4 = np.where(firingRate[:, 31] == bumpVal4)[0] # index of max theta val. 
    #print(bumpVal4, firingRate[id4, 31])


    # finally compute velocity
    #dTheta = x[id2] - x[id1]
    #print(x[id2], x[id1])
    #dt = sol.t[41] - sol.t[40]
    #print(sol.t[41], sol.t[40])
    #print(dTheta, dt)
    #vel = dTheta/dt
    #print("velocity:", vel, "at times:", sol.t[41])


    #dTheta = x[id4] - x[id3]
    #print(x[id4], x[id3])
    #dt = sol.t[31] - sol.t[30]
    #print(sol.t[31], sol.t[30])
    #print(dTheta, dt)
    #vel = dTheta/dt
    #print("velocity:", vel, "at times:", sol.t[31])

    velocities = np.zeros(len(sol.t) -1)
    for i in range(1, len(sol.t)-1):
        bumpVal1 = np.max(firingRate[:, i]) # returns max value at any node, 20th time step
        id1 = np.where(firingRate[:, i] == bumpVal1)[0] # returns index of max theta value.
        bumpVal2 = np.max(firingRate[:, i+1]) # returns max value at any node, 21st time step
        id2 = np.where(firingRate[:, i+1] == bumpVal2)[0] # index of max theta val. 
        dTheta = x[id2] - x[id1]
        #if dTheta < 0:
        #    dTheta = dTheta + 2*np.pi
        dt = sol.t[i+1] - sol.t[i]
        #print(dTheta, dt)
        angSpeed = dTheta/dt
        #print(angSpeed) 
        velocities[i] = angSpeed[0] # can make an array with larger dim.
        #print(dTheta)
        #print(dt)
        #print(angSpeed)
        #print(type(angSpeed))
        #print("timestep: ", i, "time: ", sol.t[i], "velocity:", angSpeed)
    
    velocities = velocities*1000 #move from rad/ms to rad/s

    print(velocities)

    # final velocity taken to be median. On sample runs,
    # this seems to be a pretty good way to pick true value.
    
    #print(np.mean(velocities))
    print(np.median(velocities))
    


    #plot3d(thetaGrid, timeGrid, sol.y)
    
    #w = weightMat(x)
    #matVis(w)
    #print(w)
    #eVals, eVecs = np.linalg.eig(w)
    #np.save("eigVals/eVals.npy", eVals)
