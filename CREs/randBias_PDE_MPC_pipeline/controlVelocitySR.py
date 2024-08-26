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
rng = np.random.default_rng(seed=22) # fix a random seed for replicability

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
    # uncomment below when you have a locDif computed. 
    locDifInterp = np.interp(t, np.linspace(0, tEnd, len(locDif)), locDif)
    locDifInterp = np.interp(t, t_Noised, locDif)

    # update odd portion with velocity. Found by noting each scalar multiple of odd out scales vel accordingly.
    # Eg., want 1 odd out to yield vel of 6.2 
    velAn = gamma / Tau # from paper, analytic expression for angular vel. In rad/ms currently. 
    velAn = velAn * 1000 # contert to seconds
  
    scalarMult = ((1.5+locDifInterp)/velAn)*velInterp  #1.5 due to 240 deg -> 360 deg mapping.
   
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
    #     if soln is passed in at end. Cannot be used in sys tho, 
    #     this is because ReLu is non differentiable at x = 0.
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
    # sets initial coniditon to match that of CI data
    file_path = 'data/ciDat' + state + '.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['ciDat' + state] 
    # pull out location maximal activity for ICs to match
    ciMaxLoc = np.argmax(data[:,0])

    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = 1
        # constant comes from matlab exploration
        # examaine real world data for bump loc
        if i == ciMaxLoc:
            IC[i] = 3 
    return IC

def initialCondBiasMean(x):
    # sets initial coniditon to match that of CI data
    file_path = 'data/ciDat' + state + '.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['ciDat' + state] 
    # pull out location maximal activity for ICs to match
    ciMaxLoc = pg.circ_mean(x, data[:, 0])

    IC = np.zeros(len(x))
    for i in range(len(x)):
        IC[i] = 1
        if np.isclose(x[i], ciMaxLoc):
            IC[i] = 3
    return IC


def hardCodeIC(x):
    # when you can't bias the IC based of CI data, can just hardcode it to agree more after seeing MPC result.
    IC = np.zeros(len(x))
    hardCode = -np.pi/2
    for i in range(len(x)):
        IC[i] = 1
        IC[14] = 3
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
    
def matVisPos(A):
    vmax = np.max(A)
    vmin = np.min(A)  
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter = (vmin + vmax)/2, vmax = vmax) # centers colorbar around 0
    plt.matshow(A, extent = [-np.pi, np.pi, np.pi, -np.pi], cmap='bwr', norm=norm)
    plt.xticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    plt.xlabel(r'$\theta$', fontsize = 14)
    plt.ylabel(r'$\theta$', fontsize = 14)
    plt.colorbar()
    plt.title("Visualization of S Matrix", fontsize = 20)
    plt.show()
 
## End Plotting Functions ## -------------------------------------


if __name__=="__main__":
    # set number of spatial discretizations
    N = 18

    # pull from data set with stripe or in dark
    state = 'Stripe'
    #state = 'Dark'

    # set up theta space
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)

    y1, y2, y3 = weightFunc(x)
    #print(x)

    # plot weight function
    #plotWeights(x, y1, y2, y3)
   
    # create initial condiitions
    #u0 = initialCondRand(x)
    #u0 = initialCondFlat(x)
    u0 = initialCondBias(x)
   # u0 = hardCodeIC(x)
    #u0 = initialCondBiasMean(x)
    
    print(u0)

    # define time constant
    Tau = 10
    
    
    ''' -------------------------------
    End changable parameters. 
    '''
    
    # load in velocities from the data,
    # reshape it so it plays well with others.
    
    #file_path = "data/velStripe" #velStripe: cond1_allfly2_stripe1_posDatMatch.vrot
    file_path = "data/vel" + state #velDark: cond1_allfly2_dark1_posDatMatch.vrot
    mat = scipy.io.loadmat(file_path)
    #temp = mat['velStripe']
    temp = mat['vel' + state] # print mat to see what options, should be var name in matlab
    (m, n) = temp.shape   # flipped from other data
    realVel = np.zeros(m)
    realVel = temp[:, 0]
    #print(m,n)
    #print(realVel)
   
    # instead of using imaging rate, load in time data.
    #file_path2 = "data/timeStripe"
    file_path2 = 'data/time' + state
    mat = scipy.io.loadmat(file_path2)
    #temp = mat['timeStripe']
    temp = mat['time' + state]
    (m1, n1) = temp.shape
    timeVec = np.zeros(m1)
    timeVec = temp[:-1, 0] # chop off last time for same number of vel
    tEnd = timeVec[-1]     # this also reshapes
    #tEnd = 3.5             # uncomment for faster testing

    # create LocDif, must load in noised PDE and Analytic soln (created by their respective python files)
    file_path3 = 'data/PDEfiringRates_Noised' + state + '.npy'    
    fr_Noised = np.load(file_path3)
    (m_Noised, n_Noised) = fr_Noised.shape

    file_path4 = 'data/PDEfiringRates_An' + state + '.npy'    
    fr_An = np.load(file_path4) 
    (m_An, n_An) = fr_An.shape

    thetaspace = x
    mlocAn = np.zeros(n_An)
    for i in range(n_An):
        m = pg.circ_mean(thetaspace, fr_An[:,i])
        mlocAn[i] = m

#    mlocAn = np.unwrap(mlocAn)


    mlocNoised = np.zeros(n_Noised)
    for i in range(n_Noised):
        m = pg.circ_mean(thetaspace, fr_Noised[:,i])
        mlocNoised[i] = m

#    mlocNoised = np.unwrap(mlocNoised)

    # create locDif, difference between maximal locations 
    t_Noised = np.load('data/PDEfiringTimes_Noised' + state + '.npy')
    t_An = np.load('data/PDEfiringTimes_An' + state + '.npy')
    
    # must interpoloate, they're different lengths. 
    locDif = np.interp(t_Noised, t_An, mlocAn) - mlocNoised # x coord to evaluate at, x coord of data, y coord of data.
    locDif = locDif/np.pi
    print((locDif))

    # temp plotting
    #w = weightMat(x, 0)
    #matVis(w)
    
    #w = weightMat(x, 100)
    #matVis(w)
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
 #   plot3d(thetaGrid, timeGrid, firingRate)


    np.save('data/PDEfiringRates_cont' + state + '.npy', firingRate)
    np.save('data/PDEfiringTimes_cont' + state + '.npy', sol.t)

    

#    matVisPos(firingRate) 
    

    
