# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os

def firing(t):
    ##
    # Idealized firing rate of the neuron in question. Returns hz at given time.
    ##
    peakHz = 150
    rate = peakHz*np.exp(-5*(t-.1)**2) 
    return rate

sVec = np.array([0])

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
    
    CI_MeasTemp = np.interp(t, timeVec, CI_Meas)
    k = -1
    s = k*(z - CI_MeasTemp)
    sVec = np.append(sVec, s)
    print(sVec)
    #print(s)
    
    # I really wanted to avoid doing this but I have scope issues. WARNING: JANK TO FOLLOW. 
    #filename = "sVec"
    #f = open(filename, 'a')
    #f.write(str(s))
    #f.close
    #np.save(filename, s)
    # end jank


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

def plotS(x):
    sTemp = np.load("sVec.npy")
    plt.plot(x, sTemp, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.title('Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    plt.show() 



if __name__=="__main__":
    # Load in Dan's fly data from janelia. Using cond{1}.allFlyData{1}.Strip{2}.RROIavMax
    #file_path = '~/turner-evans-lab/CRE/flymat.mat'
    file_path = 'flymat.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
    m,n = data.shape

    CI_Meas = data[1, :] #pull a particular row so we're looking at a single neuron. 
    CI_Meas = 100*CI_Meas

    # compute total time, (imaging was done at 11.4hz), create vec of time samples timeVec
    # timeVec used within solver to compute interpolated value of S at a particular time. np.interp(t, timeVec, CI_Meas)
    imRate = 11.4
    tEnd = n*(1/imRate)
    timeVec = np.linspace(0, tEnd, n)


    # define initial conditions
    X = 0 #Ca^{2+}
    Y = 100 #CI
    Z = 0   #CI^*

    # define constants/reaction parameters, kr << kf
    kf = 100
    kr = 50
    alpha = 100  #was 10 with Marcella. Uped alpha to increase production rate of Ca^{2+}. Scaled gamma accordingly. 
    gamma = 1000

    # pack up parameters and ICs
    p = [kf, kr, alpha, gamma]
    u0 = [X, Y, Z]

    # set up time for solver
    tsolve = [0, tEnd]

    # Delete sVec file to accumlate (new) s values within solver. Otherwise appends to old values. If no sVec present, comment out this line. 
    #os.system("rm sVec.npy")
    
    sol = solve_ivp(CRN, tsolve, u0, t_eval=timeVec)

    plotThreeLines(sol.t, sol.y[0,:], sol.y[1,:], sol.y[2,:])

    print(sVec)

    #sTemp = np.load("sVec.npy")
    #print(sTemp)

    #plotS(timeVec)

    #### Testing print statements below. 
    # import s and time, created by singleRing.py
    #tEvals = np.load("timesEvaled.npy")
    #S = np.load("FiringRates.npy")
    ###### Using idealized s instead ######

    #print(sol.y)
    #print(tEvals.shape, S.shape)
    #print(tEvals)
    #print((sol.y).shape)
