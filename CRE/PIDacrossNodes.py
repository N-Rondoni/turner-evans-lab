# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import time

def firing(t):
    ##
    # Idealized firing rate of the neuron in question. Returns hz at given time.
    ##
    peakHz = 150
    rate = peakHz*np.exp(-5*(t-.1)**2) 
    return rate

tPrev = 0
ePrev = 0
eSum = 0
#sVec = np.array([0])

def CRN(t, A):
    global tPrev
    global ePrev
    global eSum
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
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, kProp, kDer, kInt, beta = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    # interpolate because data doesn't have values for all times used by solver.
    CI_MeasTemp = np.interp(t, timeVec, CI_Meas[:len(timeVec)])
    
    # Keep current error to compute eProp, and der for next pass. 
    # proportional portion
    eCurrent = (z - CI_MeasTemp)
    
    # derivative portion
    if t == tPrev:
        eDer = 0
    else:
        eDer = (eCurrent - ePrev)/(t - tPrev) 

    # integral portion (approximatd with Reimann type sum)
    eSum = eSum + (t - tPrev)*eCurrent
    
    s = -kProp*eCurrent + -kDer*eDer + -kInt*eSum 

    #print(t, s)
    du = [alpha*s - gamma*x + kr*z - kf*y*x, # + beta
        kr*z - kf*y*x, 
        kf*y*x - kr*z] 

    tPrev = t
    ePrev = eCurrent

    return du

def plotThreeLines(x, y1, y2, y3):
    plt.figure(1)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$')
    plt.plot(x, y3, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig1_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    #plt.show() 

def plotErr(x, y):
    plt.figure(2)
    plt.plot(x, y, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick




if __name__=="__main__":
    # Load in Dan's fly data from janelia. Using cond{1}.allFlyData{1}.Strip{2}.RROIavMax
    #file_path = '~/turner-evans-lab/CRE/flymat.mat'
    file_path = 'flymat.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
    m,n = data.shape

    # step through rows: hope to see bump travel across all 
    for i in range(2):  #m):
        print(i)
    


    CI_Meas = data[5, :] #pull a particular row so we're looking at a single neuron. Testing figures created with 1.  
    CI_Meas = 50*CI_Meas


    # compute total time, (imaging was done at 11.4hz), create vec of time samples timeVec
    # timeVec used within solver to compute interpolated value of S at a particular time. np.interp(t, timeVec, CI_Meas)
    imRate = 11.4
    tEnd = n*(1/imRate)
    timeVec = np.linspace(0, tEnd, n)
    # short time testing uncomment below ###
    #tEnd = .1
    #nDatPoints = 5
    #timeVec = np.linspace(0, tEnd, nDatPoints)
    #########################################
    # set up time for solver
    tsolve = [0, tEnd]

    # define initial conditions
    X = 0 #Ca^{2+}
    Y = 100 #CI
    Z = 0   #CI^*

    # define constants/reaction parameters, kr << kf
    kf = 100 #x10ing this and below seems to reeally lengthen runtime. 
    kr = 5 #was 50 to create plots from meeting 5_25 /// 
    alpha = 10  #was 10 with Marcella. Uped alpha to increase production rate of Ca^{2+}. Scaled gamma accordingly. 
    gamma = 100
    beta = 4000
    # gain coeff
    kProp = 1
    kDer = 1
    kInt = 1
    
    # pack up parameters and ICs
    p = [kf, kr, alpha, gamma, kProp, kDer, kInt, beta]
    u0 = [X, Y, Z]

    # Actually solve
    start = time.time()
    #print("Solver has started at", start)
    sol = solve_ivp(CRN, tsolve, u0, t_eval=timeVec)
    end = time.time()
    #print("solution has been generated, finishing at", end)
    print('Total runtime for row ', str(i), end - start)
    
    #plotThreeLines(sol.t, sol.y[0,:], sol.y[1,:], sol.y[2,:])
   

    # Post Processing for Plotting
    # prop portion
    eCurrent = (sol.y[2,:] - CI_Meas[:len(timeVec)]) 
        
    # derivative portion
    # compute t vec (many delta t values)
    tVec = np.zeros(len(sol.t) - 1)
    for i in range(len(tVec)):
        tVec[i] = sol.t[i+1] - sol.t[i] 

    # compute der Vec
    derVec = np.zeros(len(eCurrent) - 1)

    for i in range(len(derVec)):
        if tVec[i] == 0:
            derVec = 0 
        else: 
            derVec[i] = (eCurrent[i+1] - eCurrent[i])/tVec[i]
   
    # these fuckers (sol.t and derVec) are differen length. (743) vs (742).
    #plotErr(sol.t, -1*eCurrent) 

    eCurrentSub = eCurrent[:-1]

    # compute approximate integral of error 
    eSumTemp = tVec*eCurrentSub # this is not a sum of error???????
    
    eSum = np.zeros(len(eSumTemp))
    for i in range(len(eSumTemp)):
        eSum[i] = np.sum(eSumTemp[0:i])


    sVec = -kProp*eCurrentSub + -kDer*derVec + -kInt*eSum

    # have one additional time point than sVec values, due to derivative.
    subt = np.zeros(len(sol.t) -1)
    subt = sol.t[:-1]
   

    # plot 3 values solved for in CRE
    plt.figure(1)
    y1, = sol.y[0,:]
    y2 = sol.y[1,:] 
    y3 = sol.y[2,:]
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$')
    plt.plot(x, y3, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig1_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

    # plot error across time
    plt.figure(2)
    plt.plot(sol.t, -1*eCuurrent, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick


    # Additional Plotting, this is the guts of plotS
    #plot firing rate
    plt.figure(3)
    plt.plot(subt, sVec, label = r'$S$')
    plt.title('Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig3_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    
    #plot subset of firing rate
    plt.figure(4)
    plt.plot(subt[:200], sVec[:200], label = r'$S$')
    plt.title('Subset of time, Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig4_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    
    #plotS(subt, derVec)
    plt.figure(5)
    plt.plot(subt, derVec, label = r'$S$')
    plt.title('Dynamics of derivative of Error', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$\dot{e}$', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig5_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

   #plt.show() 

    # plot measured vs simulated CI^*
    plt.figure(6)
    plt.plot(sol.t, CI_Meas, label=r'$CI^{*}_{Meas}$')
    plt.plot(sol.t, sol.y[2,:], label=r'$CI^{*}_{Sim}$')
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig6_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

    #plot Ca^{2+} seperately as concentrations are too low to see when plotted together
    plt.figure(7)
    plt.plot(sol.t, sol.y[0,:])#, label = r'$Ca^{2+}$')
    plt.title(r'Dynamics of $Ca^{2+}$')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$Ca^{2+}$', fontsize = 14)
    filename = 'CRE_fig7_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick



    #plt.show()


    #np.save("temp.dat", sol.y[0,:])
    
    #plt.figure(6)
    #plt.figure(6)
    #print(sol.y)
    #print(tEvals.shape, S.shape)
    #print(tEvals)
    #print((sol.y).shape)
