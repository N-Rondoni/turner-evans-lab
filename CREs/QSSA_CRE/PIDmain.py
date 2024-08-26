# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time

tPrev = 0
ePrev = 0
eSum = 0

def CRN(t, A):
    global tPrev
    global ePrev
    global eSum
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI, CI^* respectively. 
                A = [x]
        t : time
        p : vector of the parameters:
                p = [kf, kr, alpha, gamma]
    """
    x = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, kProp, kDer, kInt, CI_tot = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    # interpolate because data doesn't have values for all times used by solver.
    CI_MeasTemp = np.interp(t, timeVec, CI_Meas[:len(timeVec)])
    
    # Keep current error to compute eProp, and der for next pass. 
    # proportional portion
    #print(x)

    z = kf * CI_tot * x/(kr + kf) #this is CI^*
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
    du = [alpha*s - gamma*x + (kr * kf * CI_tot * x /(kr + kf)) - (kr * kf * CI_tot * x /(kr + kf * x)) ] 

    tPrev = t
    ePrev = eCurrent

    return du



if __name__=="__main__":
    # Load in Dan's fly data from janelia. Using cond{1}.allFlyData{1}.Strip{2}.RROIavMax
    #file_path = '~/turner-evans-lab/CRE/flymat.mat'
    file_path = 'data/flymat.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
    m,n = data.shape

    # step through rows: hope to see bump travel across all. rows are stepped through with driver. 
    row = int(sys.argv[1])
    print("node", row, "solution is being generated.")

    CI_Meas = data[row, :] #pull a particular row so we're looking at a single neuron. Testing figures created with 1.  
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
    X = 50 #Ca^{2+}

    # define constants/reaction parameters, kr << kf
    kf = 0.0513514
    kr = 7.6 
    alpha = 1
    gamma = 1 
    CI_tot = 20 # total number of available CI molecules? A count or given in moles? Using 100 puts into divergent regieme. 
    # gain coeff
    kProp = 1000
    kDer = 1
    kInt = 1
    
    # pack up parameters and ICs
    p = [kf, kr, alpha, gamma, kProp, kDer, kInt, CI_tot]
    u0 = [X]

    # Actually solve
    start = time.time()
    sol = solve_ivp(CRN, tsolve, u0, t_eval=timeVec)
    end = time.time()
    print('Total runtime for row ', str(row), end - start)
  
 
    # Post Processing for Plotting

    z = np.array(kf * CI_tot * sol.y/(kr + kf)) #this is CI^*    
    z = np.reshape(z, CI_Meas.shape) # reshape to be (743, ) 

    CI_MeasTemp = np.interp(sol.t, timeVec, CI_Meas)
    eCurrent = (z - CI_MeasTemp) # hopefuly same interp'd values as in solver. 


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
    print(eCurrentSub[0:3])


    # compute approximate integral of error 
    eSumTemp = tVec*eCurrentSub # this is not a sum of error, done in eSum.
    
    eSum = np.zeros(len(eSumTemp))
    for i in range(len(eSumTemp)):
        eSum[i] = np.sum(eSumTemp[0:i])


    sVec = -kProp*eCurrentSub + -kDer*derVec + -kInt*eSum


    # first term of sVec is cursed, massive due to large error. Difficult to initalize CI^* larger.
    sVec = sVec[1:]


    # have one additional time point than sVec values, due to derivative.
    subt = np.zeros(len(sol.t) -1)
    subt = sol.t[1:-1] # cut off first and last terms due to above. First term cursed, last term has no S val. 
   
    
    fpath = ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/QSSA' 
    # plot 3 values solved for in CRE
    plt.figure(1)
    plt.plot(sol.t, sol.y[0, :], linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(sol.t, z, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig1_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) # only run with this line uncommented if you are Nick

    # plot error across time
    plt.figure(2)
    plt.plot(sol.t, -1*eCurrent, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig2_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 
    
    # Additional Plotting, this is the guts of plotS
    #plot firing rate
    plt.figure(3)
    plt.plot(subt, sVec, label = r'$S$')
    plt.title('Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig3_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 
       
    #plot subset of firing rate
    plt.figure(4)
    plt.plot(subt[:200], sVec[:200], label = r'$S$')
    plt.title('Subset of time, Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig4_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 
       
    #plotS(subt, derVec)
    plt.figure(5)
    plt.plot(sol.t[:-1], derVec, label = r'$S$')
    plt.title('Dynamics of derivative of Error', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$\dot{e}$', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig5_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 
   
   #plt.show() 

    # plot measured vs simulated CI^*
    plt.figure(6)
    plt.plot(sol.t, CI_Meas, label=r'$CI^{*}_{Meas}$')
    plt.plot(sol.t, z, label=r'$CI^{*}_{Sim}$')
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'QSSA_CRE_fig6_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 
  
    #plot ordered pairs 
    plt.figure(7)
    # first create them
    
    #print(sVec[0:10])
    #plt.scatter(z[1:-1], sVec) #, label = "()" 
    plt.scatter(sol.y, z) #pointless one is computed with the other.
    plt.title(r"Ordered Pairs $(s, CI^*_{Sim})$")
    plt.xlabel(r"$S$")
    plt.ylabel(r"$CI^*_{Sim}$")
    filename = 'QSSA_CRE_fig7_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + fpath) 

    sVec = -kProp*eCurrentSub + -kDer*derVec + -kInt*eSum
 
    
    # analyze calcium buildup to back out firing rate?
    sec = 1
    Accum = np.zeros(int(np.ceil(sol.t[-1])))
    runDif = 0
    tempDif = 0
    for i in range(len(sol.y[0,:]) - 1):
        tempDif = sol.y[0, i+1] - sol.y[0, i] 
        if sol.t[i] < sec:
            if tempDif > 0:
                runDif = tempDif + runDif
        if sol.t[i] > sec:
            Accum[sec - 1] = runDif
            runDif = 0
            sec = sec + 1

    tspanFire = np.linspace(0, np.ceil(sol.t[-1]), len(Accum))
    
    plt.figure(8)
    plt.plot(tspanFire[1:], Accum[1:])
    

    print(len(Accum))
    print(len(tspanFire))

    #os.system('rm *.png')
    #plt.show()
    

    np.save('data/QSSA_s_node_' + str(row), sVec)
    np.save('data/QSSA_t_node_' + str(row), sol.t)
    np.save('data/QSSA_sol_node_' + str(row), sol.y)
    np.save('data/QSSA_PosIncreases_node_' + str(row), Accum)
    

    #np.save("temp.dat", sol.y[0,:])
    
    #plt.figure(6)
    #plt.figure(6)
    #print(sol.y)
    #print(tEvals.shape, S.shape)
    #print(tEvals)
    #print((sol.y).shape)
