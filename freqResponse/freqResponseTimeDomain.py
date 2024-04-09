# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time

def impulse(t):
    T = 1
    A = 1
    signal = np.sin(t)
    return signal




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
    
    s = impulse(t)

    du = [alpha*s - gamma*x + kr*z - kf*x*(L-z), # + beta 
        kf*x*(L-z) - kr*z] 
    
    return du


if __name__=="__main__":
    n = 1000
    imRate = 11.4
    tEnd = n*(1/imRate)
    tEnd = 100
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
  

    plt.figure()
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(sol.t, impulse(sol.t))
    axs[0].set_xlabel(r'$t$', fontsize = 16)
    axs[0].set_ylabel(r'Impulse', fontsize=16)
    
    axs[1].plot(sol.t, sol.y[0, :])
    axs[1].set_xlabel(r'$t$', fontsize = 16)
    axs[1].set_ylabel(r'$Ca^{2+}$', fontsize=16)
    
    axs[2].plot(sol.t, sol.y[1, :])
    axs[2].set_xlabel(r'$t$', fontsize = 16)
    axs[2].set_ylabel(r'$CI^*$', fontsize=16)
    


    plt.show()
    sys.exit()


    # plot 3 values solved for in CRE
    plt.figure(1)
    y1 = sol.y[0,:]
    print(y1[0:10])
    y2 = sol.y[1,:] 
    y3 = sol.y[2,:]
    plt.plot(sol.t, y1, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(sol.t, y2, '--', linewidth = 1, label = r'$CI$')
    plt.plot(sol.t, y3, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig1_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

    # plot error across time
    plt.figure(2)
    plt.plot(sol.t, -1*eCurrent, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(row) + '.png'
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
    filename = 'CRE_fig3_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    
    #plot subset of firing rate
    plt.figure(4)
    plt.plot(subt[:200], sVec[:200], label = r'$S$')
    plt.title('Subset of time, Dynamics of S backsolved from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'S (hz)', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig4_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    
    #plotS(subt, derVec)
    plt.figure(5)
    plt.plot(subt, derVec, label = r'$S$')
    plt.title('Dynamics of derivative of Error', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$\dot{e}$', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig5_' + str(row) + '.png'
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
    filename = 'CRE_fig6_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

    #plot Ca^{2+} seperately as concentrations are too low to see when plotted together
    plt.figure(7)
    plt.plot(sol.t, sol.y[0,:])#, label = r'$Ca^{2+}$')
    plt.title(r'Dynamics of $Ca^{2+}$')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$Ca^{2+}$', fontsize = 14)
    filename = 'CRE_fig7_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick

    #np.save('data/s_node_' + str(row), sVec)
    #np.save('data/t_node_' + str(row), sol.t)
    #np.save('data/sol_node_' + str(row), sol.y)


    plt.show()


    #np.save("temp.dat", sol.y[0,:])
    
    #plt.figure(6)
    #plt.figure(6)
    #print(sol.y)
    #print(tEvals.shape, S.shape)
    #print(tEvals)
    #print((sol.y).shape)
