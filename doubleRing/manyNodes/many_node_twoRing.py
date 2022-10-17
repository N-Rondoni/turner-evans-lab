# Purpose:
# Use:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os

def thetaDivider(thetaStart, thetaStop, n, thetaDensity):
    """
    Creates subintervals and needed midpoints for spacial discretization

    Arguments: 
        thetaStart   : starting spatial value (real number)
        thetaStop    : ending spatial value (real number)
        n            : number of subintervals
        thetaDensity : number of values in each subinterval   TODO: refactor, thetaDensity is irrelevant

    """
    endPoints = np.linspace(thetaStart, thetaStop, num = n+1)    
    
    # dense interval creations
    intervals = np.empty([n, thetaDensity])
    i = 0
    while i<n:
        intervals[i,:] = np.linspace(endPoints[i], endPoints[i+1], num = thetaDensity) 
        i = i+1

    # midpoint creation
    midpoints = np.empty(n)
    j = 0
    while j<n:
        midpoints[j] = (endPoints[j]+endPoints[j+1])/2 
        j = j+1

    return midpoints, intervals

# create two subintervals comprised of two theta values each. Store as intervals.
# create an array of the midpoints of the intervals.

def ws(theta):
    """
    weight function for the "same" ring. 
    
    Arguments: 
            Theta: the midpoint of our discretization intervals
    """
    [J0, J1] = [-60, 30]
    # can change the offset connection points with these angles below
    phiDeg = 89
    psiDeg = 45
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightSame = J0 + J1*np.cos(phi - theta - psi)
    return weightSame


def wd(theta):
    """
    weight function for the "different" ring. 
    
    Arguments: 
            Theta: the midpoint of our discretization intervals
    """
    [K0, K1] = [-5, 80]
    # can change the offset connection points with these angles below
    phiDeg = 89
    psiDeg = 45
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightDifferent = K0 + K1*np.cos(phi - theta - psi)
    return weightDifferent


def ic_maker_enum(spatial_discretizations):
    n = 2*spatial_discretizations
    i = 0
    s0 = np.zeros(n)
    while i < n:
        s0[i] = i
        i = i + 1
    #print(s0)
    return s0


def ic_maker_erf(spatial_discretizations):
    n = 2*spatial_discretizations
    midpoints, intervals = thetaDivider(-np.pi, np.pi, n, n)
    s0 = (2/np.pi)*np.exp(-(midpoints-(np.pi/2))**2)
    return s0


def ic_maker_periodic(spatial_discretizations):
    n = 2*spatial_discretizations
    midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_discretizations, spatial_discretizations)
    midpoints = np.concatenate((midpoints, midpoints))
    s0 = np.sin(midpoints)
    s0 = np.maximum(s0, 0)
    return s0


def weight_maker(spatial_discretizations):
    n = 2*spatial_discretizations
    weightVecL = np.zeros(n)
    weightVecR = np.zeros(n)
    midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_discretizations, spatial_discretizations)
    # need to loop over the same midpoints for both left and right rings
    midpoints = np.concatenate((midpoints, midpoints))
    i = 0
    while (i < n):
        if i < n/2:
            weightVecL[i] = ws(midpoints[i])
            weightVecR[i] = wd(midpoints[i])
        if i >= n/2: 
            weightVecL[i] = wd(midpoints[i])
            weightVecR[i] = ws(midpoints[i])
        i = i + 1
    return weightVecL, weightVecR


def SYS(t, S):
    weightVecL, weightVecR = weight_maker(spatial_num)
    midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, spatial_num)
    theta_step = intervals[0][-1] - intervals[0][0]

    # return vector created with while looooop
    states = np.zeros(2*spatial_num)
    i = 0
    while i < 2*spatial_num:
        if i < spatial_num:
            states[i] = -S[i] + (np.maximum(theta_step*((1/(2*np.pi)))*np.dot(weightVecL, S) + bl, 0))/Tau 
        if i >= spatial_num:
            states[i] = -S[i] + (np.maximum(theta_step*((1/(2*np.pi)))*np.dot(weightVecR, S) + br, 0))/Tau 
        i = i + 1
    return states


if __name__=='__main__':
    ## PARAMTERS TO CHANGE:______________________________________________________________________________
    # set number of spatial discretizations in each ring, must be an even number                       #|
    spatial_num = 100                                                                          
                                                                                                       #|
    # handles impulse or change to firing rates.                                                       #|
    b0 = -40                                                                                            #|
    delta_b = 20                                                                                       #|
    br = b0 + delta_b                                                                                  #|   
    bl = b0 - delta_b                                                                                  #|
                                          
    # time constant Tau
    Tau = 80
                                                                                                       #| 
    # what time window should the PDE be approximated on                                               #|
    t_span = [0, 5]     
    time_density = 1000 #number of time snapshots soln is saved at within tspan                        #|
    #s0 = 10*ic_maker_erf(spatial_num)
    #s0 = ic_maker_enum(spatial_num)
    s0 = ic_maker_erf(spatial_num)
    ## END CHANGABLE PARAMETERS. (Can also edit initial conditions in function ic_maker)_______________#|


    # define necessary constants, parameters used in solver/plotting
    t_start, t_stop = t_span
    t = np.linspace(t_start, t_stop, time_density)
    midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, spatial_num)
    theta_space, time = np.meshgrid(midpoints, t)
    theta_space = np.transpose(theta_space)
    time = np.transpose(time)


    # Finally call  solver
    # setting dense_output to True computes a continuous solution. Default is false.
    sol = solve_ivp(SYS, t_span, s0, t_eval=t, dense_output=False)

    # testing print statements        ------
    #print(time.shape)
    #print(theta_space.shape)
    #print(sol.y.shape)
    #print(sol.y[0:spatial_num, :].shape)

    #print(time)
    #print(theta_space)
    #print(sol.y)
    #print("Slcing now:")
    
    # pulls left ring soln
    print(sol.y[0:spatial_num, :])
    print("__________")
    # pulls right ring soln
    print(sol.y[spatial_num: , :])
    # end testing print statements       ------

    # plot the left ring
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time, theta_space, sol.y[0:spatial_num, :], rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Left Ring")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$s_l$')
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    filename = 'LR_' + str(spatial_num) + '.png'
    fig1.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node') # only run with this line uncommented if you are Nick


    # then the right
    fig2 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time, theta_space, sol.y[spatial_num: , :], rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Right Ring")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$s_r$')
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    filename = 'RR_' + str(spatial_num) + '.png'
    fig2.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node') # only run with this line uncommented if you are Nick



