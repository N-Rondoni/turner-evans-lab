# Purpose:
# Use:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os


def thetaDivider(thetaStart, thetaStop, n):
    """
    Creates subintervals and needed midpoints for spacial discretization

    Arguments: 
        thetaStart   : starting spatial value (real number)
        thetaStop    : ending spatial value (real number)
        n            : number of points in the interval

    """
    theta_grid = np.linspace(thetaStart, thetaStop, num = n+1, endpoint=False) 
    theta_grid = np.concatenate((theta_grid, theta_grid))
    return theta_grid

# create two subintervals comprised of two theta values each. Store as intervals.
# create an array of the midpoints of the intervals.

def ws(theta):
    """
    weight function for the "same" ring. 
    
    Arguments: 
            Theta: value of theta on the grid.
    """
    [J0, J1] = [-60, 80]
    # can change the offset connection points with these angles below
    phiDeg = 80
    psiDeg = 50
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightSame = J0 + J1*np.cos(phi - theta - psi)
    return weightSame


def wd(theta):
    """
    weight function for the "different" ring. 
    
    Arguments: 
            Theta: value of theta on the grid
    """
    [K0, K1] = [-5, 80]
    # can change the offset connection points with these angles below
    phiDeg = 80
    psiDeg = 50
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightDifferent = K0 + K1*np.cos(phi - theta - psi)
    return weightDifferent


def weight_maker(spatial_discretizations):
    n = int(2*spatial_discretizations)
    
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


def weight_mx_old(spatial_num):
    thetas = thetaDivider(-np.pi, np.pi, spatial_num)
    n = 2*spatial_num + 2
    weightMatrix = np.zeros((n, n))
    i = 0
    # fill first row
    while i < n:
        if i < n/2:
            weightMatrix[0, i] = thetas[i]
        if i >= n/2:
            weightMatrix[0, i] = thetas[i]
        i = i+1  
    # fill other rows with first row. 
    j = 0
    while j < n:
        weightMatrix[j, :] = weightMatrix[0, :]
        j = j+1
    #print(weightMatrix)     
    return weightMatrix
    #print("weight mx thetas:")
    #print(thetas)


## OR:
def weight_mx(spatial_num):
    thetas = thetaDivider(-np.pi, np.pi, spatial_num)
    n = 2*spatial_num + 2
    weightMatrix = np.zeros((n, n))
    i = 0
    while (i < n):
        j = 0
        while (j < n):
            if j < n/2:
                weightMatrix[i, j] = ws(thetas[i] - thetas[j])
            if j >= n/2: 
                weightMatrix[i, j] = wd(thetas[i] - thetas[j])
            j = j+1
        i = i+1
    #print(weightMatrix)
    weightMatrix = ((2*np.pi)/(spatial_num + 1))*weightMatrix # this could be wrong. 
    print(weightMatrix)                                      # should divide by 2pi? 
    return weightMatrix
    #

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
    spatial_num = 2                                                                          
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
    
    # END CHANGABLE PARAMETERS. (Can also edit initial conditions in function ic_maker)_______________#|


    # define necessary constants, parameters used in solver/plotting
    t_start, t_stop = t_span
    t = np.linspace(t_start, t_stop, time_density)
    theta_grid = thetaDivider(-np.pi, np.pi, spatial_num)
    weight_mx_old(spatial_num)
    weight_mx(spatial_num)


    theta_space, time = np.meshgrid(theta_grid, t)
    theta_space = np.transpose(theta_space)
    time = np.transpose(time)



    # Finally call  solver
    # setting dense_output to True computes a continuous solution. Default is false.
    
    #sol = solve_ivp(SYS, t_span, s0, t_eval=t, dense_output=False)

    # testing print statements        ------
    #print(time.shape)
    #print(theta_space.shape)
    #print(sol.y.shape)
    #print(sol.y[0:spatial_num, :].shape)

    #print(time)
    #print(theta_space)
    #print(sol.y)
    #print("Slcing now:")
    

    quit() 

    # pulls left ring soln
    #print(sol.y[0:spatial_num, :])
    #print("__________")
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



