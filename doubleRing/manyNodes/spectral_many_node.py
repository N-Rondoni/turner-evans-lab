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


def ic_maker_periodic(spatial_discretizations):
    n = 2*spatial_discretizations + 2
    thetas = thetaDivider(-np.pi, np.pi, spatial_discretizations)
    s0 = np.sin(thetas)
    s0 = np.maximum(s0, 0)
    return s0


def ws(theta):
    """
    weight function for the "same" ring. 
    
    Arguments: 
            Theta: value of theta on the grid.
    """
    #marginal params: 
    [J0, J1] = [-60, 80]

    # divergent params: 
    #[J0, J1] = [60, 80]

    
    # homogenous params:
    #[J0, J1] = [-60, -80]


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
    # marginal params:
    [K0, K1] = [-5, 80]
    
    # divergent params:
    #[K0, K1] = [100, 80]


    # homogenous params: 
    #[K0, K1] = [-100, -80]
    # can change the offset connection points with these angles below
    phiDeg = 80
    psiDeg = 50
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightDifferent = K0 + K1*np.cos(phi - theta - psi)
    return weightDifferent


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
    weightMatrix = ((1/(2*np.pi))/(spatial_num + 1))*weightMatrix # this could be wrong. 
    #print(weightMatrix)                                       # divide by n? mult by 2pi? 
    return weightMatrix
    

def SYS(t, S):
    '''
    spatial_num, an even number, must be defined in main before you can call this function.
    '''
    weightMat = weight_mx(spatial_num)
    thetas = thetaDivider(-np.pi, np.pi, spatial_num)

    states = np.zeros(2*spatial_num + 2)
    id_mat = np.identity(2*spatial_num + 2)
    sys_mat = (weightMat -  id_mat) #if weight mat depends on random parameters, this will as well. 

    states = np.dot(sys_mat, S)

    #print(states)
    #print(states.shape)

    # add feedforward/feedbackwards input
    for i in range(0, spatial_num + 1):
        states[i] = states[i] + bl
    for i in range(spatial_num + 1, 2*spatial_num + 2):
        states[i] = states[i] + br

    states = np.maximum(states, 0)
    states = states/Tau

    return states


def interp_poly_vecs(spatial_num, x):
    '''
    returns a matrix, each column of which is g(theta_i)
    '''
    #theta = thetaDivider(-np.pi, np.pi, spatial_num)
    
    thetas = np.linspace(-np.pi, np.pi, spatial_num + 1)

    g_mat = np.zeros((2*spatial_num + 2, 2*spatial_num + 2))
    
    n = spatial_num + 1

    i = 0
    while i < spatial_num + 1:
        j = 0
        while j < spatial_num + 1: #causes divide by zero errors.
            g_mat[j, i] = (1/n)*np.sin(n *(x[j] - thetas[i])/2)*(1/(np.tan((x[j] - thetas[i])/2)))
            j = j+1
        i = i+1
    
    print(g_mat)
    
    return g_mat



if __name__=='__main__':
    ## PARAMTERS TO CHANGE:______________________________________________________________________________
    # set number of spatial discretizations in each ring, must be an even number                       #|
    spatial_num = 100                                                                          
                                                                                                       #|
    # handles impulse or change to firing rates.                                                       #|
    b0 = 40                                                                                            #|
    delta_b = -8                                                                                       #|
    br = b0 + delta_b                                                                                  #|   
    bl = b0 - delta_b                                                                                  #|
                                          
    # time constant Tau
    Tau = 80
                                                                                                       #| 
    # what time window should the PDE be approximated on                                               #|
    t_span = [0, 5]     
    time_density = 1000 #number of time snapshots soln is saved at within tspan                        #|
    s0 = ic_maker_periodic(spatial_num)
    
    # END CHANGABLE PARAMETERS. (Can also edit initial conditions in function ic_maker)_______________#|


    # define necessary constants, parameters used in solver/plotting
    t_start, t_stop = t_span
    t = np.linspace(t_start, t_stop, time_density)
    #theta_grid = thetaDivider(-np.pi, np.pi, spatial_num)
    theta_grid = np.linspace(-np.pi, np.pi, spatial_num + 1)
    weight_mx(spatial_num)


    theta_space, time = np.meshgrid(theta_grid, t)
    theta_space = np.transpose(theta_space)
    time = np.transpose(time)


    # Finally call  solver
    # setting dense_output to True computes a continuous solution. Default is false.
    
    sol = solve_ivp(SYS, t_span, s0, t_eval=t, dense_output=False)

    # sol becomes coeff for interpolating polynomial
    
    # lr/rr solution, each column a time instance.
    lr_sol = sol.y[0:spatial_num+1, :],
    rr_sol = sol.y[spatial_num+1:, :],

    # finally interpolat

    #interp_poly_vecs(spatial_num, theta_grid)

    # testing print statements        -------------------------------
    #print(time.shape)
    #print(theta_space.shape)
    #print(sol.y.shape)
    #print(sol.y[0:spatial_num, :].shape)

    #print(time)
    #print(theta_space)
    #print(sol.y)
    #print("Slcing now:")


    # pulls left ring soln
    #print(sol.y[0:spatial_num, :])
    #print("__________")
    # pulls right ring soln
    print(sol.y[spatial_num+1: , :].shape)
    print(sol.y[0:spatial_num+1, :].shape)
    print(time.shape)
    print(theta_space.shape)
    # end testing print statements       ------------------------------ 

    # plot the left ring
    fig1 = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time, theta_space, sol.y[0:spatial_num+1, :], rstride=1, cstride=1, cmap='viridis') 
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
    surf = ax.plot_surface(time, theta_space, sol.y[spatial_num+1: , :], rstride=1, cstride=1, cmap='viridis') 
    ax.set_title("Firing Rate vs Time in Right Ring")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$s_r$')
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    filename = 'RR_' + str(spatial_num) + '.png'
    fig2.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node') # only run with this line uncommented if you are Nick



