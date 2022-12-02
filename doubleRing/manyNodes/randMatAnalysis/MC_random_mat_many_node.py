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
    weight function for the "same" ring. CONTAINS RANDOM PARAMETERS: J0, J1
    
    Arguments: 
            Theta: value of theta on the grid.
    """
    # want perturbations of the following:
    [J0, J1] = [-60, 80] # marginal 
    #[J0, J1] = [60, 80] # divergent
    #[J0, J1] = [-60, -80] # homogenous

    #std_dev = 200
    #J0 = np.random.normal(-60, std_dev)
    #J1 = np.random.normal(80, std_dev)

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
    # want perturbations of the following:
    [K0, K1] = [-5, 80] # marginal 
    #[K0, K1] = [100, 80] # divergent
    #[K0, K1] = [-100, -80] # homogenous

    #std_dev = 200
    #K0 = np.random.normal(-5, std_dev)
    #K1 = np.random.normal(80, std_dev)

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
    weightMatrix = (1/(2*np.pi*(spatial_num + 1)))*weightMatrix # this could be wrong. 
    #weightMatrix = (1/(spatial_num + 1))*weightMatrix # this could be wrong. 

    #print(weightMatrix)
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

    #eVals, eVecs = np.linalg.eig(sys_mat)
    #np.save("eVals_random_matrix", eVals)

    states = np.dot(sys_mat, S)
    
    b0 = 40                                                                                            #|
    delta_b = -8                                                                                       #|
    br = b0 + delta_b                                                                                  #|   
    bl = b0 - delta_b                                                                                  #|
     

    # add feedforward/feedbackwards input
    for i in range(0, spatial_num + 1):
        states[i] = states[i] + bl
    for i in range(spatial_num + 1, 2*spatial_num + 2):
        states[i] = states[i] + br

    states = np.maximum(states, 0)
    states = states/Tau
    
    S = states

    return states


if __name__=='__main__':
    ## PARAMTERS TO CHANGE:______________________________________________________________________________
    # set number of spatial discretizations in each ring, must be an even number                       #|
    spatial_num = 100
                                                                                                       #|
    # handles impulse or change to firing rates.                                                       #|
    b0 = -40                                                                                            #|
    delta_b = 8                                                                                       #|
    br = b0 + delta_b                                                                                  #|   
    bl = b0 - delta_b                                                                                  #|
                                          
    # time constant Tau
    Tau = 80
                                                                                                       #| 
    # what time window should the PDE be approximated on                                               #|
    t_span = [0, 10]     
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

    # finally interpolate

    # fix a moment in time:
    lr_particular = sol.y[0:spatial_num+1, 0]
    print(lr_particular.shape)
    print(theta_grid.shape)
    # create g_mat, rows are vector g for varying k., theta_grid is the  theta_k 
    g_mat = np.zeros((spatial_num+1, spatial_num+1))
    for i in range(0, spatial_num+1):
        for j in range(0, spatial_num+1):
            if (i == j):
                g_mat[i, j] = i
            else: 
                g_mat[i, j] = (1/(spatial_num+1))*np.sin((spatial_num+1)*(theta_grid[i] - theta_grid[j])/2)*(1/np.tan((spatial_num+1)*(theta_grid[i] - theta_grid[j])/2))
    
    #print(g_mat)
    #print(g_mat.shape)
    #print(lr_particular)
    #print(lr_particular * g_mat[0,:])

   

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
    #print(sol.y[spatial_num+1: , :].shape)
    #print(sol.y[0:spatial_num+1, :].shape)
    #print(time.shape)
    #print(theta_space.shape)
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
    filename = 'LR_random' + str(spatial_num) + '.png'
    fig1.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node/UQ') # only run with this line uncommented if you are Nick


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
    filename = 'RR_random' + str(spatial_num) + '.png'
    fig2.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node/UQ') # only run with this line uncommented if you are Nick



