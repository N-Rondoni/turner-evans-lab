# Purpose:
# Use:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os
import time as soup



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
    #s0[spatial_discretizations+1:-1] = np.flip(s0[spatial_discretizations+1:-1]) # makes right ring IC mirror image
    return s0


def ws(theta):
    """
    weight function for the "same" ring. CONTAINS RANDOM PARAMETERS: J0, J1
    
    Arguments: 
            Theta: value of theta on the grid.
    """
    # want perturbations of the following:
    #[J0, J1] = [-60, 80] # marginal 
    #[J0, J1] = [60, 80] # divergent
    #[J0, J1] = [-60, -80] # homogenous
    
    # if you randomize here, recursion is too intense, takes too long to finish.

    std_dev = 200
    J0 = np.random.normal(-60, std_dev)
    J1 = np.random.normal(80, std_dev)

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
    #[K0, K1] = [-5, 80] # marginal 
    #[K0, K1] = [100, 80] # divergent
    #[K0, K1] = [-100, -80] # homogenous

    # if you randomize here, recursion is too intense, takes too long to finish.

    std_dev = 200
    K0 = np.random.normal(-5, std_dev)
    K1 = np.random.normal(80, std_dev)

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
    #
    states = np.dot(sys_mat, S)
     

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
    startTime = soup.perf_counter()
    

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
    time_density = 200 #number of time snapshots soln is saved at within tspan                        #|
    s0 = ic_maker_periodic(spatial_num) 
    # END CHANGABLE PARAMETERS. (Can also edit initial conditions in function ic_maker)_______________#|

    # define necessary constants, parameters used in solver/plotting
    #theta_grid = thetaDivider(-np.pi, np.pi, spatial_num)
    theta_grid = np.linspace(-np.pi, np.pi, spatial_num + 1)


        
    MC_iter = 1000

    max_eig = np.zeros(1)
    for i in range(0, MC_iter):
        weightMat = weight_mx(spatial_num)
        [m, n] = weightMat.shape
        # m = n for the weight mat. 
        I = np.eye(int(m))
        eVals, eVecs = np.linalg.eig(weightMat - I)
        max_eig = np.append(max_eig, max(eVals, key = abs)) 

    # remove first entry of zero:
    max_eig = max_eig[1:]
    print(len(max_eig))
    np.save("maximal_eVals_random_matrix", max_eig)

    #print(max_eig)
    print("________________________________________________________")
   
    realPart = np.real(max_eig)
    pos_counter = np.zeros(len(realPart))

    for i in range(len(realPart)):
        if realPart[i] >= 0:
            pos_counter[i] = 1
    print(np.sum(pos_counter), "out of", MC_iter, "had positive real part of their maximal eigenvalue")
    print("or as a percentage: ", np.sum(pos_counter)/MC_iter)
    
    stopTime = soup.perf_counter()
    print("Time to complete run:", stopTime - startTime, "s")
    

