import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time


def CRN(t, A):
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
    kf, kr, alpha, gamma, CI_tot = p

    # define chemical master equation 
    # this is the RHS of the system 

    du = [alpha*s - gamma*x + kr*z - kf*y*x, # + beta
            kr*z - kf*y*x, 
            kf*y*x - kr*z] 
 
    return du
