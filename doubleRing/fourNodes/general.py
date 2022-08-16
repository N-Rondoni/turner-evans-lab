import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import os



spatial_num = 4

def thetaDivider(thetaStart, thetaStop, n, thetaDensity):
    """
    Creates subintervals and needed midpoints for spacial discretization

    Arguments: 
        thetaStart   : starting spacial value (real number)
        thetaStop    : ending spacial value (real number)
        n            : number of subintervals
        thetaDensity : number of values in each subinterval

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
    [J0, J1] = [-60, 80]
    # can change the offset connection points with these angles below
    phiDeg = 90
    psiDeg = 45
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightSame = J0 + J1*np.cos(phi - theta - psi)
    return weightSame


def RHS(X, t, p):
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        X : vector of the firing rates for varying theta:
                X = [theta_1, theta_2, ... , theta_spacial_num]
        t : time
        p : vector of the parameters:
                p = [dTheta, theta_1, theta_2, bl, br, phi, psi, Tau]
    """
    sl_1, sl_2, sl_3, sl_4, sr_1, sr_2, sr_3, sr_4 = X
    theta_step, theta_1, theta_2, theta_3, theta_4, bl, br, Tau = p
    
    # define f = [sl_1', sl_2', sl_3', sl_4', sr_1', sr_2', sr_3', sr_4']
    # this is the RHS of the system
    f = [(-sl_1 + np.maximum(theta_step*(1/(2*np.pi))*(ws(theta_1)*sl_1 + wd(theta_1)*sr_1 + ws(theta_2)*sl_2 + wd(theta_2)*sr_2 + ws(theta_3)*sl_3 + wd(theta_3)*sr_3 + ws(theta_4)*sl_4 + wd(theta_4)*sr_4) + bl, 0))/Tau,
         (-sl_2 + np.maximum(theta_step*(1/(2*np.pi))*(ws(theta_1)*sl_1 + wd(theta_1)*sr_1 + ws(theta_2)*sl_2 + wd(theta_2)*sr_2 + ws(theta_3)*sl_3 + wd(theta_3)*sr_3 + ws(theta_4)*sl_4 + wd(theta_4)*sr_4) + bl, 0))/Tau,
         (-sl_3 + np.maximum(theta_step*(1/(2*np.pi))*(ws(theta_1)*sl_1 + wd(theta_1)*sr_1 + ws(theta_2)*sl_2 + wd(theta_2)*sr_2 + ws(theta_3)*sl_3 + wd(theta_3)*sr_3 + ws(theta_4)*sl_4 + wd(theta_4)*sr_4) + bl, 0))/Tau,
         (-sl_4 + np.maximum(theta_step*(1/(2*np.pi))*(ws(theta_1)*sl_1 + wd(theta_1)*sr_1 + ws(theta_2)*sl_2 + wd(theta_2)*sr_2 + ws(theta_3)*sl_3 + wd(theta_3)*sr_3 + ws(theta_4)*sl_4 + wd(theta_4)*sr_4) + bl, 0))/Tau,

         (-sr_1 + np.maximum(theta_step*(1/(2*np.pi))*(wd(theta_1)*sl_1 + ws(theta_1)*sr_1 + wd(theta_2)*sl_2 + ws(theta_2)*sr_2 + wd(theta_3)*sl_3 + ws(theta_3)*sr_3 + wd(theta_4)*sr_4 + ws(theta_4)*sr_4) + br, 0))/Tau,
         (-sr_2 + np.maximum(theta_step*(1/(2*np.pi))*(wd(theta_1)*sl_1 + ws(theta_1)*sr_1 + wd(theta_2)*sl_2 + ws(theta_2)*sr_2 + wd(theta_3)*sl_3 + ws(theta_3)*sr_3 + wd(theta_4)*sr_4 + ws(theta_4)*sr_4) + br, 0))/Tau,
         (-sr_3 + np.maximum(theta_step*(1/(2*np.pi))*(wd(theta_1)*sl_1 + ws(theta_1)*sr_1 + wd(theta_2)*sl_2 + ws(theta_2)*sr_2 + wd(theta_3)*sl_3 + ws(theta_3)*sr_3 + wd(theta_4)*sr_4 + ws(theta_4)*sr_4) + br, 0))/Tau,
         (-sr_4 + np.maximum(theta_step*(1/(2*np.pi))*(wd(theta_1)*sl_1 + ws(theta_1)*sr_1 + wd(theta_2)*sl_2 + ws(theta_2)*sr_2 + wd(theta_3)*sl_3 + ws(theta_3)*sr_3 + wd(theta_4)*sr_4 + ws(theta_4)*sr_4) + br, 0))/Tau]
    return f



# function that generates from sl_i + onwards for any n

def RHSconstructor(n):
    # takes spacial discretizations, 
    # returns f in the above and X that is passed in



