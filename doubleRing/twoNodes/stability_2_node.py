# solves a nonlinear system of 4 eqns to find stable state in space of firing rates
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

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

def wd(theta):
    """
    weight function for the "different" ring. 
    
    Arguments: 
            Theta: the midpoint of our discretization intervals
    """
    [K0, K1] = [-5, 80]
    # can change the offset connection points with these angles below
    phiDeg = 90
    psiDeg = 45
    phi = phiDeg*(np.pi)/180
    psi = psiDeg*(np.pi)/180
    weightDifferent = K0 + K1*np.cos(phi - theta - psi)
    return weightDifferent

spatial_num = 2
midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, 2)
del_theta = intervals[0][-1] - intervals[0][0]
theta_1 = midpoints[0]
theta_2 = midpoints[1]
print(theta_1)

def sys(sl1, sl2, sr1, sr2):
    # theta step size, comes from system
    #print(ws(theta_1))
    return [-sl1 + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*sl1 + wd(theta_1)*sr1 + ws(theta_2)*sl2 + wd(theta_2)*sr2),
            -sl2 + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*sl1 + wd(theta_1)*sr1 + ws(theta_2)*sl2 + wd(theta_2)*sr2),
            -sr1 + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*sr1 + wd(theta_1)*sl1 + ws(theta_2)*sr2 + wd(theta_2)*sl2),
            -sr2 + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*sr1 + wd(theta_1)*sl1 + ws(theta_2)*sr2 + wd(theta_2)*sl2)]

def sys2(S):
    # S = [sl1, sl2, sr1, sr2]
    #print(ws(theta_1))
    return [-S[0] + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[2] + ws(theta_2)*S[1] + wd(theta_2)*S[3]),
            -S[1] + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[2] + ws(theta_2)*S[1] + wd(theta_2)*S[3]),
            -S[2] + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[2] + wd(theta_1)*S[0] + ws(theta_2)*S[3] + wd(theta_2)*S[1]),
            -S[3] + ((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[2] + wd(theta_1)*S[0] + ws(theta_2)*S[3] + wd(theta_2)*S[1])]


x0 = [100, 10, 60, 100]
fixed_point = fsolve(sys2, x0)
print(fixed_point)



