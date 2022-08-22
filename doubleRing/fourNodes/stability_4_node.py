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

spatial_num = 4
midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, spatial_num)
del_theta = intervals[0][-1] - intervals[0][0]

theta_1, theta_2, theta_3, theta_4 = midpoints[0], midpoints[1], midpoints[2], midpoints[3]
#print(del_theta)
#print(theta_1)
#print(theta_2)
#print(theta_3)
#print(theta_4)

def sys(S):
    b0 = 40
    delta_b = 30
    br = b0 + delta_b
    bl = b0 - delta_b

    Tau = 0.080 # time constant in ms

    # S = [sl1, sl2, sl3, sl4, sr1, sr2, sr3, sr4]
    return [(-S[0] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[4] + ws(theta_2)*S[1] + wd(theta_2)*S[5] + ws(theta_3)*S[2] + wd(theta_3)*S[6] + ws(theta_4)*S[3] + wd(theta_4)*S[7]) + bl, 0))/Tau,
            (-S[1] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[4] + ws(theta_2)*S[1] + wd(theta_2)*S[5] + ws(theta_3)*S[2] + wd(theta_3)*S[6] + ws(theta_4)*S[3] + wd(theta_4)*S[7]) + bl, 0))/Tau,
            (-S[2] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[4] + ws(theta_2)*S[1] + wd(theta_2)*S[5] + ws(theta_3)*S[2] + wd(theta_3)*S[6] + ws(theta_4)*S[3] + wd(theta_4)*S[7]) + bl, 0))/Tau, 
            (-S[3] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + wd(theta_1)*S[4] + ws(theta_2)*S[1] + wd(theta_2)*S[5] + ws(theta_3)*S[2] + wd(theta_3)*S[6] + ws(theta_4)*S[3] + wd(theta_4)*S[7]) + bl, 0))/Tau, 
                                                 # These terms are the same as the above, just reordered with a more obvious pattern
            (-S[4] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + ws(theta_2)*S[1] + ws(theta_3)*S[2] + ws(theta_3)*S[3] + wd(theta_1)*S[4] + wd(theta_2)*S[5] + wd(theta_3)*S[6] + wd(theta_4)*S[7]) + br, 0))/Tau,
            (-S[5] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + ws(theta_2)*S[1] + ws(theta_3)*S[2] + ws(theta_3)*S[3] + wd(theta_1)*S[4] + wd(theta_2)*S[5] + wd(theta_3)*S[6] + wd(theta_4)*S[7]) + br, 0))/Tau,
            (-S[6] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + ws(theta_2)*S[1] + ws(theta_3)*S[2] + ws(theta_3)*S[3] + wd(theta_1)*S[4] + wd(theta_2)*S[5] + wd(theta_3)*S[6] + wd(theta_4)*S[7]) + br, 0))/Tau,
            (-S[7] + np.maximum(((1/(2*np.pi)))*(del_theta)*(ws(theta_1)*S[0] + ws(theta_2)*S[1] + ws(theta_3)*S[2] + ws(theta_3)*S[3] + wd(theta_1)*S[4] + wd(theta_2)*S[5] + wd(theta_3)*S[6] + wd(theta_4)*S[7]) + br, 0))/Tau]


init_guess = [0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.3, 0.1]

#sys(init_guess)
fixed_point = fsolve(sys, init_guess)
print(fixed_point)



