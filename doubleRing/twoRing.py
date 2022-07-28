# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# Solved same problem as gillespie.py with different method
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

# set number of spacial discretizations
spatial_num = 2

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
    sl_1, sl_2, sr_1, sr_2 = X
    dTheta, theta_1, theta_2, bl, br, phi, psi, Tau = p
    
    # define small pieces so f isn't too messy
    [J0, J1, K0, K1] = [-60, 80, -5, 80]
    Ws1 = J0 + J1*np.cos(phi - theta_1 - psi)
    Ws2 = J0 + J1*np.cos(phi - theta_2 - psi)

    Wd1 = K0 + K1*np.cos(phi - theta_1 - psi)
    Wd2 = K0 + K1*np.cos(phi - theta_2 - psi)


    # define f = (sl_1', sl_2', sr_1', sr_2')
    # this is the RHS of the system
    f = [(-sl_1 + np.maximum(dTheta*(1/(2*np.pi))*(Ws1*sl_1 + Wd1*sr_1 + Ws2*sl_2 + Wd2*sr_2) + bl, 0))/Tau,
         (-sl_2 + np.maximum(dTheta*(1/(2*np.pi))*(Ws1*sl_1 + Wd1*sr_1 + Ws2*sl_2 + Wd2*sr_2) + bl, 0))/Tau,
         (-sr_1 + np.maximum(dTheta*(1/(2*np.pi))*(Wd1*sl_1 + Ws1*sr_1 + Wd2*sl_2 + Ws2*sr_2) + br, 0))/Tau,
         (-sr_2 + np.maximum(dTheta*(1/(2*np.pi))*(Wd1*sl_1 + Ws1*sr_1 + Wd2*sl_2 + Ws2*sr_2) + br, 0))/Tau]
    return f

# define initial conditions (firing rates)
sl_1 = 0
sl_2 = 0.001
sr_1 = 0.001
sr_2 = 0

# define constants/reaction parameters (dependent on c, figure those out)
midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, 2)
dTheta = intervals[0][-1] - intervals[0][0]
theta_1 = midpoints[0]
theta_2 = midpoints[1]


 # can change b0, delta_b
b0 = -5
delta_b = 1
br = b0 + delta_b
bl = b0 = delta_b

 # can change phiDeg, psiDeg
phiDeg = 90
psiDeg = 45
phi = phiDeg*(np.pi)/180
psi = psiDeg*(np.pi)/180

 # Tau represents travel time of signal between synapses. Small, around 80ms. 
Tau = 0.080

# pack up parameters and ICs
p = [dTheta, theta_1, theta_2, bl, br, phi, psi, Tau]
X0 = [sl_1, sl_2, sr_1, sr_2]

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 0.8
#numpoints = 1001
numpoints = 1000 # number of time points in [0, stoptime]

# set up time
t = np.linspace(0, stoptime, numpoints)


# call the ODE solver
sol = odeint(RHS, X0, t, args=(p,), atol=abserr, rtol=relerr) 


def plotFriend(tt, sol):
    plt.plot(tt, sol[:, 0], label= "Left Ring, Theta = -pi/2")
    plt.plot(tt, sol[:, 1], label= "Left Ring, Theta = pi/2")
    plt.plot(tt, sol[:, 2], label= "Right Ring, Theta = -pi/2")
    plt.plot(tt, sol[:, 3], label= "Right Ring, Theta = pi/2")
    plt.legend(loc='upper right')
    plt.title('Dynamics of Firing Rate')
    plt.xlabel('time')
    plt.ylabel('Firing Rate (Hz)')
    filename = 'Firing_Rate_discr_'+ str(X0[0]) + '_' + str(X0[1]) + '_' + str(X0[2]) + '_' + str(X0[3]) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing') # only run with this line uncommented if you are Nick
    plt.show()



plotFriend(t, sol)



