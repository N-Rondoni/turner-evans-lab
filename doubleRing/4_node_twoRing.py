# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# Solved same problem as gillespie.py with different method
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import odeint
import os

# set number of spacial discretizations
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
    
    # define f = (sl_1', sl_2', sr_1', sr_2')
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

# define initial conditions (firing rates)
sl_1 = 0.001
sl_2 = 0.01
sl_3 = 0.01
sl_4 = 0.001
sr_1 = 0.001
sr_2 = 0.001
sr_3 = 0.001
sr_4 = 0

# define constants/reaction parameters (dependent on c, figure those out)
midpoints, intervals = thetaDivider(-np.pi, np.pi, spatial_num, spatial_num)
theta_step = intervals[0][-1] - intervals[0][0]
#print("midpoints:", midpoints)
#print("intervals:", intervals)
#print(theta_step)
theta_1 = midpoints[0]
theta_2 = midpoints[1]
theta_3 = midpoints[2]
theta_4 = midpoints[3]

 # can change b0, delta_b
b0 = -5
delta_b = 1
br = b0 + delta_b
bl = b0 = delta_b


 # Tau represents travel time of signal between synapses. Small, around 80ms. 
Tau = 0.080

# pack up parameters and ICs
p = [theta_step, theta_1, theta_2, theta_3, theta_4, bl, br, Tau]
X0 = [sl_1, sl_2, sl_3, sl_4, sr_1, sr_2, sr_3, sr_4]

# ODE solver parameters
abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 0.8
numpoints = 1000 # number of time points in [0, stoptime]

# set up time
t = np.linspace(0, stoptime, numpoints)

# call the ODE solver
sol = odeint(RHS, X0, t, args=(p,), atol=abserr, rtol=relerr) 

# in addition to firing rate vs time with all theta positions as their own line,
# plot firing rate vs theta for each ring. 

theta_space, time = np.meshgrid(midpoints, t)

#print(sol[:, 0:4]) # left ring
#print(sol[:, 4:8].shape) # right ring
#print(time.shape)
#print(theta_space.shape)

# plot the left ring

def plotfriend(sol_start, sol_stop)
fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(time, theta_space, sol[:, 0:4], rstride=1, cstride=1, cmap='viridis') #
ax.set_title("Firing Rate vs Time")
ax.set_xlabel('Time')
ax.set_ylabel(r'$\theta$')
ax.set_zlabel(r'$s_l$')
plt.colorbar(surf)
filename = '3d_plot_' + str(X0[0]) + '_' + str(X0[1]) + '_' + str(X0[2]) + '_' + str(X0[3]) + '.png'
fig.savefig(filename)
os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing') # only run with this line uncommented if you are Nick







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



#plotFriend(t, sol)



