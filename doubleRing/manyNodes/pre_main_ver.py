# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# Solved same problem as gillespie.py with different method
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os

# set number of spacial discretizations, MUST BE AN EVEN NUMBER
spatial_num = 10

# handles offsets in ring connections
b0 = 40
delta_b = 30
br = b0 + delta_b
bl = b0 - delta_b

# what time window should the PDE be approximated on
t_span = [0, 1]

def thetaDivider(thetaStart, thetaStop, n, thetaDensity):
    """
    Creates subintervals and needed midpoints for spacial discretization

    Arguments: 
        thetaStart   : starting spatial value (real number)
        thetaStop    : ending spatial value (real number)
        n            : number of subintervals
        thetaDensity : number of values in each subinterval   TODO: refactor, thetaDensity is irrelevant

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


def ic_maker(spatial_discretizations):
    n = 2*spatial_discretizations
    i = 0
    s0 = np.zeros(n)
    while i < n:
        s0[i] = i
        i = i + 1
    #print(s0)
    return s0


def weight_maker(spatial_discretizations):
    n = 2*spatial_discretizations
    weightVec = np.zeros(n)
    midpoints, intervals = thetaDivider(-np.pi, np.pi, n, n)
    i = 0
    while (i < n):
        if i < n/2:
            weightVec[i] = ws(midpoints[i])
        if i >= n/2: 
            weightVec[i] = wd(midpoints[i])
        i = i + 1
    #print(weightVec)
    return weightVec


def SYS(t, S):
    weightVec = weight_maker(spatial_num)
    midpoints, intervals = thetaDivider(-np.pi, np.pi, 2*spatial_num, 2*spatial_num)
    theta_step = intervals[0][-1] - intervals[0][0]

    # return vector created with while looooop
    states = np.zeros(2*spatial_num)
    i = 0
    while i < 2*spatial_num:
        if i < spatial_num:
            states[i] = -S[i] + np.maximum(theta_step*((1/(2*np.pi)))*np.dot(weightVec, S) + bl, 0) 
        if i >= spatial_num:
            states[i] = -S[i] + np.maximum(theta_step*((1/(2*np.pi)))*np.dot(weightVec, S) + br, 0) 
        i = i + 1
    return states




# define necessary constants, parameters used in solver/plotting

t_start, t_stop = t_span
t = np.linspace(t_start, t_stop, 100)
midpoints, intervals = thetaDivider(-np.pi, np.pi, 2*spatial_num, 2*spatial_num)
theta_space, time = np.meshgrid(midpoints, t)
s0 = ic_maker(spatial_num)


# Finally call  solver
# setting dense_output to True computes a continuous solution. Default is false.
sol = solve_ivp(SYS, t_span, s0, t_eval=t, dense_output=False)

print(sol.y)
print(sol.t.shape)
print(sol.t)






# create meshgrids in order to plot
#t_start, t_stop = t_span
#t = np.linspace(t_start, t_stop, 100)
#midpoints, intervals = thetaDivider(-np.pi, np.pi, 2*spatial_num, 2*spatial_num)
#theta_space, time = np.meshgrid(midpoints, t)

# create jank plot functions, probably shouldn't do this. 
def plotfriend_left(sol_start, sol_stop):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time, theta_space, sol[:, sol_start:sol_stop], rstride=1, cstride=1, cmap='viridis') #
    ax.set_title("Firing Rate vs Time")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$s_l$')
    #ax.view_init(30, 132) #uncomment to see backside
    plt.colorbar(surf)
    filename = 'LR_' + str(X0[0]) + '_' + str(X0[1]) + '_' + str(X0[2]) + '_' + str(X0[3]) + '.png'
    fig.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/new_plots') # only run with this line uncommented if you are Nick


def plotfriend_right(sol_start, sol_stop):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(time, theta_space, sol[:, sol_start:sol_stop], rstride=1, cstride=1, cmap='viridis') #
    ax.set_title("Firing Rate vs Time")
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\theta$')
    ax.set_zlabel(r'$s_l$')
    #ax.view_init(30, 180) uncomment to see backside
    plt.colorbar(surf)
    filename = 'RR_' + str(X0[0]) + '_' + str(X0[1]) + '_' + str(X0[2]) + '_' + str(X0[3]) + '.png'
    fig.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/new_plots') # only run with this line uncommented if you are Nick



#plotfriend_left(0, 4)
#plotfriend_right(4, 8)


