# Purposes: deterministic/ continuous apprpoximation of ODEs derived from chemical reaction network
# attempts to back out firing rate via proportional feedback given recorded concentrations of CI^* (from Janelia)  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.io
import os
import sys
import time
import do_mpc
from casadi import *
 
def tvp_fun(t_now):
    for k in range(n_horizon + 1):
            tvp_template['_tvp', k, 'Ci_m'] = np.interp(t_now, timeVec, CI_Meas)
    return tvp_template

   

tPrev = 0
ePrev = 0
eSum = 0
#sVec = np.array([0])

def CRN(t, A):
    global tPrev
    global ePrev
    global eSum
    """
    Defines the differential equations for a coupled chemical reaction network.
    
    Arguments: 
        A : vector of the state variables: Ca^{2+}, CI, CI^* respectively. 
                A = [x, y, z]
        t : time
        p : vector of the parameters:
                p = [kf, kr, alpha, gamma]
    """
    x, y, z = A
    #kf, kr, alpha, gamma, beta = p
    kf, kr, alpha, gamma, kProp, kDer, kInt = p

    # define chemical master equation 
    # this is the RHS of the system     
    
    # interpolate because data doesn't have values for all times used by solver.
    CI_MeasTemp = np.interp(t, timeVec, CI_Meas[:len(timeVec)])
    
    # Keep current error to compute eProp, and der for next pass. 
    # proportional portion
    eCurrent = (z - CI_MeasTemp)
    
    # derivative portion
    if t == tPrev:
        eDer = 0
    else:
        eDer = (eCurrent - ePrev)/(t - tPrev) 

    # integral portion (approximatd with Reimann type sum)
    eSum = eSum + (t - tPrev)*eCurrent
    
    s = -kProp*eCurrent + -kDer*eDer + -kInt*eSum 

    #print(t, s)
    du = [alpha*s - gamma*x + kr*z - kf*y*x, # + beta
        kr*z - kf*y*x, 
        kf*y*x - kr*z] 
    
    #print(t, "////", du) 

    tPrev = t
    ePrev = eCurrent

    return du

def plotThreeLines(x, y1, y2, y3):
    plt.figure(1)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$')
    plt.plot(x, y3, linewidth = 2, label = r'$CI^{*}$')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig1_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
    #plt.show() 

def plotErr(x, y):
    plt.figure(2)
    plt.plot(x, y, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick




if __name__=="__main__":
    # load in necessary data
    # Load in Dan's flour. data from janelia. Using cond{1}.allFlyData{1}.Strip{2}.RROIavMax
    file_path = 'data/flymat.mat'
    mat = scipy.io.loadmat(file_path)
    data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
    m,n = data.shape
    # step through rows: hope to see bump travel across all. rows are stepped through with driver. 
    row = int(sys.argv[1])
    # pull corresponding row of data
    CI_Meas = data[row, :] # looks at a single neuron.  
    CI_Meas = 50*CI_Meas

    # load in imaging times
    file_path2 = "data/time_cond1_allFly1_stripe2.mat"
    mat = scipy.io.loadmat(file_path2)
    temp = mat['time']
    (m1, n1) = temp.shape
    timeVec = np.zeros(m1)
    timeVec = temp[:, 0] 
    tEnd = timeVec[-1]  
    
    # follow MPC example ``batch bioreactor`` on do-mpc website
    model_type = 'continuous' # or discrete
    model = do_mpc.model.Model(model_type)
    
    # states struct, optimization variables
    # S is an unknown parameter. _x denotes var, _p param?
    Ca = model.set_variable('_x', 'Ca')
    CiF = model.set_variable('_x', 'CiF')
    Ci = model.set_variable('_x', 'Ci') #

    # define ODEs and parameters, kr << kf
    kf = 0.0513514
    kr = 7.6 
    alpha = 1
    gamma = 1
    s = model.set_variable('_u', 's')   # control variable ( input )
    CI_m = model.set_variable('_tvp', 'Ci_m') # timve varying parameter, or just hardcode

    model.set_rhs('Ca', alpha*s - gamma*Ca + kr*CiF - kf*Ci*Ca)
    model.set_rhs('Ci', kr*CiF - kf*Ci*Ca)
    model.set_rhs('CiF', kf*Ci*Ca - kr*CiF)
   
    model.setup()
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters, can change collocation/state discretization here.
    # does not impact my actual horizon or stepsize for simulation? 
    setup_mpc = {
            'n_horizon': 6, # pretty short horizion
            't_step': 1/6, # (s)
            'n_robust': 1,
            'store_full_solution': True,
            }
    
    mpc.set_param(**setup_mpc)
    n_horizon = 6
    t_step = 1/6
    n_robust = 1


    # define objective, which is to miminize the difference between Ci_m and Ci. 
    mterm = (model.x['Ci'] - model.tvp['Ci_m'])**2                   # terminal cost
    lterm = mterm                                                    # stage cost 
 
    mpc.set_objective(mterm = mterm, lterm = lterm)
    mpc.set_rterm(s=1.0) # sets a penalty on changes in s

    # make sure the objective/cost updates with CI_measured and time.    
    tvp_template = mpc.get_tvp_template()
    
    mpc.set_tvp_fun(tvp_fun)


    # define constraints
    mpc.bounds['lower', '_x', 'Ca'] = 0.0
    mpc.bounds['lower', '_x', 'Ci'] = 0.0
    mpc.bounds['lower', '_x', 'CiF'] = 0.0

    mpc.bounds['lower', '_u', 's'] = - 0.1 # slow diffusion


    mpc.setup()
    
    # Estimator: assume all states can be directly measured
    estimator = do_mpc.estimator.StateFeedback(model)


    # Simulator
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
            'integration_tool': 'cvodes', # look into this
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 1/6, # (s)
            }

    simulator.set_param(**params_simulator)
    simulator.setup()




    # compute total time, (imaging was done at 11.4hz), create vec of time samples timeVec
    # timeVec used within solver to compute interpolated value of S at a particular time. np.interp(t, timeVec, CI_Meas)
    imRate = 1/6
    tEnd = n*(imRate)
    timeVec = np.linspace(0, tEnd, n)
    # short time testing uncomment below ###
    #tEnd = .1
    #nDatPoints = 5
    #timeVec = np.linspace(0, tEnd, nDatPoin

    # define initial conditions
    X = 0 #Ca^{2+}
    Y = 50 #CI
    Z = 50  #CI^* #was previously 0, real data readouts start with some concentration.

        # gain coeff
    kProp = 100
    kDer = 1
    kInt = 1
    
    # pack up parameters and ICs
    p = [kf, kr, alpha, gamma, kProp, kDer, kInt]
    u0 = [X, Y, Z]

