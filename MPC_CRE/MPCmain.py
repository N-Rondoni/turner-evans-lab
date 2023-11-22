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
        tvp_template['_tvp', k, 'Ci_m'] = np.interp(t_now + k*t_step, timeVec, CI_Meas)
    return tvp_template

def tvp_fun_sim(t_now):
    tvp_template1['Ci_m'] = np.interp(t_now, timeVec, CI_Meas)
    return tvp_template1

def plotThreeLines(x, y1, y2, y3):
    plt.figure(1)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$ (mol)')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$ (mol)')
    plt.plot(x, y3, linewidth = 2, label = r'$CI^{*}$ (mol)')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration', fontsize = 14)
    plt.legend()
    filename = 'MPC_fig1_node' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
  

def plotFourLines(x, y1, y2, y3, y4):
    plt.figure(2)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$ (mol)')
    plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$ (mol)')
    plt.plot(x, y3, '--', linewidth = 1, label = r'$CI^{*}$ (mol)')
    plt.plot(x, y4, '--', linewidth = 1, label = r'$s$ (Hz)')
    plt.title('Dynamics Derived from CRN', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Concentration (mol), Rate (Hz)', fontsize = 14)
    plt.legend()
    filename = 'MPC_fig2_node' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick
   


def plotErr(x, y):
    plt.figure(2)
    plt.plot(x, y, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick




if __name__=="__main__":
    # frequently changed parameters:
    penalty = 0.01
    
    # define initial conditions
    Ca_0 = 5  #Ca^{2+} [mol/area]?
    Ci_0 = 50   #CI         
    CiF_0 = 51  #CI^* #was previously 0, real data readouts start with some concentration.
    x0 = np.array([Ca_0, Ci_0, CiF_0])

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
    Ci = model.set_variable('_x', 'Ci') #
    CiF = model.set_variable('_x', 'CiF')

    # define ODEs and parameters, kr << kf
    kf = 0.0513514
    kr = 7.6 
    alpha = 1
    gamma = 1   # passive diffusion
    Ca_ext = 100 # constant extracellular calcium. Constant (assumes external is sink)
    s = model.set_variable('_u', 's')   # control variable ( input )
    CI_m = model.set_variable('_tvp', 'Ci_m') # timve varying parameter, or just hardcode

    model.set_rhs('Ca', alpha*s - gamma*(Ca - Ca_ext) + kr*CiF - kf*Ci*Ca)
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
  
    #mpc.settings.set_linear_solver(solver_name = "MA27") # recommended but bad.
    mpc.settings.supress_ipopt_output() # supresses output of solver

    mpc.set_param(**setup_mpc)
    n_horizon = 6
    t_step = 1/6
    n_robust = 1


    # define objective, which is to miminize the difference between Ci_m and Ci. 
    mterm = (model.x['CiF'] - model.tvp['Ci_m'])**2                   # terminal cost
    lterm = mterm                                                    # stage cost 
 
    mpc.set_objective(mterm = mterm, lterm = lterm)
    mpc.set_rterm(s = penalty) # sets a penalty on changes in s

    # make sure the objective/cost updates with CI_measured and time.    
    tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(tvp_fun)


    # define constraints
    mpc.bounds['lower', '_x', 'Ca'] = 0.0
    mpc.bounds['lower', '_x', 'Ci'] = 0.0
    mpc.bounds['lower', '_x', 'CiF'] = 0.0
    mpc.bounds['lower', '_u', 's'] = -5 # slow diffusion


    mpc.setup()
    
    # Estimator: assume all states can be directly measured
    estimator = do_mpc.estimator.StateFeedback(model)


    # Simulator
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
            'integration_tool': 'cvodes', # look into this
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 1/6, # (s) mean step is 6.11368547250401 in data
            }
    simulator.set_param(**params_simulator)
    # account for tvp
    tvp_template1 = simulator.get_tvp_template() # must differ from previous template name
    simulator.set_tvp_fun(tvp_fun_sim)


    simulator.setup()

    # finally begin closed loop simulation
    
       
    # set for controller, simulator, and estimator
    mpc.x0 = x0 #x0 in changable params. 
    simulator.x0 = x0
    estimator.x0 = x0
    mpc.set_initial_guess()

    # finally perform closed loop simulation
    n_steps = 700
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)


#    print(mpc.data['_u'])
#    print(mpc.data['_x'])
#    print(mpc.data['_time'])

    # pull final solutions for ease of use
    Ca_f = mpc.data['_x'][:, 0]
    Ci_f = mpc.data['_x'][:, 1]
    CiF_f = mpc.data['_x'][:, 2]
    t_f = mpc.data['_time']
    s = mpc.data['_u']
   

    print(np.shape(mpc.data['_x']))
    sol = np.transpose(mpc.data['_x'])

    #print(np.shape(Ca_f), np.shape(Ci_f), np.shape(CiF_f) )

    plotThreeLines(t_f, Ca_f, Ci_f, CiF_f)
    plotFourLines(t_f, Ca_f, Ci_f, CiF_f, s)


    # check error between Ci_M and Ci_sim
    CI_Meas_interp = np.interp(t_f, timeVec, CI_Meas)
    CI_Meas_interp = CI_Meas_interp[:, 0] # CiF_f different shape
#    print(np.shape(t_f))
#    print(np.shape(CI_Meas_interp))
#    print(np.shape(CiF_f))
    print(np.linalg.norm(CI_Meas_interp - CiF_f))

#    print(np.shape((CI_Meas_interp - CiF_f)))


    plt.figure(3)    
    plt.plot(t_f, (CI_Meas_interp - CiF_f))
    plt.title(r'Error as a function of time')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$CI^{*}_{Meas} - CI^*_{Sim}$', fontsize = 14)
    filename = 'CRE_fig3_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    plt.figure(4)
    plt.plot(t_f, CI_Meas_interp, label=r'$CI^{*}_{Meas}$')
    plt.plot(t_f, CiF_f, label=r'$CI^{*}_{Sim}$')
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig4_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    plt.figure(5)
    plt.plot(t_f, s, label=r'$s (Hz)$')
    plt.title(r'Signal s (maybe Hz)')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig5_' + str(row) + '.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    np.save('data/s_node_' + str(row), s)
    np.save('data/t_node_' + str(row), t_f)
    np.save('data/sol_node_' + str(row), sol)

    

    plt.show()
