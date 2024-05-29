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
import pandas as pd
from spikeCounter import spikeCounter
import seaborn as sns
from datetime import date
import spikefinder_eval as se
from spikefinder_eval import _downsample

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
    #plt.savefig(filename)
    #os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/CRE_across_nodes/') # only run with this line uncommented if you are Nick
  

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
    #plt.savefig(filename)
    #os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick
   


def plotErr(x, y):
    plt.figure(2)
    plt.plot(x, y, label = r'$Error$')
    plt.title('Dynamics of Error as a function of time', fontsize = 18)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'Error', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig2_' + str(i) + '.png'
    #plt.savefig(filename)
    #os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick




if __name__=="__main__":
    # frequently changed parameters:
    penalty = 0.01
    row = 4 # 6 has nans for testing, row 2 subsetAmount = 1000 is a great data subset tho. 
    #row = int(sys.argv[1])   
    
    dset = 5
    file_path = 'data/' + str(dset) + '.test.calcium.csv'
    data1 = pd.read_csv(file_path).T 

    data1 = np.array(data1)
    # calcium data is so large, start with a subset.
    subsetAmount = np.max(np.shape(data1[row,:])) # the way its set up, must be divisble by factor or stuff breaks. 
    #subsetAmount = 1000
    m, n = data1.shape
    CI_Meas = data1[row, :subsetAmount]
    

    print(np.shape(CI_Meas))


    # looks at a single neuron.  
    #CI_Meas = 50*CI_Meas
    #print(CI_Meas)

    # set up timevec, recordings were made at 59.1 hz
    tEnd = n*(1/59.1) 
    print("Simulating until final time", tEnd, "seconds, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False)
    

    # define initial conditions
    Ca_0 = 5  #Ca^{2+} [mol/area]?
#    Ci_0 = 7   #CI         
    CiF_0 = CI_Meas[0]  #CI^* #was previously 0, real data readouts start with some concentration.
    x0 = np.array([Ca_0, CiF_0])


    # follow MPC example ``batch bioreactor`` on do-mpc website
    model_type = 'continuous' # or discrete
    model = do_mpc.model.Model(model_type)
    
    # states struct, optimization variables
    # S is an unknown parameter. _x denotes var, _p param?
    Ca = model.set_variable('_x', 'Ca')
#    Ci = model.set_variable('_x', 'Ci') #
    CiF = model.set_variable('_x', 'CiF')

    # define ODEs and parameters, kr << kf
    kf = 0.0513514
    kr = 7.6 
    alpha = 20 
    gamma = 0.1   # passive diffusion
    L = CiF_0 + 7      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
    s = model.set_variable('_u', 's')         # control variable ( input )
    CI_m = model.set_variable('_tvp', 'Ci_m') # timve varying parameter, or just hardcode

    model.set_rhs('Ca', alpha*s - gamma*Ca+ kr*CiF - kf*Ca*(L - CiF))
#   model.set_rhs('Ci', kr*CiF - kf*Ci*Ca)
    model.set_rhs('CiF', kf*Ca*(L - CiF) - kr*CiF)
   
    model.setup()
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters, can change collocation/state discretization here.
    # does not impact my actual horizon or stepsize for simulation? 
    setup_mpc = {
            'n_horizon': 6, # pretty short horizion
            't_step': 1/59.1, # (s)
            'n_robust': 1,
            'store_full_solution': True,
            }
  
    #mpc.settings.set_linear_solver(solver_name = "MA27") # recommended but bad.
    mpc.settings.supress_ipopt_output() # supresses output of solver

    mpc.set_param(**setup_mpc)
    n_horizon = 6
    t_step = 1/59.1
    n_robust = 1

#    print(model.u.keys())

    # define objective, which is to miminize the difference between Ci_m and Ci.
    baseLine = 1 # 2.5 was nice for row 2
    mterm = ((model.x['CiF']-baseLine)/baseLine - model.tvp['Ci_m'])**2
    #mterm = (model.x['CiF'] - model.tvp['Ci_m'])**2                    # terminal cost
    
    #
    #lterm = .001*model.u['s']**2 #+ (model.x['CiF'] - model.tvp['Ci_m'])**2 # stage cost 
    lterm = mterm


    mpc.set_objective(mterm = mterm, lterm = lterm)
    mpc.set_rterm(s = penalty) # sets a penalty on changes in s, defined at top of main for ease
    # see latex doc for description, but essentialy adds penalty*(s_k - s_{k-1}) term.

    # make sure the objective/cost updates with CI_measured and time.    
    tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(tvp_fun)


    # define constraints
    #mpc.bounds['lower', '_x', 'Ca'] = 0.0
    #mpc.bounds['lower', '_x', 'Ci'] = 0.0
    #mpc.bounds['lower', '_x', 'CiF'] = 0.0
    mpc.bounds['lower', '_u', 's'] = 0 # slow diffusion
#    mpc.bounds['upper', '_u', 's'] = 100
   
    # once mpc.setup() is called, no model parameters can be changed.
    mpc.setup()
    
    # Estimator: assume all states can be directly measured
    estimator = do_mpc.estimator.StateFeedback(model)


    # Simulator
    simulator = do_mpc.simulator.Simulator(model)
    params_simulator = {
            'integration_tool': 'cvodes', # look into this
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': 1/59.1, # (s) mean step is 6.11368547250401 in data
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
    n_steps = int(tEnd/t_step) #ensures we hit final time
    for k in range(n_steps):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    # pull final solutions for ease of use
    Ca_f = mpc.data['_x'][:, 0]
#    Ci_f = mpc.data['_x'][:, 1]
    CiF_f = mpc.data['_x'][:, 1]
    t_f = mpc.data['_time']
    s = mpc.data['_u']
   
    Ci_f = L - CiF_f
    CiF_f = (CiF_f-baseLine)/baseLine # normalize, this was used in cost function

    print("Data shape:", np.shape(mpc.data['_x']))
    sol = np.transpose(mpc.data['_x'])

    #print(np.shape(Ca_f), np.shape(Ci_f), np.shape(CiF_f) )

    plotThreeLines(t_f, Ca_f, Ci_f, CiF_f)
    plotFourLines(t_f, Ca_f, Ci_f, CiF_f, s)


    # check error between Ci_M and Ci_sim
    CI_Meas_interp = np.interp(t_f, timeVec, CI_Meas)
    CI_Meas_interp = CI_Meas_interp[:, 0] # CiF_f different shape
    print("Relative MSE of tracking:", np.linalg.norm(CI_Meas_interp - CiF_f)/len(CiF_f))



    plt.figure(3)    
    plt.plot(t_f, (CI_Meas_interp - CiF_f))
    plt.title(r'Error as a function of time')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$CI^{*}_{Meas} - CI^*_{Sim}$', fontsize = 14)
    filename = 'CRE_fig3_' + str(row) + '.png'
#    plt.savefig(filename)
#    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    plt.figure(4)
    plt.plot(t_f, CI_Meas_interp, label=r'$CI^{*}_{Meas}$')
    plt.plot(t_f, CiF_f, label=r'$CI^{*}_{Sim}$') ## subtracting baseline
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig4_' + str(row) + '.png'
#    plt.savefig(filename)
#    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    plt.figure(5)
    plt.plot(t_f, s, label=r'$s (Hz)$')
    plt.title(r'Signal s (maybe Hz)')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.legend()
    filename = 'CRE_fig5_' + str(row) + '.png'
#    plt.savefig(filename)
#    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    # load in actual truth data
    file_path2 = 'data/5.test.spikes.csv'
    spikeDat = pd.read_csv(file_path2).T #had to figure out why data class is flyDat from print(mat). No clue. 
    spikeDat = np.array(spikeDat)
    spikeDat = spikeDat[:, :subsetAmount]
    mSpike,nSpike = spikeDat.shape
    spikeDatRaw = spikeDat[row, :]
    
    ##----------------------------------------
    # POST PROCESS FOR DOWNSAMPLING BELOW
    ##----------------------------------------
    # save things for later eval too
    np.save('data/s_node_' + str(row) + 'dset_' + str(dset), s)
    #np.save('data/t_node_' + str(row), t_f)
    #np.save('data/sol_node_' + str(row), sol)
    

    # remove NaNs AT START
    s = np.array(s[:,0]) # gotta reshape s 
    naninds = np.isnan(spikeDatRaw) | np.isnan(s)
    #print("nanID shape:", np.shape(naninds))
    #print("spikeDatRaw shape:", np.shape(spikeDatRaw))
    #print("shape of s", np.shape(s))
    #print(spikeDatRaw)
    spikeDatRaw = spikeDatRaw[~naninds]
    s = s[~naninds]
    #print(spikeDatRaw)
    
    #print("SpikeDat post nan removal", np.shape(spikeDatRaw))
    #print("s shape post nan removal", np.shape(s))
    
    factor= 4#32 #how much to downsample by
    spikeDat = _downsample(spikeDatRaw, factor)
    s = _downsample(s, factor)

    m1 = min([len(s), len(spikeDat)])
    s = s[0:m1]
    spikeDat = spikeDat[0:m1]
    newTime = np.linspace(0, t_f[-1], m1)#, endpoint = True) # only used for plotting

    # TIME TESTING, make sure things line up. newTime a little off, should be OK just for plotting.
    #print("final timeVec val:", timeVec[-1], np.shape(timeVec))
    #print("final t_f val:", t_f[-1], np.shape(t_f))
    #print("timeVec:", timeVec[0:4], timeVec[-4:-1])
    #print("t_f:", t_f[0:4], t_f[-4:-1])
    #print("newTime", newTime[0:4], newTime[-4:-1])


       
    # finally scale so viewing is more clear
    s = (np.max(spikeDat)/np.max(s))*s # correlation coeff. invariant wrt scaling. 

    # compute correlation coefficient -----------------------------------------------
    #interpS = np.interp(timeVec[::factor], t_f[::factor,0], s)
    #corrCoef = np.corrcoef(interpS, spikeDat)[0, 1]
    #print("interp coeff:", corrCoef) -----------------------------------------------
    corrCoef = np.corrcoef(s, spikeDat)[0, 1] # toss first 200 time instants, contains bad transients.
    print("no interp:", corrCoef)     # ---------------------------------------------

   
    plt.figure(6)
    #plt.plot(t_f[::factor, 0], s, label=r'Simulated Rate')         these explode as soon as len(timeVec) isn't evenly divisible by factor
    #plt.plot(timeVec[::factor], spikeDat, label="Recorded Spike")
    plt.plot(newTime, s, label=r'Simulated Rate')
    plt.plot(newTime, spikeDat, label="Recorded Spike Rate")
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.title("Expected and Recorded spikes")#, bin size of " + str(1000*binSizeTime) + " ms")
    plt.legend()
    
    plt.show()
    sys.exit()


    #np.save('data/s_node_' + str(row), s)
    #np.save('data/t_node_' + str(row), t_f)
    #np.save('data/sol_node_' + str(row), sol)
    



    # trying to use se.score out of the box doesn't work well. This is attempted below. #
#    print(np.shape(s))                                                                              
#    print(np.shape(spikeDat))
#    s = np.array(s[:,0])
#    c = np.array(se.score(s, spikeDatRaw, method='corr', downsample=factor))
#    print(c)

    plt.show()
