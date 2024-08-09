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

def plotTwoLines(x, y1, y2):
    plt.figure(1)
    plt.plot(x, y1, '--', linewidth = 1, label = r'$Ca^{2+}$ (mol)')
    #plt.plot(x, y2, '--', linewidth = 1, label = r'$CI$ (mol)')
    plt.plot(x, y2, linewidth = 2, label = r'$CI^{*}$ (mol)')
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


def movingAverage(signal, winLen):
    wLen = winLen # window length
    mAvg = np.zeros(np.shape(signal))
    for i in range(len(mAvg) - wLen):
        avg = np.sum(signal[i: i+ wLen])/wLen
        mAvg[i] = avg
    mAvg[-wLen:] = signal[-wLen:]
    return mAvg


def sigmoid(signal):
    #sig = np.zeros(np.shape(signal)):
    sig = 1/(1 + np.exp(-1*(signal - 1)))
    return sig

def indicator(signal):
    #out = np.zeros(np.shape(signal))
    print(float(signal))
    if float(signal) > 0.05:
        out = 1
    else: 
        out = 0 
    print(signal, out)
    return out

if __name__=="__main__":
    # frequently changed parameters:
    start = time.time()
    penalty = 0.01
    row = int(sys.argv[1])   
    dset = int(sys.argv[2])
    stat = str(sys.argv[3])

    file_path = 'data/resampled/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.calcium.npy'
    file_path2R = 'data/resampled/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'

    tempCalc = np.load(file_path) 

    # uncommend hardcoded subsetAmount to test smaller subsets
    subsetAmount = np.shape(tempCalc)[0]
    #print("subamount:", subsetAmount)
    #subsetAmount = 10000
    CI_Meas = tempCalc[:subsetAmount]
    n = len(CI_Meas)
    
  
    #CI_Meas = movingAverage(CI_Meas, 30)
    CI_Meas = sigmoid(CI_Meas) - 0.15

    # set up timevec, recordings were made at 59.1 hz

    if dset == 1:
        imRate = 1/100 # (resampled)
    if dset == 2:
        imRate = 1/11.8
    if dset in [3, 5]:
        imRate = 1/59.1
    if dset == 4:
        imRate = 1/7.8

    #imRate = 1/50
    tEnd = n*(imRate) 
    print("Simulating until final time", tEnd/60, "minutes, consisting of", n, "data points")
    timeVec = np.linspace(0, tEnd, n, endpoint = False) #used in interp call
    

    # define initial conditions
    Ca_0 = 5  #Ca^{2+} [mol/area]?
#    Ci_0 = 7   #CI         
    CiF_0 = CI_Meas[0]  #CI^* #was previously 0, real data readouts start with some concentration.
    x0 = np.array([Ca_0, CiF_0])


    # follow MPC example ``batch bioreactor`` on do-mpc website
    model_type = 'continuous' # or discrete
    model = do_mpc.model.Model(model_type)
    
    # states struct, optimization variables
    Ca = model.set_variable('_x', 'Ca')
#    Ci = model.set_variable('_x', 'Ci') #
    CiF = model.set_variable('_x', 'CiF')

    # define ODEs and parameters, kr << kf
   
    # same params as run_6_13_morning, i'm pretty sure
    if dset in [1, 2, 3]: # params found from BFS, worked well on 5 for sure.
        #print("you got dset 2 or 4 tho dummy")
        kf = 0.1
        kr = 10 
        alpha = 16.666666 
        gamma = 1.3666666   # passive diffusion
        L = CiF_0 + 100      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
        baseLine = 1 # 2.5 was nice for row 2
    
#    if dset == 2:
#        kf = 0.004
#        kr = 1 
#        alpha = 1
#        gamma = .1   # passive diffusion
#        L = CiF_0 + 100      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
#        baseLine = 1 # 2.5 was nice for row 

#    if dset == 5:
#        kf = 0.0069444444
#        kr = 1 
#        alpha = 1
#        gamma = 1   # passive diffusion
#        L = CiF_0 + 100      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
#        baseLine = 1 # 2.5 was nice for row 


    if dset == 5: # params found from BFS, worked well on 5 for sure.
        kf = 0.2
        kr = 10 
        alpha = 10 
        gamma = 0.73333   # passive diffusion
        L = CiF_0 + 100      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
        baseLine = 0.5 # 2.5 was nice for row 2
    
    if dset == 4:
        kf = 0.1
        kr = 10 
        alpha = 10 
        gamma = 0.73333   # passive diffusion
        L = CiF_0 + 100      # total amount of calcium indicator, assumes 10 units of unflor. calcium indicator.
        baseLine = 1.0 # 2.5 was nice for row 2
    
    tstep = 1/50 #1/100 seems to just do great
    if dset == 1:
        tstep = 1/100
    #.292995469758242 node: 1 dset: 5 alpha: 10.0 gamma: 0.7333333333333333 kf: 0.2 kr: 10.0 bl 0.5


    s = model.set_variable('_u', 's')         # control variable ( input )
    CI_m = model.set_variable('_tvp', 'Ci_m') # timve varying parameter, or just hardcode
    Ca_ext = 40

    model.set_rhs('Ca', alpha*s - gamma*Ca + kr*CiF - kf*Ca*(L - CiF))
#   model.set_rhs('Ci', kr*CiF - kf*Ci*Ca)
    model.set_rhs('CiF', (kf*Ca*(L - CiF) - kr*CiF))
   
    model.setup()
    mpc = do_mpc.controller.MPC(model)

    # Optimizer parameters, can change collocation/state discretization here.
    setup_mpc = {
            'n_horizon': 6, # pretty short horizion
            't_step': tstep, # (s)
            'n_robust': 0,
            'store_full_solution': True,
            }
  
    #mpc.settings.set_linear_solver(solver_name = "MA27") # recommended but bad.
    mpc.settings.supress_ipopt_output() # supresses output of solver

    mpc.set_param(**setup_mpc)
    n_horizon = 6
    t_step = tstep
    n_robust = 1

#    print(model.u.keys())

    # define objective, which is to miminize the difference between Ci_m and Ci.
    #mterm = ((model.x['CiF']-baseLine)/baseLine - model.tvp['Ci_m'])**2
    mterm =(sigmoid(model.x['CiF']) - model.tvp['Ci_m'])**2                    # terminal cost

    
    #mterm = (sigmoid((model.x['CiF']-baseLine)/baseLine) - model.tvp['Ci_m'])**2
    #lterm = 0.001*model.u['s']
 
    #lterm = .001*model.u['s']**2 #+ (model.x['CiF'] - model.tvp['Ci_m'])**2 # stage cost 
    lterm = mterm


    mpc.set_objective(mterm = mterm, lterm = lterm)
    mpc.set_rterm(s = penalty) # sets a penalty on changes in s, defined at top of main for ease
    # see latex doc for description, but essentialy adds penalty*(s_k - s_{k-1}) term.

    # make sure the objective/cost updates with CI_measured and time.    
    tvp_template = mpc.get_tvp_template()
    mpc.set_tvp_fun(tvp_fun)


    mpc.bounds['lower', '_u', 's'] = 0 # slow diffusion
   
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
            't_step': tstep, # (s) mean step is 6.11368547250401 in data
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
    n_steps = int(tEnd/tstep) #ensures we hit final time
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
    
    end = time.time()
    print("Solve completed in",  (end-start)/60, "minutes")
    

    #s = movingAverage(s, 30)

    Ci_f = L - CiF_f
    #CiF_f = (CiF_f-baseLine)/baseLine # normalize, this was used in cost function
    CiF_f = sigmoid(CiF_f)
    #CiF_f = sigmoid((CiF_f-baseLine)/baseLine) # normalize, this was used in cost function

    sol = np.transpose(mpc.data['_x'])

    #print(np.shape(Ca_f), np.shape(Ci_f), np.shape(CiF_f) )

    #plotThreeLines(t_f, Ca_f, Ci_f, CiF_f)
    plotTwoLines(t_f, Ca_f, CiF_f)
    plotFourLines(t_f, Ca_f, Ci_f, CiF_f, s)


    # check error between Ci_M and Ci_sim
    CI_Meas_interp = np.interp(t_f, timeVec, CI_Meas)
    CI_Meas_interp = CI_Meas_interp[:, 0] # CiF_f different shape
    
    s = np.array(s[:,0]) # gotta reshape s 
#    print("you rang:", np.shape(s), np.shape(t_f), np.shape(timeVec))
    s_interp = np.interp(timeVec, t_f[:,0], s)
#    print(np.shape(s_interp))
    print("Relative MSE of tracking:", np.linalg.norm(CI_Meas_interp - CiF_f)/len(CiF_f))



    plt.figure(3)    
    plt.plot(t_f, (CI_Meas_interp - CiF_f))
    plt.title(r'Error as a function of time')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$CI^{*}_{Meas} - CI^*_{Sim}$', fontsize = 14)
    filename = 'CRE_fig3_' + str(row) + '.png'
#    plt.savefig(filename)
#    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/MPC_CRE_across_nodes/') # only run with this line uncommented if you are Nick


    # load in actual truth data
    file_path2 = 'data/resampled/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'
#   spikeDat = pd.read_csv(file_path2).T #had to figure out why data class is flyDat from print(mat). No clue. 
#   spikeDat = np.array(spikeDat)
#   spikeDat = spikeDat[:, :subsetAmount]
#   mSpike,nSpike = spikeDat.shape
#    spikeDatRaw = spikeDat[row, :]
    spikeDatRaw = np.load(file_path2)
    spikeDatRaw = spikeDatRaw[:subsetAmount] 
    #spikeDatRaw = np.interp(t_f, timeVec, spikeDatRaw) I think interpolating s to fit these time points makes more sense
#    print(spikeDatRaw[0:25])

    ##----------------------------------------
    # POST PROCESS FOR DOWNSAMPLING BELOW
    ##----------------------------------------
    # save things for later eval too
    saveLoc = 'data/resampled/solutions/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.sVals'
#    print("shapes:", np.shape(spikeDatRaw), np.shape(s_interp))  

    s = s_interp
    np.save(saveLoc, s)

    factor = int(np.ceil((1/10)*(1/imRate)))  #5 #4#32 #how much to downsample by
    spikeDat = _downsample(spikeDatRaw, factor)
    s = _downsample(s, factor)

    m1 = min([len(s), len(spikeDat)])
    s = s[0:m1]
    ispikeDat = spikeDat[0:m1]
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
    #corrCoef = np.corrcoef(s[300:], spikeDat[300:])[0, 1] # toss first 200 time instants, contains bad transients.
    corrCoef = np.corrcoef(s[100:], spikeDat[100:])[0, 1] # lessed for dset5, neuron 5

    print("Corr Coef, no interp:", corrCoef)     # ---------------------------------------------

    neuron = row
    plt.figure(4)
    plt.plot(t_f, CiF_f, label=r'$CI^{*}_{Sim}$') ## subtracting baseline
    plt.plot(t_f, CI_Meas_interp, label=r'$CI^{*}_{Meas}$', alpha = 0.7)
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'Tracking_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)
    os.system('cp ' + filename + '.png /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
    os.system('rm ' + filename + '.png')

    
    subL, subH = 1000, 2000

    plt.figure(5)
    plt.plot(t_f[subL:subH], CiF_f[subL:subH], label=r'$CI^{*}_{Sim}$') ## subtracting baseline
    plt.plot(t_f[subL:subH], CI_Meas_interp[subL:subH], label=r'$CI^{*}_{Meas}$', alpha = 0.7)
    plt.title(r'$CI^{*}$, simulated and measured')
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'CI', fontsize = 14)
    plt.legend()
    filename = 'Tracking_subset_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)
    os.system('cp ' + filename + '.png /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
    os.system('rm ' + filename + '.png')

    plt.figure(6)
    subL, subH = int(subL/factor), int(subH/factor)
    plt.plot(newTime[subL:subH], s[subL:subH], label=r'Simulated Rate')
    plt.plot(newTime[subL:subH], spikeDat[subL:subH], label="Recorded Spike Rate", alpha = 0.7)
    plt.xlabel(r'$t$', fontsize = 14)
    plt.ylabel(r'$s$', fontsize = 14)
    plt.legend()
    plt.title("Expected and Recorded spikes")#, bin size of " + str(1000*binSizeTime) + " ms")
    filename = 'Spikes_dset'+ str(dset) + "_neuron" + str(neuron)
    plt.savefig(filename)
    os.system('cp ' + filename + '.png /mnt/c/Users/nicho/Pictures/Gt_sim/dset' + str(dset) +'/neuron' + str(neuron)) # only run with this line uncommented if you are Nick
    os.system('rm ' + filename + '.png')

    #plt.show()

