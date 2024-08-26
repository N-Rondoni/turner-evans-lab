import numpy as np
import sys
import scipy.io
#import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

state = 'Stripe'
#state = 'Dark'

## Handle MPC solution

file_path = 'data/ciDat' + state + '.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['ciDat' + state] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape
n = np.shape(np.load('data/MPC_' + state + '_s_node_1.npy'))[0]
t = np.load('data/MPC_' + state + '_t_node_1.npy')
t = np.ndarray.flatten(t)

firingRates = np.zeros((m, n)) 
for i in range(m):
    firingRates[i, :] = np.reshape(np.load('data/MPC_' + state + '_s_node_' + str(i) + '.npy'), (700, ))

pFr = np.zeros(np.shape(firingRates))
# Define the index mapping
index_mapping = [0, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 9, 17]
# Copy rows from the input matrix to the new matrix based on the index mapping
for i in range(np.shape(firingRates)[0]):
             pFr[i] = firingRates[index_mapping[i]]  # create permuted firing rate, pFr


thetaSpace = np.linspace(-np.pi, np.pi, m)
mlocMPC = np.zeros(len(t))
for i in range(len(t)):
#    m = pg.circ_mean(thetaSpace, firingRates[:,i])
    m = pg.circ_mean(thetaSpace, pFr[:,i])
    vecLength = pg.circ_r(thetaSpace, pFr[:,i])
    mlocMPC[i] = m
mlocMPC = np.unwrap(mlocMPC)


## Compute maximal firing location with circmean for all three PDE solutions
file_path3 = 'data/PDEfiringRates_Noised' + state + '.npy'    
fr_Noised = np.load(file_path3)
t_Noised = np.load('data/PDEfiringTimes_Noised' + state + '.npy')
(m_Noised, n_Noised) = fr_Noised.shape

file_path4 = 'data/PDEfiringRates_An' + state + '.npy'    
fr_An = np.load(file_path4) 
t_An = np.load('data/PDEfiringTimes_An' + state + '.npy')
(m_An, n_An) = fr_An.shape

file_path5 = 'data/PDEfiringRates_cont' + state + '.npy'
fr_Cont = np.load(file_path5)
t_Cont = np.load('data/PDEfiringTimes_cont' + state + '.npy') 
(m_Cont, n_Cont) = fr_Cont.shape
print(np.shape(t_Cont), n_Cont)


mlocAn = np.zeros(n_An)
for i in range(n_An):
    m = pg.circ_mean(thetaSpace, fr_An[:,i])
    mlocAn[i] = m

mlocAn = np.unwrap(mlocAn)


mlocNoised = np.zeros(n_Noised)
for i in range(n_Noised):
    m = pg.circ_mean(thetaSpace, fr_Noised[:,i])
    mlocNoised[i] = m

mlocNoised = np.unwrap(mlocNoised)


mlocCont = np.zeros(n_Cont)
for i in range(n_Cont):
    m = pg.circ_mean(thetaSpace, fr_Cont[:,i])
    mlocCont[i] = m

mlocCont = np.unwrap(mlocCont)





fig1 = plt.figure()
plt.plot(t, mlocMPC, label="MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)
plt.plot(t_An, mlocAn, label="PDE, no noise")
plt.plot(t_Noised, mlocNoised, label="PDE, with noise")
plt.plot(t_Cont, mlocCont, label="PDE, with control")
plt.legend()
plt.title("Location of Maximal Firing, " + state, fontsize = 20)
plt.xlabel("t", fontsize = 16)
plt.ylabel(r"$\theta$", fontsize = 16)



plt.show()
