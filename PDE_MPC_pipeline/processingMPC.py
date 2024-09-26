import numpy as np
import sys
import scipy.io
#import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

state = 'Stripe'
#state = 'Dark'

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



fig1a = plt.figure()
pFr = np.zeros(np.shape(firingRates))
# Define the index mapping
index_mapping = [0, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 9, 17]
# Copy rows from the input matrix to the new matrix based on the index mapping
for i in range(np.shape(firingRates)[0]):
             pFr[i] = firingRates[index_mapping[i]]  
# create permuted firing rate, pFr
pFr = firingRates[index_mapping]
# finally, plot (later)
#figMat = plt.imshow(pFr, extent = [0, t[-1], -np.pi, np.pi])
#plt.title('Reordered heatmap of firing rate, MPC, ' + state, fontsize = 20)
#plt.xlabel(r'Time (s)', fontsize = 14)
#plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
#plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
#plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])

# find location  of maximal firing rate (theta value) DOES WAY WORSE THAN CIRCMEAN, TOO SENSITIVE
thetaSpace = np.linspace(-np.pi, np.pi, m)
thetas = np.zeros(len(firingRates[1,:]))
for i in range(len(pFr[1,:])):
    indx = np.argmax(pFr[:, i])
    thetas[i] = thetaSpace[indx] 

# attempt to find location of maximal firing rate in MPC with circmean
#x = np.linspace(-np.pi, np.pi, N, endpoint=False)
print(len(thetaSpace))
mLoc = np.zeros(len(t))
for i in range(len(t)):
#    m = pg.circ_mean(thetaSpace, firingRates[:,i])
    m = pg.circ_mean(thetaSpace, pFr[:,i])
    vecLength = pg.circ_r(thetaSpace, pFr[:,i])
    #print(vecLength)
    #if vecLength > 0.95:
    #    m1 = mLoc1[i-1]
    #print(t[i], vecLength, m1)
    mLoc[i] = m

print(len(mLoc))

# load in firing rate data from SR solve (moved from singleRing/data)
firingRatesSR = np.load('data/PDEfiringRates' + state + '.npy')
firingTimesSR = np.load('data/PDEfiringTimes' + state + '.npy')

# the below logic computes maximal theta location, can do better with circmean. 
m1, n1 = np.shape(firingRatesSR)
thetaSpace = np.linspace(-np.pi, np.pi, m1)
thetas1 = np.zeros(len(firingRatesSR[1,:]))
for i in range(len(firingRatesSR[1,:])):
    indx = np.argmax(firingRatesSR[:, i])
    thetas1[i] = thetaSpace[indx] 


# attempt to find location of maximal firing rate with circmean
#x = np.linspace(-np.pi, np.pi, N, endpoint=False)
print(len(thetaSpace))
mLoc1 = np.zeros(len(firingTimesSR))
for i in range(len(firingTimesSR)):
    m1 = pg.circ_mean(thetaSpace, firingRatesSR[:,i])
    #    print(m1)
    mLoc1[i] = m1

#print("m1:", len(mLoc1))
#print(len(firingTimesSR))

#fig2 = plt.figure()
#plt.plot(t, mLoc, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)
#plt.plot(firingTimesSR, mLoc1, label="Maximal location from PDE")
#plt.legend()
#plt.title("Location of Maximal Firing")
#plt.xlabel("t")
#plt.ylabel(r"$\theta$")



fig3 = plt.figure() # MPC heatmap scaled to match that of PDE
PDEmax = np.max(firingRatesSR)
MPCmax = np.max(pFr)
pFr = (PDEmax/MPCmax)*pFr
figMat = plt.imshow(pFr, extent = [0, t[-1], -np.pi, np.pi])
plt.title('Heatmap of firing rate, MPC, ' + state, fontsize = 20)
plt.xlabel(r'Time (s)', fontsize = 14)
plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])


fig4 = plt.figure()
figMat = plt.imshow(firingRatesSR, extent = [0, firingTimesSR[-1], -np.pi, np.pi])
plt.title('Heatmap of Firing Rate, PDE, ' + state, fontsize = 20)
plt.xlabel(r'Time (s)', fontsize = 14)
plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])
    

# attempt polar plot of angles, like fig 2
fig5, ax = plt.subplots(subplot_kw={'projection': 'polar'})
print(np.shape(mLoc))
print(np.shape(t))
ax.plot(mLoc, t)
ax.plot(mLoc1, firingTimesSR)
ax.grid(True)


fig6 = plt.figure()
unwrapFr = np.unwrap(mLoc)
unwrapFrSr = np.unwrap(mLoc1)
unwrapFrRough = np.unwrap(thetas)
plt.plot(t, unwrapFr, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)i
plt.plot(firingTimesSR, unwrapFrSr, label="Maximal location from PDE")
#plt.plot(t, unwrapFrRough, label="max loc without circmean") # do not need, circmean does good
plt.legend()
plt.title("Location of Maximal Firing, Unwrapped, " + state, fontsize = 20)
plt.xlabel("t")
plt.ylabel(r"$\theta$")


#fig7 = plt.figure()
#plt.plot(t, mLoc, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)i
#plt.plot(firingTimesSR, mLoc1, label="Maximal location from PDE")
#plt.plot(t, thetas, label="max loc without circmean")
#plt.legend()
#plt.title("Location of Maximal Firing, Unwrapped, " + state, fontsize = 20)
#plt.xlabel("t")
#plt.ylabel(r"$\theta$")


# pull max loc from pos, raw CI dat
print(np.shape(data))
m, n = np.shape(data)
mLoc2 = np.zeros(n)
index_mapping = [0, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 9, 17]
# Copy rows from the input matrix to the new matrix based on the index mapping
#print(data[1:2, 10:20])
data = data[index_mapping]  
#print(data[1:2, 10:20])
for i in range(n):
#    m = pg.circ_mean(thetaSpace, firingRates[:,i])
    m2 = pg.circ_mean(thetaSpace, data[:, i])
    mLoc2[i] = m2
mLoc2 = np.unwrap(mLoc2)


# convert velocity data to positioning data
#file_path = "data/velStripe" #velStripe: cond1_allfly2_stripe1_posDatMatch.vrot
file_path = "data/vel" + state #velDark: cond1_allfly2_dark1_posDatMatch.vrot
mat = scipy.io.loadmat(file_path)
#temp = mat['velStripe']
temp = mat['vel' + state] # print mat to see what options, should be var name in matlab
(m, n) = temp.shape   # flipped from other data
realVel = np.zeros(m)
realVel = temp[:, 0]
print(m,n)


# load in times for samples
#file_path2 = "data/timeStripe"
file_path2 = 'data/time' + state
mat = scipy.io.loadmat(file_path2)
#temp = mat['timeStripe']
temp = mat['time' + state]
(m1, n1) = temp.shape
timeVec = np.zeros(m1)
timeVec = temp[:-1, 0] # chop off last time for same number of vel
tEnd = timeVec[-1]     # this also reshapes
#



posDat = np.zeros(len(realVel))
posDat[0] = mLoc2[0]
print("posDat", len(posDat))
print("time len", len(timeVec))
for i in range(1, len(posDat)):
#    print(i, t[i])
#    print(realVel[i])
    posDat[i] = posDat[i-1] + (timeVec[i] - timeVec[i-1])*realVel[i]*(-1)

posDat = np.unwrap(posDat)


fig8 = plt.figure()
plt.plot(t, unwrapFr, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)i
plt.plot(firingTimesSR, unwrapFrSr, label="Maximal location from PDE")
plt.plot(timeVec, mLoc2[:-1], label = "Maximal Location from Raw CI Data")
plt.plot(timeVec, posDat, label = "Maximal location from positioning data")
#plt.plot(t, thetas, label="max loc without circmean")
plt.legend()
plt.title("Location of Maximal Firing, Unwrapped, " + state, fontsize = 20)
plt.xlabel("t")
plt.ylabel(r"$\theta$")




plt.show()




