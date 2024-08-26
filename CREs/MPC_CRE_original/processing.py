import numpy as np
import sys
import scipy.io
#import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg

file_path = 'data/flymat.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape
n = np.shape(np.load('data/s_node_1.npy'))[0]
#print(m, n)
t = np.load('data/t_node_1.npy')
t = np.ndarray.flatten(t)

#print(np.shape(t))

#print(t)
#print(t[1][-1])
#print(t[-1][0])


firingRates = np.zeros((m, n)) #sub 1 because of the way sVec created
for i in range(m):
    firingRates[i, :] = np.reshape(np.load('data/s_node_' + str(i) + '.npy'), (700, ))



#print(np.min(firingRates))

#firingRates = np.transpose(firingRates)
#print(np.shape(firingRates))
#showcase what is hopefuly the firing rates
#print(firingRates)


# reorder for it to make sense, which the below does not do
#fig1 = plt.figure()
#figMat = plt.imshow(firingRates, extent = [0, t[-1], -np.pi, np.pi])
#plt.title('Heatmap of Firing Rate, MPC', fontsize = 20)
#plt.xlabel(r'Time (s)', fontsize = 14)
#plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
#plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
#plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])


fig1a = plt.figure()
pFr = np.zeros(np.shape(firingRates))
# Define the index mapping
index_mapping = [0, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16, 9, 17]
# Copy rows from the input matrix to the new matrix based on the index mapping
for i in range(np.shape(firingRates)[0]):
             pFr[i] = firingRates[index_mapping[i]]  
# create permuted firing rate, pFr
pFr = firingRates[index_mapping]
figMat = plt.imshow(pFr, extent = [0, t[-1], -np.pi, np.pi])
plt.title('Reordered heatmap of firing rate, MPC', fontsize = 20)
plt.xlabel(r'Time (s)', fontsize = 14)
plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])

# find location  of maximal firing rate (theta value)
thetaSpace = np.linspace(-np.pi, np.pi, m)
thetas = np.zeros(len(firingRates[1,:]))
for i in range(len(pFr[1,:])):
    indx = np.argmax(pFr[:, i])
    thetas[i] = thetaSpace[indx] 

# attempt to find location of maximal firing rate with circmean
#x = np.linspace(-np.pi, np.pi, N, endpoint=False)
print(len(thetaSpace))
mLoc = np.zeros(len(t))
for i in range(len(t)):
    m = pg.circ_mean(thetaSpace, firingRates[:,i])
    mLoc[i] = m

print(len(mLoc))

# load in firing rate data from SR solve (moved from singleRing/data)
firingRatesSR = np.load("data/firingRates18Node.npy")
firingTimesSR = np.load("data/firingTimes18Node.npy")

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
    mLoc1[i] = m1

print("m1:", len(mLoc1))
print(len(firingTimesSR))

fig2 = plt.figure()
plt.plot(t, mLoc, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)
plt.plot(firingTimesSR, mLoc1, label="Maximal location from PDE")
plt.legend()
plt.title("Location of Maximal Firing")
plt.xlabel("t")
plt.ylabel(r"$\theta$")



fig3 = plt.figure() # MPC heatmap scaled to match that of PDE
PDEmax = np.max(firingRatesSR)
MPCmax = np.max(pFr)
pFr = (PDEmax/MPCmax)*pFr
figMat = plt.imshow(pFr, extent = [0, t[-1], -np.pi, np.pi])
plt.title('Reordered heatmap of firing rate, MPC', fontsize = 20)
plt.xlabel(r'Time (s)', fontsize = 14)
plt.ylabel(r'HD Cell $\theta$', fontsize = 14)
plt.colorbar(shrink=.20)#location="bottom")
#the below places ticks at linspace locations, then labels. Extent above determines width.
plt.yticks(np.linspace(-np.pi, np.pi, 5), [r'$-\pi$', r'$-\pi/2$',r'$0$', r'$\pi/2$', r'$\pi$'])


fig4 = plt.figure()
figMat = plt.imshow(firingRatesSR, extent = [0, firingTimesSR[-1], -np.pi, np.pi])
plt.title('Heatmap of Firing Rate, PDE', fontsize = 20)
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
plt.plot(t, unwrapFr, label="Maximal location from MPC") #was plotting (t, thetas) versus (firingTimesSR, thetas1)
plt.plot(firingTimesSR, unwrapFrSr, label="Maximal location from PDE")
plt.legend()
plt.title("Location of Maximal Firing, Unwrapped", fontsize = 20)
plt.xlabel("t")
plt.ylabel(r"$\theta$")






plt.show()
sys.exit()








#compare this to actual data
plt.figure(1)
plt.imshow(data, aspect='auto')
plt.title(r"Matrix Visualization of $CI^*_{meas}$", fontsize = 18)
plt.xlabel(r'Steps of $\Delta t$', fontsize =14)
plt.ylabel(r'Node', fontsize = 14)
plt.colorbar()

# fft business, plot amplitudes
plt.figure(2)
Ffr = np.fft.rfft(firingRates)
amplitude = np.abs(Ffr)
#print(amplitude[1, 0:100])
#print(amplitude.shape)
fig = plt.imshow(np.log(amplitude), aspect='auto')
plt.title(r"Log of Amplitude of $\hat{S}(\omega$)", fontsize=20)
plt.xlabel(r'Steps of $\Delta \omega$', fontsize = 16)
plt.ylabel('Node', fontsize = 16)
plt.colorbar()

# plot power spectrum
plt.figure(3)
powerSpec = np.abs(Ffr)**2
#print(amplitude[1, 0:100])
#print(amplitude.shape)
fig = plt.imshow(np.log(powerSpec), aspect='auto')
plt.title(r"Log of Power Spectrum of $\hat{S}(\omega$)", fontsize = 20)
plt.xlabel(r'Steps of $\Delta \omega$', fontsize = 16)
plt.ylabel('Node', fontsize = 16)
plt.colorbar()


# plot phase spectrum
plt.figure(4)
Ffr = np.fft.rfft(firingRates)
phase = np.angle(Ffr)
#print(amplitude[1, 0:100])
#print(amplitude.shape)
fig = plt.imshow(phase, aspect='auto')
plt.title(r"Phase Spectrum of $\hat{S}(\omega$)", fontsize = 20)
plt.xlabel(r'Steps of $\Delta \omega$', fontsize = 16)
plt.ylabel('Node', fontsize = 16)
plt.colorbar()

#for i in range(m):
#    print("max amps:")
#    print(np.max(amplitude[i,1:]))
#    print(max(amplitude[i,1:]))
#    #print((amplitude[i, int((n-1)/2)]))
#    print("min amps:")
#    print(min((amplitude[i, :])))


# plot firing rates as a heatmap
plt.figure(5)
fig = plt.imshow(firingRates, aspect='auto')#, vmin=-2, vmax = 12)
print(np.median(firingRates))
#plt.title(r"Heatmap of Log $S(\theta, t)$", fontsize = 20)
plt.title(r"$S(\theta, t)$ vs Time", fontsize = 20)
plt.xlabel(r'Steps of $\Delta t$', fontsize =16)
plt.ylabel(r'Node', fontsize = 16)
plt.colorbar()


imRate = 11.4
freq = np.fft.rfftfreq(n, 1/imRate)

#print(imRate*.5, imRate*-.5)

#plt.figure(6)
#plt.plot(freq, Ffr.real[1]**2 + Ffr.imag[1]**2)
#plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.log(Ffr.real[0,:]**2 + Ffr.imag[0,:]**2)))
#plt.plot(freq, amplitude[2])
#print(freq.shape)
#print(amplitude.shape)


plt.figure(6)
for i in range(0, 9):
    plt.plot(freq, np.log(amplitude[i,:]), alpha=0.5, label = 'node '+ str(i))
    plt.legend(loc='best')
    plt.title('Semilog Plot of Amplitude vs Frequency', fontsize = 20)
    plt.xlabel(r'Frequency $\omega$', fontsize = 16)
    plt.ylabel(r'Log of Amplitude of $\hat{S}(\omega)$', fontsize= 16)

    #print(np.mean(amplitude[i,:]))


plt.figure(7)
#plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.log(Ffr.real[0,:]**2 + Ffr.imag[0,:]**2)))
nodeNum = 1
plt.plot(freq, np.log(amplitude[nodeNum,:]), label='node ' + str(nodeNum))
plt.title('Single Node Semilog Plot of Amplitude vs Frequency', fontsize = 20)
plt.xlabel(r'Frequency $\omega$', fontsize = 16)
plt.ylabel(r'Log of Amplitude of $\hat{S}(\omega)$', fontsize= 16)
plt.legend(loc= 'best')



#plt.imshow(np.log(amplitude), aspect='auto')
plt.figure(8)
subLength = 150
FfrSub = np.fft.rfft(firingRates[nodeNum, subLength:])
freqSub = np.fft.rfftfreq(n - subLength, 1/imRate)
plt.plot(freqSub, np.log(np.abs(FfrSub))) #using real fft so no shift required. 
#print(freqSub)
plt.title('Semilog Plot of Amplitude vs Frequency, Subset', fontsize = 20)
plt.xlabel(r'Frequency $\omega$', fontsize = 16)
plt.ylabel(r'Log of Amplitude of $\hat{S}(\omega)$', fontsize= 16)
plt.legend(loc= 'best')


plt.figure(9)
tVals = np.zeros(n)
tVals = np.load('data/t_node_' + str(0) + '.npy')
subt = np.zeros(len(tVals) - 1)
subt = tVals[:-1]
subt = tVals
# the above are needed elsewhere, not necessarily for this plot. 
for i in range(0, 9):
    FfrSub = np.fft.rfft(firingRates[i, subLength:])
    plt.plot(freqSub, np.log(np.abs(FfrSub)), alpha=0.5, label = 'node '+ str(i))
    plt.legend(loc='best')
    plt.title('Semilog Plot of Amplitude vs Frequency, subset', fontsize = 20)
    plt.xlabel(r'Frequency $\omega$', fontsize = 16)
    plt.ylabel(r'Log of Amplitude of $\hat{S}(\omega)$', fontsize= 16)




#plot firing rates
plt.figure(10)
for i in range(1, 9):
    #plt.plot(subt, np.log(firingRates[i, :]), alpha=0.5, label = 'node '+ str(i))
    plt.plot(subt[150:], firingRates[i, 150:], alpha=0.5, label = 'node '+ str(i))
    plt.legend(loc = 'best')
    plt.title(r'Plot of $S(\theta, t)$ vs Time', fontsize = 20)
    #plt.title(r'Semilog Plot of $S(\theta, t)$ vs Time', fontsize = 20)
    plt.xlabel(r'$t$ (s)', fontsize = 16)
    plt.ylabel(r'Log of $S(\theta, t)$', fontsize = 16)
    plt.ylabel(r'$S(\theta, t)$', fontsize = 16)

#subset of time
plt.figure(11)
for i in range(0, 8):
    #plt.plot(subt[480:500], np.log(firingRates[i, 480:500]), alpha=0.5, label = 'node '+ str(i))
    plt.plot(subt[480:500], firingRates[i, 480:500], alpha=0.5, label = 'node '+ str(i))
    plt.legend(loc = 'best')
    plt.title(r'$S(\theta, t)$, Subset of Time', fontsize = 20)
    #plt.title(r'Log Plot of $S(\theta, t)$ vs Time')
    plt.xlabel(r'$t$ (s)', fontsize = 16)
    plt.ylabel(r'$S(\theta, t)$', fontsize = 16)
#   plt.ylabel(r'Log of $S(\theta, t)$', fontsize = 16)


# heatmap of CI_sim
plt.figure(12)
CIsim = np.zeros((m, n))
for i in range(0, m):
    sol = np.load('data/sol_node_' + str(i) + '.npy')
    CIsim[i, :] = sol[0, :]  #normalized
plt.title(r"Normalized Matrix Visualization of $CI^*_{sim}$", fontsize = 18)
plt.xlabel(r'Steps of $\Delta t$', fontsize =14)
plt.ylabel(r'Node', fontsize = 14)
plt.imshow(CIsim, aspect='auto')
plt.colorbar()

# testing plo, node 1 has different IC
plt.figure(13)
plt.plot(subt, firingRates[1, :], label = "node 1")
plt.plot(subt, firingRates[2,:], label = "node 2")
plt.title(r'$S(\theta, t)$, Few Nodes', fontsize = 20)
#plt.title(r'Log Plot of $S(\theta, t)$ vs Time')
plt.xlabel(r'$t$ (s)', fontsize = 16)
plt.ylabel(r'$S(\theta, t)$', fontsize = 16)
plt.legend(loc = 'best')
#


plt.show()
# testing if K_d (diffusion coeff) lines up with expected JRGECO val of 148
#temp = []
#for i in range(0, m):
#    sol = np.load('data/QSSA_sol_node_' + str(i) + '.npy')
#    kdSim = sol[1, :] * sol[0,:] / (sol[2,:]+.000000001) # CI*Ca^{2+}/CI^*
#    temp = np.append(temp, np.mean(kdSim))

