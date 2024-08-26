import numpy as np
import sys
import scipy.io
#import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'

file_path = 'data/flymat.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape


firingRates = np.zeros((m, n-1)) #sub 1 because of the way sVec created
for i in range(m):
    firingRates[i, :] = np.load('data/QSSA_s_node_' + str(i) + '.npy')

#firingRates = firingRates - 40



#showcase what is hopefuly the firing rates


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
tVals = np.load('data/QSSA_t_node_' + str(0) + '.npy')
subt = np.zeros(len(tVals) - 1)
subt = tVals[:-1]
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
    sol = np.load('data/QSSA_sol_node_' + str(i) + '.npy')
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


# testing if K_d (diffusion coeff) lines up with expected JRGECO val of 148
#temp = []
#for i in range(0, m):
#    sol = np.load('data/QSSA_sol_node_' + str(i) + '.npy')
#    kdSim = sol[1, :] * sol[0,:] / (sol[2,:]+.000000001) # CI*Ca^{2+}/CI^*
#    temp = np.append(temp, np.mean(kdSim))

#print(temp)
plt.figure(14)
tspanFire = np.linspace(0, np.ceil(tVals[-1]), int(np.ceil(tVals[-1])))
amount = 9
posIncreases = np.zeros((amount, 66)) #sub 1 because of the way sVec created
for i in range(amount):
    posIncreases[i, :] = np.load('data/QSSA_PosIncreases_node_' + str(i) + '.npy')
    plt.plot(tspanFire[1:], posIncreases[i, 1:])


plt.figure(15)
plt.imshow(posIncreases[:,1:])


plt.show()

# heatmap of CI_sim + S
#plt.figure(12)
#SimS = (CIsim[:, :-1]) + (firingRates -40)
#plt.imshow(SimS[:, 150:], aspect = 'auto', vmin=-3, vmax=12)
#plt.colorbar()

#for i in range(0, 9):
#    plt.plot(tVals, CIsim[i] , alpha=0.5, label = 'node '+ str(i))
#    plt.legend(loc = 'best')
#    plt.title(r'$S(\theta, t)$ vs Time')
#    plt.xlabel(r'$t$ (s)')
#    plt.ylabel(r'$S(\theta, t)$')




#plt.plot(firingRates[1, :)]


#plt.show()

# testing print statements
#print(Ffr[1, 1:10])
#inv1 = np.fft.ifft(Ffr)
#print(firingRates[1, 1:10])
#print(inv1[1, 1:10])
#print("___")
#subset = np.fft.fft(firingRates[1, :])
#print(subset[1:10])
#print(np.fft.ifft(subset))


