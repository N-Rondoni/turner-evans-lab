import numpy as np
import sys
import scipy.io
#import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'data/flymat.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape


firingRates = np.zeros((m, n-1)) #sub 1 because of the way sVec created
for i in range(m):
    firingRates[i, :] = np.load('data/s_node_' + str(i) + '.npy')

firingRates = firingRates - 40


#showcase what is hopefuly the firing rates
#mat = plt.matshow(firingRates[:, 100:])
#plt.colorbar(mat)


#compare this to actual data
plt.figure(1)
plt.imshow(data[:, 300:])
plt.colorbar()

# fft business, plot amplitudes
plt.figure(2)
Ffr = np.fft.fft(firingRates)
amplitude = np.abs(Ffr)
#print(amplitude[1, 0:100])
#print(amplitude.shape)
plt.title("Amplitude")
fig = plt.imshow(np.log(amplitude)) #normalized
plt.colorbar()

# plot power spectrum
plt.figure(3)
Ffr = np.fft.fft(firingRates)
amplitude = np.abs(Ffr)
powerSpec = np.abs(Ffr)**2
#print(amplitude[1, 0:100])
#print(amplitude.shape)
plt.title("Power spectrum")
fig = plt.imshow(powerSpec[:, 1:])
plt.colorbar()


# plot phase spectrum
plt.figure(4)
Ffr = np.fft.fft(firingRates)
phase = np.angle(Ffr)
#print(amplitude[1, 0:100])
#print(amplitude.shape)
plt.title("Phase Spectrum")
fig = plt.imshow(phase[:, 1:])
plt.colorbar()



plt.figure(5)
imRate = 11.4
m, n = Ffr.shape

freq = np.fft.fftfreq(n, 1/imRate)

#plt.plot(freq[1:10], Ffr.real[1][1:10], freq[1:10], Ffr.imag[1][1:10])
plt.plot(freq[1:], Ffr.real[1][1:], label = "real part") 
plt.plot(freq[1:], Ffr.imag[1][1:], label = "real part")
#plt.plot(freiq, Ffr.imag[1])
#plt.xlim(-100, 100)


plt.figure(6)
#plt.plot(freq, Ffr.real[1]**2 + Ffr.imag[1]**2)
plt.plot(freq, np.log(Ffr.real[0,:]**2 + Ffr.imag[0,:]**2))
#plt.plot(freq, amplitude[2])


plt.figure(7)
plt.plot(np.fft.fftshift(freq), np.fft.fftshift(np.log(Ffr.real[0,:]**2 + Ffr.imag[0,:]**2)))


#print(Ffr[:, 0])

print(Ffr[:, 1:10])
print(freq[0:10])

plt.show()

# testing print statements
#print(Ffr[1, 1:10])
#inv1 = np.fft.ifft(Ffr)
#print(firingRates[1, 1:10])
#print(inv1[1, 1:10])
#print("___")
#subset = np.fft.fft(firingRates[1, :])
#print(subset[1:10])
#print(np.fft.ifft(subset))


