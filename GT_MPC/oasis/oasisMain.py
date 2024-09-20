import numpy as np
import matplotlib.pyplot as plt
from sys import path
path.append('..')
from oasis.functions import gen_data, gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
import time
import sys


start = time.time()


row = int(sys.argv[1])   
dset = int(sys.argv[2])
stat = str(sys.argv[3])


file_path_calc = 'data/resampled/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.calcium.npy'
file_path_spike = 'data/resampled/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.spikes.npy'

trueSpikes = np.load(file_path_spike)

tempCalc = np.load(file_path_calc) 
subsetAmount = np.shape(tempCalc)[0]
CI_Meas = tempCalc[:subsetAmount]

c, s, b, g, lam = deconvolve(CI_Meas, penalty=1)

stop = time.time()
runtime = stop - start
print("run time:", runtime, "s")

# save for use in processingOasis
saveLoc = 'data/resampled/solutions/node'+ str(row) + '_dset' + str(dset) + '.' + str(stat) + '.sVals.Oasis'
np.save(saveLoc, s)

print("corrCoef: ", np.corrcoef(s, trueSpikes)[0,1])

