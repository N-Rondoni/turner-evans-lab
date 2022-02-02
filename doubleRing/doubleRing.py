#########################################################################
# Implementation of Double Ring model put forward by Xie et al
# for left and right rings, we have coupled ODEs:
# 
# \tau dS_l/dt + s_l = f_l
#
# where s_l and f_l are funtions of \theta, t. 
# solving for s_l and s_r, which represent
# synaptic activation indexed by \theta at time t
#
# Author: Nick Rondoni
#########################################################################


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.stats import vonmises


# ------------------------------------
# Define constants to be used in model
# ____________________________________

# tau: neuron time constant, in seconds
# represents travel time of signal between synapses. Usually very small, around 80ms

tau = 0.080


# J0, J1, K0, K1: Synaptic connection parameters
# Determines the connection strenth

[J0, J1, K0, K1] = [-60, 80, -5, 80]


# phi, psi: Angular offsets, entered in degrees 
# or comment out degree entries and enter in radians
phiDeg = 90
psiDeg = 45
phi = phiDeg*(pi)/180
psi = psiDeg*(pi)/180


# b_l, b_r: Neurons in each ring's uniform feedforward input
# b_0 starting position, delta_b change in position
# These appear in the forcing function, like a kick to get things started
# Paper often discusses delta_b_normalised = delta_b / b0
b0 = ?
delta_b = ?
b_r = b0 + delta_b
b_l = b0 - delta_b



#----------------------------------------
# Define weight functions, used in definition of forcing functions
#________________________________________

def WeightSame(theta):
    y = J0 + J1*np.cos(theta)

def WeightDifferent(theta):
    y = K0 + K1*np.cos(theta) 


#----------------------------------------
# Define forcing functions
#________________________________________





if __name__ == '__main__':
    print('soup')
    print(phi)
