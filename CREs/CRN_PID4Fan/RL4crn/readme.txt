In order to run ODES_pid.py an integer value must be provided.
E.g., python3 ODES_pid.py 1

This specifies a row of the data set, which corresponds to a particular neuron.

driver.py loops through each of these rows. processing.py is then used to plot the resultant s of all neurons (not included here).

I hope you'll just need to redefine s to be the output of your reinforcement learning model, instead of the PID it currently is. 

