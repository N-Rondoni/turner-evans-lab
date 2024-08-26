import os
import sys
import scipy.io

file_path = 'data/flymat.mat' #must match file in ODES_pid.py 
mat = scipy.io.loadmat(file_path)
data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape

for i in range(m):
    #print(('python PIDmain.py ' + str(i)))
    os.system('python3 ODES_pid.py ' + str(i))

