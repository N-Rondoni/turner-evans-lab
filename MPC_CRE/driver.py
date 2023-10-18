import os
import sys
import scipy.io

file_path = 'data/flymat.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['flyDat'] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape

for i in range(m):
    #print(('python PIDmain.py ' + str(i)))
    os.system('python3 MPCmain.py ' + str(i))

os.system('rm *.png')

os.system('python3 processing.py')
