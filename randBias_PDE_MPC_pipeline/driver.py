import os
import sys
import scipy.io

state = 'Stripe'
#state = 'Dark'

file_path = 'data/ciDat' + state + '.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['ciDat'+ state] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape

os.system("python3 PDEanalyticVelocitySR.py")

for i in range(m):
    #print(('python PIDmain.py ' + str(i)))
    os.system('python3 MPCmain.py ' + str(i))

os.system('rm *.png')

os.system('python3 processingMPC.py')
