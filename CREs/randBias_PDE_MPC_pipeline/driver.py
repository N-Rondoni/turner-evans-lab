import os
import sys
import scipy.io

state = 'Stripe'
#state = 'Dark'

file_path = 'data/ciDat' + state + '.mat' #must match file in PIDmain.py 
mat = scipy.io.loadmat(file_path)
data = mat['ciDat'+ state] #had to figure out why data class is flyDat from print(mat). No clue. 
m,n = data.shape

for i in range(m):
    os.system('python3 MPCmain.py ' + str(i))


os.system("python3 analyticVelocitySR.py")  # computes analytic soln

os.system("python3 noisedVelocitySR.py")    # computes noised solution to pde 

#os.system('python3 processingMPC.py')       # computes difference between the two solutions

os.system("python3 controlVelocitySR.py")   # solves pde again with control signal.

os.system("python3 processingFinal.py")   # solves pde again with control signal.


#os.system('rm *.png')


