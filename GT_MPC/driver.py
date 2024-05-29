import os
import sys
import scipy.io

#row = 4

    
file_path = 'data/' + str(dset) + '.test.calcium.csv'
data1 = pd.read_csv(file_path).T 
data1 = np.array(data1)
subsetAmount = np.max(np.shape(data1[row,:])) # the way its set up, must be divisble by factor or stuff breaks.     
CI_Meas = data1[row, :subsetAmount]
m,n = data1.shape



dsets = [1, 5]

for dset in dsets
    print(dset)
    print(m, n)


