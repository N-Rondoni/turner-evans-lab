# purupose: visualize eigenvalues (plot them) 
# data is written out to eVals.npy 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
from scipy import stats 
import os


def scatterFriend(x, y):
    plt.figure()
    plt.scatter(x, y)
    plt.title('Eigenvalues of Matrix W')
    plt.xlabel('Real')
    plt.ylabel('Im')
    #filename = 'MC_eig_vals_visualized.png'
    #plt.savefig(filename)
    #os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node/UQ') 
    plt.show()


def plotFriend(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.title('Estimated PDF Evaluated on Real Part of Eigenvalues')
    plt.xlabel("Real Part of Eigenvalues")
    plt.ylabel("Density")
    #filename = 'PDF_eigvals.png'
    #plt.savefig(filename)
    #os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node/UQ') 
    plt.show()


if __name__ =="__main__":
    #eigvals = np.load("eVals_oddOff.npy")
    eigvals = np.load("eVals.npy")
    real_part = np.real(eigvals)
    real_part = np.sort(real_part)
    im_part = np.imag(eigvals)
    #print(len(real_part))
    kernel = stats.gaussian_kde(real_part)
    
    print("Mean of real part of maximal eig vals:")
    print(np.sum(real_part)*(1/len(real_part)))

    print("Covariance of real part of maximal eig vals")
    print(kernel.covariance)

    # estimate pdf:
    #print((kernel.evaluate(real_part)*real_part))
    
    # evaluate estimated pdf on real part of eig vals.
    y = kernel.pdf(real_part)

    # check that it is pretty close to integrating to 1. 
    print(kernel.integrate_box(-20, 20))
    
    scatterFriend(real_part, im_part)
    plotFriend(real_part, y)


