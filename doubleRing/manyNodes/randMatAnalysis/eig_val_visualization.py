# purupose: visualize eigenvalues (plot them) from random matrix created in random_mat_eig_vals.py
# data is written out to eig_vals_rand_mat.dat 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import solve_ivp
import os


def scatterFriend(x, y):
    plt.figure()
    plt.scatter(x, y)
    plt.title('Eigenvalues of Matrix used to Evolve System')
    plt.xlabel('Real')
    plt.ylabel('Im')
    filename = 'eig_vals_visualized.png'
    plt.savefig(filename)
    os.system('cp ' + filename + ' /mnt/c/Users/nicho/Pictures/doubleRing/many_node/UQ') 
    plt.show()



if __name__ =="__main__":
    eigvals = np.load("eVals_random_matrix.npy")
    real_part = np.real(eigvals)
    im_part = np. imag(eigvals)

    print(eigvals)

    scatterFriend(real_part, im_part)



