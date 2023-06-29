import numpy as np
import matplotlib.pyplot as plt


Ca = np.load("temp.dat.npy")
print(Ca)

print(np.max(Ca))

n = len(Ca)
t = np.linspace(0, 64, n)



plt.plot(t, Ca)
plt.title(r"Dynamis of $CA^{2+}$", fontsize = 18)
plt.show()



