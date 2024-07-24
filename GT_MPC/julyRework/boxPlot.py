import numpy  as np
import matplotlib.pyplot as plt


cors = np.load("allScores.npy")
plt.boxplot(cors)
plt.show()
