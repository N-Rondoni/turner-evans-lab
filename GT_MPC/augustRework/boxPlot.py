import numpy  as np
import matplotlib.pyplot as plt


cors = np.load("data/allScores.npy")
plt.boxplot(cors)
plt.show()
