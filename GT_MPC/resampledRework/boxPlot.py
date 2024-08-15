import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cors = np.load("data/allScores.npy")
cors1 = np.load("data/allScoresDset1.npy")
cors2 = np.load("data/allScoresDset2.npy")
cors3 = np.load("data/allScoresDset3.npy")
cors4 = np.load("data/allScoresDset4.npy")
cors5 = np.load("data/allScoresDset5.npy")
#print(cors1)

labels = ["1", "2", "3", "4", "5", "All"]

plt.figure(1)
plt.title(r"Box plots of correlation scores by data set, $n_{all} = $" + str(np.shape(cors)[0]), fontsize = 18)
plt.xlabel("Data Set")
plt.ylabel("Correlation Coefficient")
plt.boxplot([cors1, cors2, cors3, cors4, cors5, cors], tick_labels=labels)


plt.figure(2)
colors = np.random.rand(len(cors))
colors = cm.rainbow(np.linspace(0, 1, len(cors)))
plt.title("Scatter plot of correlation scores")
plt.scatter(cors, cors, c=colors, alpha = 0.7, marker = ".")


plt.figure(3)
binVals= np.linspace(0,1,11)
print(binVals)
plt.hist(cors)#, bins = binVals)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coefficient", fontsize = 14)
plt.title("Correlation coefficients across all datasets", fontsize = 18)
plt.show()






