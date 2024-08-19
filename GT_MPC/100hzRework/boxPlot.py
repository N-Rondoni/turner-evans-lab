import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cors = np.load("data/allScores.npy")
cors1 = np.load("data/allScoresDset1.npy")
cors2 = np.load("data/allScoresDset2.npy")
cors3 = np.load("data/allScoresDset3.npy")
cors4 = np.load("data/allScoresDset4.npy")
cors5 = np.load("data/allScoresDset5.npy")
cors6 =  np.load("data/allScoresDset6.npy")
cors7 =  np.load("data/allScoresDset7.npy")
cors8 =  np.load("data/allScoresDset8.npy")
cors9 =  np.load("data/allScoresDset9.npy")
allVPDs = np.load("data/allVPDs.npy")


#print(cors1)

labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "All"]

plt.figure(1)
plt.title(r"Box plots of correlation scores by data set, $n_{all} = $" + str(np.shape(cors)[0]), fontsize = 18)
plt.xlabel("Data Set")
plt.ylabel("Correlation Coefficient")
plt.boxplot([cors1, cors2, cors3, cors4, cors5, cors6, cors7, cors8, cors9, cors], labels = labels) #labels depreciated, will break in newer ver


plt.figure(2)
colors = np.random.rand(len(cors))
colors = cm.rainbow(np.linspace(0, 1, len(cors)))
plt.title("Scatter plot of correlation scores")
plt.scatter(cors, cors, c=colors, alpha = 0.7, marker = ".")


plt.figure(3)
binVals= np.linspace(0,1,11)
#print(binVals)
plt.hist(cors)#, bins = binVals)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coefficient", fontsize = 14)
plt.title("Correlation coefficients across all datasets", fontsize = 18)

# plotting VP distances

plt.figure(4)
plt.hist(allVPDs, color='c', edgecolor="k",bins=20)
plt.axvline(np.median(allVPDs), color='k', linestyle="dashed", alpha = 0.65)
#plt.axvline(np.mean(allVPDs), color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distance, all datasets", fontsize = 18)
min_ylim, max_ylim = plt.ylim()
#plt.text(np.mean(allVPDs)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(allVPDs)))
plt.text(np.median(allVPDs)*1.1, max_ylim*0.9, 'Median: {:.2f}'.format(np.median(allVPDs)))


plt.figure(5)
plt.hist(allVPDs[allVPDs <= 10000])
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distance, no outlier", fontsize = 18)

plt.figure(6)
plt.hist(allVPDs[allVPDs <= 2000])
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distances for datasets less than 2000", fontsize = 18)


plt.figure(7)
plt.boxplot(allVPDs)
plt.title(r"Box plot of VP distance, $n_{all} = $" + str(np.shape(allVPDs)[0]), fontsize = 18)



plt.show()

