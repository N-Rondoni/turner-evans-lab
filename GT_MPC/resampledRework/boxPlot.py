import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataParser import pcc, medianFromDset
#import dataParser
import pandas as pd


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


#plt.figure(2)
#colors = np.random.rand(len(cors))
#colors = cm.rainbow(np.linspace(0, 1, len(cors)))
#plt.title("Scatter plot of correlation scores")
#plt.scatter(cors, cors, c=colors, alpha = 0.7, marker = ".")


plt.figure(3)
binVals= np.linspace(0,1,11)
#print(binVals)
stmMeanCor = 0.3866859940274038 # found with computations below
plt.hist(cors, color = 'c', edgecolor = 'k')#, bins = binVals)
plt.axvline(np.mean(cors), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(stmMeanCor, color = 'r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coefficient", fontsize = 14)
plt.title("Correlation coefficients across all datasets", fontsize = 18)
min_ylim, max_ylim = plt.ylim()
#plt.text(np.mean(allVPDs)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(allVPDs)))
plt.text(np.mean(cors)*0.6, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(cors)))
plt.text(stmMeanCor*1.2, max_ylim*0.9, 'STM Mean: {:.2f}'.format(np.mean(cors)), color = 'r')

# plotting VP distances

plt.figure(4)
plt.hist(allVPDs, color='violet', edgecolor="k",bins=20)
plt.axvline(np.median(allVPDs), color='k', linestyle="dashed", alpha = 0.65)
#plt.axvline(np.mean(allVPDs), color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distance, all datasets", fontsize = 18)
min_ylim, max_ylim = plt.ylim()
#plt.text(np.mean(allVPDs)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(allVPDs)))
plt.text(np.median(allVPDs)*1.4, max_ylim*0.9, 'Median: {:.2f}'.format(np.median(allVPDs)))


plt.figure(6)
plt.hist(allVPDs[allVPDs <= 2000])
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distances for datasets less than 2000", fontsize = 18)

file_path1 = 'data/results_22_06_17.csv'
df = pd.read_csv(file_path1)
stm_df = df[(df['algo'] == 'stm')]
#print(stm_df)
print(stm_df['value'].mean()) # used to place mean line in earlier plot


# swapped to median, keeping names as mean bc bad programmer
meanD1 = medianFromDset(df, 'deneux', 1)
meanD2 = medianFromDset(df, 'deneux', 2)
meanD3 = medianFromDset(df, 'deneux', 3)
meanD4 = medianFromDset(df, 'deneux', 4)
meanD5 = medianFromDset(df, 'deneux', 5)

medianS1 = medianFromDset(df, 'stm', 1)
medianS2 = medianFromDset(df, 'stm', 2)
medianS3 = medianFromDset(df, 'stm', 3)
medianS4 = medianFromDset(df, 'stm', 4)
medianS5 = medianFromDset(df, 'stm', 5)


# all the below say mean, should be median, don't want to rename them all
figNo = 8
plt.figure(figNo)
simDat = cors1
methodsMean = medianS1
dset = 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors2
methodsMean = medianS2
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors3
methodsMean = medianS3
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors4
methodsMean = medianS4
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors5
methodsMean = medianS5
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Median: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Median: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1



plt.show()

