import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from dataParser import pcc, medianFromDset, meanFromDset
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
plt.title(r"Correlation scores by data set, $n_{all} = $" + str(np.shape(cors)[0]), fontsize = 18)
plt.xlabel("Data Set", fontsize = 14)
plt.ylabel("Correlation Coefficient", fontsize = 14)
plt.boxplot([cors1, cors2, cors3, cors4, cors5, cors6, cors7, cors8, cors9, cors], labels = labels) #labels depreciated, will break in newer ver


#plt.figure(2)
#colors = np.random.rand(len(cors))
#colors = cm.rainbow(np.linspace(0, 1, len(cors)))
#plt.title("Scatter plot of correlation scores")
#plt.scatter(cors, cors, c=colors, alpha = 0.7, marker = ".")


plt.figure(3)
binVals= np.linspace(0,1,11)
#print(binVals)
#stmMeanCor = 0.3866859940274038 # found with computations below, only on test data. 
stmMeanCor = .4244426
oasisMeanCorInfo = 0.45 #$ from jneursci stringer paper
oasisMeanCorNoInfo = 0.3397627 
oasisInfoColor = "magenta"#"hotpink"#"darkorchid"#"hotpink"
oasisNoInfoColor = "green"#"Seagreen"#"indigo"#"seagreen"
plt.hist(cors, color = 'c', edgecolor = 'k')#, bins = binVals)
plt.axvline(np.mean(cors), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(stmMeanCor, color = 'r', linestyle="dashed", alpha = 0.65)
plt.axvline(oasisMeanCorInfo, color = oasisInfoColor, linestyle="dashed", alpha = 0.65)
plt.axvline(oasisMeanCorNoInfo, color = oasisNoInfoColor, linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =16)
plt.xlabel("Correlation Coefficient", fontsize = 16)
plt.title("Correlation Coefficients, All Datasets", fontsize = 20)
min_ylim, max_ylim = plt.ylim()
#plt.text(np.mean(allVPDs)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(allVPDs)))
plt.text(np.mean(cors)*.89, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(cors)))
plt.text(stmMeanCor*1.01, max_ylim*0.9, 'STM Mean: {:.2f}'.format(float(stmMeanCor)), color = 'r')
plt.text(oasisMeanCorInfo*1.01, max_ylim*0.8, 'Oasis Mean: {:.2f}'.format(float(oasisMeanCorInfo)), color = oasisInfoColor)
plt.text(oasisMeanCorNoInfo*.8, max_ylim*0.9, 'Oasis* Mean: {:.2f}'.format(float(oasisMeanCorNoInfo)), color = oasisNoInfoColor)


# plotting VP distances
oasisVPDmedian = 901.39 # computed in oasis directory, boxplot.py 
plt.figure(4)
plt.hist(allVPDs[allVPDs<=7000], color='violet', edgecolor="k",bins=20)
plt.axvline(np.median(allVPDs), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(oasisVPDmedian, color=oasisNoInfoColor, linestyle="dashed", alpha = 0.65)
#plt.axvline(np.mean(allVPDs), color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =16)
plt.xlabel("Victor-Purpura Distance", fontsize = 16)
plt.title("Victor-Purpura Distance, All Datasets", fontsize = 20)
min_ylim, max_ylim = plt.ylim()
#plt.text(np.mean(allVPDs)*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(allVPDs)))
plt.text(np.median(allVPDs)*(-.1), max_ylim*0.96, 'Median: {:.2f}'.format(np.median(allVPDs)))
plt.text(oasisVPDmedian*1.1, max_ylim*0.7, 'Oasis* Median: {:.2f}'.format(oasisVPDmedian), color=oasisNoInfoColor)


plt.figure(6)
plt.hist(allVPDs[allVPDs <= 2000])
plt.ylabel("Count", fontsize =14)
plt.xlabel("Victor-Purpura Distance", fontsize = 14)
plt.title("Victor-Purpura distances for datasets less than 2000", fontsize = 18)

file_path1 = 'data/results_22_06_17.csv'
file_path2 = 'data/spikefinder.csv'
df = pd.read_csv(file_path1)
stm_df = df[(df['algo'] == 'stm')]
print("stm average on test data:", stm_df['value'].mean()) # used to place mean line in earlier plot



#df1 = pd.read_csv(file_path2)
stm_df = pcc(file_path2, 'stm')
#print(list(stm_df.keys()))
#print(stm_df['1.test.spikes'])
j = 0
runSum = 0
accumed = []
for key in stm_df:
    print(key)
    temp = stm_df[key]
    res = float(temp[0])
    accumed = np.append(accumed, res)
    runSum = runSum + res
    j = j + 1
print("average for stm across all datasets:", runSum/j)
print("median across all for stm:", np.median(accumed))


print("stm dset 7 mean:", float(stm_df['7.train.spikes'][0]))
print("stm dset 3 mean:", float(stm_df['3.test.spikes'][0]))


#stm_df = df1[(df1['algo_algorithm'] == 'stm')]
#print(stm_df['value'].mean()) # used to place mean line in earlier plot
#print(stm_df.keys())
#
medianD1 = medianFromDset(df, 'deneux', 1) # pulls median of both train and test data together
medianD2 = medianFromDset(df, 'deneux', 2)
medianD3 = medianFromDset(df, 'deneux', 3)
medianD4 = medianFromDset(df, 'deneux', 4)
medianD5 = medianFromDset(df, 'deneux', 5)

medianS1 = medianFromDset(df, 'stm', 1)
medianS2 = medianFromDset(df, 'stm', 2)
medianS3 = medianFromDset(df, 'stm', 3)
medianS4 = medianFromDset(df, 'stm', 4)
medianS5 = medianFromDset(df, 'stm', 5)


meanS3 = meanFromDset(df, 'stm', 3)
meanS2 = meanFromDset(df, 'stm', 2)

# all the below say mean, should be median
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
methodsMean = meanS2
dset = dset + 1
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors3
methodsMean = meanS3
dset = dset + 1
dset = 3
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.001, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat))) #fontsize = 14) too big
plt.text(methodsMean*1.001, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r') #fontsize = 14)
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



plt.figure(figNo)
simDat = cors7
methodsMean = float(stm_df['7.train.spikes'][0])
dset = 7
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.01, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*.7, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1

plt.figure(figNo)
simDat = cors8
methodsMean = float(stm_df['8.train.spikes'][0])
dset = 8
plt.hist(simDat, color = 'c', edgecolor='k')
plt.axvline(np.mean(simDat), color='k', linestyle="dashed", alpha = 0.65)
plt.axvline(methodsMean, color='r', linestyle="dashed", alpha = 0.65)
plt.ylabel("Count", fontsize =14)
plt.xlabel("Correlation Coeff.", fontsize = 14)
plt.title("Correlation Coefficient, dataset " + str(dset), fontsize = 18)
min_ylim, max_ylim = plt.ylim()
plt.text(np.mean(simDat)*1.01, max_ylim*0.9, 'Mean: {:.2f}'.format(np.mean(simDat)))
plt.text(methodsMean*.55, max_ylim*0.9, 'STM Mean: {:.2f}'.format(methodsMean), color='r')
figNo = figNo + 1




plt.show()

