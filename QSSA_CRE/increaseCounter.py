import numpy as np

y = [1, 3, 6, 3, 4, 6, 7, 2, 5, 8, 11, 14, 
    10, 15, 13, 10, 1, 9, 45, 20, 12, 14, 16,
    1, 5, 6, 9] 
t =   [0, 0.08783752, 0.17567504, 0.26351255, 0.35135007, 0.43918759, 0.52702511, 0.61486263, 0.70270015, 0.79053766, 0.87837518, 0.9662127,
 1.05405022, 1.14188774, 1.22972526, 1.31756277, 1.40540029, 1.49323781, 1.58107533, 1.66891285, 1.75675037, 1.84458788, 1.9324254, 
 2.02026292, 2.10810044, 2.19593796, 2.28377548]
#print(len(t))
#print(len(y))
sec = 1
Accum = np.zeros(int(np.ceil(t[-1])))
runDif = 0
tempDif = 0
for i in range(len(y) - 1):
    tempDif = y[i+1] - y[i] 
    if t[i] < sec:
        if tempDif > 0:
            runDif = tempDif + runDif
    if t[i] > sec:
        Accum[sec - 1] = runDif
        runDif = 0
        sec = sec + 1


print(np.gradient(y))
print(y)
print(Accum)
#sec + 1 






