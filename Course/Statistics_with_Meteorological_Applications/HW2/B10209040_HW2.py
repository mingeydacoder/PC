import numpy as np
from tqdm import tqdm 

data1 = np.loadtxt('HW2_Taipei_Ts_1961_1970.txt')
data2 = np.loadtxt('HW2_Taipei_Ts_2001_2010.txt')
data3 = np.loadtxt('HW2_Taitung_Ts_1961_1970.txt')
data4 = np.loadtxt('HW2_Taitung_Ts_2001_2010.txt')

#1.1 mean ts on July

def mean_Ts (data):
    sum = 0
    count = 0
    for i in tqdm(range(10),colour="green"):
        sum = sum + np.nansum(data[:,i+186:i+216])
        count = count + np.count_nonzero(~np.isnan(data[:,i+186:i+216]))
        i += 372
    print(sum/count)

mean_Ts(data1)
mean_Ts(data2)
mean_Ts(data3)
mean_Ts(data4)

#1.2 min & MAX

def extreme_value(data):
    min = np.zeros([10,31])
    max = np.zeros([10,31])
    for i in tqdm(range(10),colour='blue'):
        for j in range(31):
            min[i,j] = np.nanmin(data[:,i+j+186])
            max[i,j] = np.nanmax(data[:,i+j+186])
            j += 1
        i += 372
    #print(max)
    print(np.nanmean(min))
    print(np.nanmean(max))

extreme_value(data1)
extreme_value(data2)
extreme_value(data3)
extreme_value(data4)