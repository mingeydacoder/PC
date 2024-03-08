import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('TS.txt', unpack='true')

mean = np.nanmean(data)
standard_deviation = np.std(data)
print('Mean:',mean)
print('Standard Deviaion:',standard_deviation)

print('± 1 std:', mean+standard_deviation, mean-standard_deviation,'\n± 2 std:', 
    mean+2*standard_deviation,mean-2*standard_deviation,'\n± 3 std:',
    mean+3*standard_deviation,mean-3*standard_deviation,'\n')

condition_1 =  np.logical_and(mean-standard_deviation < data, data < mean+standard_deviation)
condition_2 =  np.logical_and(mean-2*standard_deviation < data, data < mean+2*standard_deviation)
condition_3 =  np.logical_and(mean-3*standard_deviation < data, data < mean+3*standard_deviation)

within_1std = data[condition_1]
within_2std = data[condition_2]
within_3std = data[condition_3]

print((len(within_1std)/len(data))*100,'%')
print((len(within_2std)/len(data))*100,'%')
print((len(within_3std)/len(data))*100,'%')

