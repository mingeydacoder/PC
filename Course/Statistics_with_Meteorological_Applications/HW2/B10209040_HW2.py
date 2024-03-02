import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

data1 = np.loadtxt('HW2_Taipei_Ts_1961_1970.txt')
data2 = np.loadtxt('HW2_Taipei_Ts_2001_2010.txt')
data3 = np.loadtxt('HW2_Taitung_Ts_1961_1970.txt')
data4 = np.loadtxt('HW2_Taitung_Ts_2001_2010.txt')

'''
#1.1 mean ts on July

def mean_Ts (data):
    sum = 0
    count = 0
    for i in tqdm(range(10),colour="green"):
        for j in range(31):
            sum = sum + np.nansum(data[:,i+j+186:i+j+216])
            count = count + np.count_nonzero(~np.isnan(data[:,i+j+186:i+j+216]))
            j += 1
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
    print('min =',np.nanmean(min))
    print('MAX =',np.nanmean(max))

extreme_value(data1)
extreme_value(data2)
extreme_value(data3)
extreme_value(data4)


#1.3 DTR & variance


def DTR(data):
    min = np.zeros([10,31])
    max = np.zeros([10,31])
    dtr = np.zeros([10,31])
    for i in tqdm(range(10),colour='blue'):
        for j in range(31):
            min[i,j] = np.nanmin(data[:,i+j+186])
            max[i,j] = np.nanmax(data[:,i+j+186])
            j += 1
        i += 372
    dtr = max - min 
    print('Mean Diurnal Temp Range:',np.nanmean(dtr))

DTR(data1)
DTR(data2)
DTR(data3)
DTR(data4)


def VAR(data):
    july_data = np.empty((24,31,10))
    for i in tqdm(range(10)):
        data = np.array(data)
        seg = data[:,i*372+186:i*372+217]
        july_data[:,:,i] = seg
    var = np.nanvar(july_data)
    print(var)
VAR(data1)
VAR(data2)
VAR(data3)
VAR(data4)

'''
#1.4 Averege diurnal cycle

def ADC(data):
    sum = 0
    count = 0
    adc = np.empty(24)
    for hr in tqdm(range(24),colour="red"):
        for i in range(10):
            for j in range(31):
                sum = sum + np.nansum(data[hr,i+j+186:i+j+216])
                count = count + np.count_nonzero(~np.isnan(data[hr,i+j+186:i+j+216]))
                j += 1
            i += 372
        adc[hr] = sum/count
    return(adc)



x = np.linspace(0,23,24)
y = ADC(data1)
y2 = ADC(data2)
y3 = ADC(data3)
y4 = ADC(data4)

plt.plot(x,y)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.title('Average Diurnal Cycle for 1960s and 2000s')
plt.xticks(np.linspace(0,23,24))
plt.xlabel('hours')
plt.ylabel('temp. [c]')
plt.legend(['Taipei 1960s','Taipei 2000s','Taitung 1960s','Taitung 2000s'])

