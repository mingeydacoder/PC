import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import os

new_directory = "/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week9_lab/01"
os.chdir(new_directory)
#read files
filelist = []

for i in range(101,132,1):
    name = '47918.2016_0'+str(i)+'_00.txt'
    filelist.append(name)


data_dict = {}

i = 101
for file_name in filelist:
    data = np.loadtxt(file_name, skiprows=5, unpack='true')
    data_dict[i] = data
    i = i+1

missing_value = 999.9

blh = np.zeros(31)

for k in range(31):
    blh[k] = data_dict[k+101][1][1] 

blh[2] = data_dict[103][1][2]
blh[7] = data_dict[108][1][2]
blh[16] = data_dict[117][1][2]
blh[21] = data_dict[122][1][2]
blh[23] = data_dict[124][1][3]
blh[24] = data_dict[125][1][2]
blh[28] = data_dict[129][1][2]

time = np.linspace(1,31,31)

plt.plot(time,blh)
plt.xticks(ticks=[1,5,10,15,20,25,30])
plt.xlabel('Day')
plt.ylabel('Height [m]')
plt.title('BLH 2016 Jan time series')

print(time)









