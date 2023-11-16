import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from windrose import WindroseAxes

new_directory = "/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week9_lab/01"
os.chdir(new_directory)
#read files
filelist = []

for i in range(2010,2020,1):
    for k in range(101,132,1):
        name = '47918.'+str(i)+'_0'+str(k)+'_00.txt'
        filelist.append(name)


data_dict = {}


for j in range(310):
    data = np.loadtxt(filelist[j], skiprows=5, unpack='true')
    data_dict[j] = data
    j = j+1


index_850 = np.zeros(310)
for a in range(310):
    for b in range(0,16,1):
        if data_dict[a][0][b] == 850.0:  
            index_850[a] = int(b)

index_850 = index_850.astype(int)

wind_speed = np.zeros(310)
wind_direction = np.zeros(310)


for x in range(310):
    wind_speed[x] = data_dict[x][7][index_850[x]]*0.5144
    wind_direction[x] = data_dict[x][6][index_850[x]]

theta = np.deg2rad(wind_direction)
print(theta)


cmap = mlp.colormaps['rainbow']
ax = WindroseAxes.from_ax()
ax.bar(wind_direction, wind_speed, normed=True, opening=0.8, edgecolor='white', cmap=cmap, nsector = 16,bins=np.linspace(0,15,6))
ax.set_yticks([0,3,6,9,12,15])
ax.set_yticklabels([0,3,6,9,12,15])
ax.set_legend(labels = ['0-3','3-6','6-9','9-12','12-15','>15'])
ax.set_title('Windrose at 00Z in Jan.,2010-2019 \n 47918: Ishigaki Island',fontsize=20)
