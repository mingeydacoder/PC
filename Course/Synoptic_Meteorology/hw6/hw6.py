import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mecalc
from metpy.units import units


filelist = []
num = []

for i in range(6,15,1):
    num.append(f"{i:02}")

for i in range(9):
    name = 'TNNUA_20210402_'+num[i]+'00_L4.txt'
    filelist.append(name)


data_dict = {}

i = 0
plt.figure(figsize=[4.8,6.4])
for file_name in filelist:
    data = np.loadtxt(file_name, skiprows=14, unpack='true')
    data_dict[i] = data
    i = i+1


for j in range(9):
    plt.plot(data_dict[j][5], data_dict[j][16])

plt.legend(['06','07','08','09','10','11','12','13','14'])
plt.ylim([0,3000])
plt.xlim([10,40])
plt.xlabel('Temperature [c]')
plt.ylabel('Height [m]')
plt.grid()
plt.plot([24, 0], [0, 2400], ls="--", c=".3")
plt.plot([34, 0], [0, 3400], ls="--", c=".3")

A = np.array([625,9])
path = 'output.txt'
f = open(path, 'w')

plt.figure(figsize=[6.4,6.4])
for k in range(9):
    theta = mecalc.potential_temperature(data_dict[k][4]*units.mbar, (data_dict[k][5]+273.15)*units.kelvin)
    plt.plot(theta, data_dict[k][16])
    print(theta[0:10])
    content = str(theta)
    f.write(content)
    f.write("\n")


plt.plot(295.23,112.44,'ro') 
plt.plot(296.94,125.69,'ro') 
plt.plot(304.64,811.76,'ro') 
plt.plot(303.96,876.98,'ro')
plt.plot(305.59,971.10,'ro')
plt.plot(309.16,1380.37,'ro')
plt.plot(307.24,1321.15,'ro')
plt.plot(309.31,1365.12,'ro')
plt.plot(308.80,1162.07,'ro')



plt.legend(['06','07','08','09','10','11','12','13','14'])
plt.ylim([0,3000])
plt.xlim([290,320])
plt.grid()
plt.xlabel('Potential temp [K]')
plt.ylabel('Height [m]')