import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
#from tqdm import tqdm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

a = 169
b = a+12

topo = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/TOPO.nc')
print(topo)

zz = np.loadtxt('fort.98', skiprows=188, usecols=(1,))
qvto = np.loadtxt('fort1.98', skiprows=3, usecols=(5,))

lat = topo.variables['lat'][62:1000]
lon = topo.variables['lon'][180:835]
to = topo.variables['TOPO'][62:1000,180:835]

index = to.astype(int)

#mixing ratio data

filelist = []
num = []

for i in range(a,b,1):
    num.append(f"{i:03}")

for i in range(12):
    name = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num[i]+'.nc'
    filelist.append(name)

'''
test = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000100.nc')
print(test)
'''

qv_dict = {}

i = 0
for file_name in filelist:
    qv = nc.Dataset(file_name)
    qv_dict[i] = qv.variables['qv'][0,:,62:1000,180:835]*1000
    i = i+1

qv_dict = np.array(list(qv_dict.values()))
ave = np.average(qv_dict, axis=0)

qvisland = np.zeros([938,655])

for i in range(938):
    for j in range(655):
        qvisland[i,j] = ave[index[i,j]+1,i,j]

#print(np.ndim(preci_dict))

#u,v wind data

filelist1 = []
num1 = []

for j in range(a,b,1):
    num1.append(f"{j:03}")

for j in range(12):
    name1 = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Dynamic-000'+num1[j]+'.nc'
    filelist1.append(name1)

u_dict = {}
v_dict = {}

uwind_index = np.zeros([938,655])
vwind_index = np.zeros([938,655])

wind = nc.Dataset(filelist1[0])
u_dict = wind.variables['u'][0,:,62:1000,180:835]
v_dict = wind.variables['v'][0,:,62:1000,180:835]

for i in range(938):
    for j in range(655):
        to[i,j] = zz[index[i,j]]
        uwind_index[i,j] = u_dict[index[i,j]+1,i,j]
        vwind_index[i,j] = v_dict[index[i,j]+1,i,j]

# create a figure using subplots
fig, ax = plt.subplots(1,1,figsize=[10,10],dpi=300)
# set aspect ratio
ax.set_aspect(1)
# draw
ax.contour(lon, lat, to, levels=0, linewidths=1.5, colors='k')
ax.contour(lon, lat, to, levels=[200,1000,2500], linewidths=1, colors='k')
#ax.contourf(lon, lat, qvisland*1000, levels=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], colors=['#fafdce', '#f0f9b7', '#ddf2b2', '#c6e9b4', '#9ed9b8', '#76cabc', '#53bdc1', '#36abc3', '#2296c1', '#1f7bb6',
#'#225da8', '#24459c', '#21308b', '#102369', '#081d58'])
a = ax.contourf(lon, lat, qvisland, levels=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], colors=['#fafdce', '#f0f9b7', '#ddf2b2', '#c6e9b4', '#9ed9b8', '#76cabc', '#53bdc1', '#36abc3', '#2296c1', '#1f7bb6',
'#225da8', '#24459c', '#21308b', '#102369', '#081d58'], extend='max')
ax.quiver(lon[::30], lat[::30], uwind_index[::30,::30], vwind_index[::30,::30])


#ax.grid()
ax.set_xlabel('Lontitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('taiwanvvmL_20160107_SHL       14LT', fontsize=18)

divider = make_axes_locatable(ax) 
colorbar_axes = divider.append_axes("right", size="3%", pad=0.1) 

eletic = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
cb = plt.colorbar(a, cax=colorbar_axes, ticks=eletic)

plt.savefig('9.png')





