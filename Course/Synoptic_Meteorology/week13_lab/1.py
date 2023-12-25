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

a = 60
b = a+12

topo = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/TOPO.nc')
print(topo)

zz = np.loadtxt('fort.98', skiprows=188, usecols=(1,))

lat = topo.variables['lat'][62:1000]
lon = topo.variables['lon'][180:835]

to = topo.variables['TOPO'][62:1000,180:835]
index = to.astype(int)

#precipitation data

filelist = []
num = []

for i in range(a,b,1):
    num.append(f"{i:03}")

for i in range(12):
    name = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.C.Surface-000'+num[i]+'.nc'
    filelist.append(name)

preci_dict = {}

i = 0
for file_name in filelist:
    precipitation = nc.Dataset(file_name)
    preci_dict[i] = precipitation.variables['sprec'][0,62:1000,180:835]*300
    i = i+1

preci_dict = np.array(list(preci_dict.values()))

sum0 = np.sum(preci_dict[:,:,:], axis=0)

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

#u_dict = np.array(list(u_dict.values()))
#v_dict = np.array(list(v_dict.values()))


print(u_dict.shape)

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
ax.quiver(lon[::30], lat[::30], uwind_index[::30,::30], vwind_index[::30,::30])

a = ax.contourf(lon, lat, sum0 , levels=[1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300], colors=['#a0fffa','#00cdff','#0096ff', '#0069ff','#329600','#32ff00',
'#ffff00','#ffc800','#ff9600',
'#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5',
'#ff64ff', '#ffc8ff'], extend='max')


#ax.grid()
ax.set_title("", fontsize=16)
ax.set_xlabel('Lontitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('taiwanvvmL_20160107_SHL       05LT', fontsize=18)

divider = make_axes_locatable(ax) 
colorbar_axes = divider.append_axes("right", size="3%", pad=0.1) 

eletic = [1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300]
cb = plt.colorbar(a, cax=colorbar_axes, ticks=eletic)

plt.savefig('img-005.png')





