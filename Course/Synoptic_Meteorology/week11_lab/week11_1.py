import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from tqdm import tqdm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

topo = nc.Dataset('TOPO.nc')
print(topo)

zz = np.loadtxt('fort.98', skiprows=188, usecols=(1,))
precipitation = nc.Dataset('/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week10_lab/precipitation/taiwanvvmL_20160107_SHL.C.Surface-000088.nc')
print(precipitation)

lat = topo.variables['lat'][62:1000]
lon = topo.variables['lon'][180:835]

to = topo.variables['TOPO'][62:1000,180:835]
index = to.astype(int)

#precipitation data

filelist = []
num = []

for i in range(0,185,1):
    num.append(f"{i:03}")

for i in range(185):
    name = '/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week10_lab/precipitation/taiwanvvmL_20160107_SHL.C.Surface-000'+num[i]+'.nc'
    filelist.append(name)

preci_dict = {}

i = 0
for file_name in filelist:
    precipitation = nc.Dataset(file_name)
    preci_dict[i] = precipitation.variables['sprec'][0,62:1000,180:835]*300
    i = i+1

preci_dict = np.array(list(preci_dict.values()))
print(preci_dict.shape)

sum0 = np.sum(preci_dict[61:72,:,:], axis=0)



for i in tqdm(range(938), colour="green"):
    for j in range(655):
        to[i,j] = zz[index[i,j]]


# create a figure using subplots
fig, ax = plt.subplots(1,1,figsize=[10,10],dpi=300)
# set aspect ratio
ax.set_aspect(1)
# draw
ax.contour(lon, lat, to, levels=0, linewidths=1.5, colors='k')
ax.contour(lon, lat, to, levels=[200,1000,2500], linewidths=1, colors='k')
a = ax.contourf(lon,lat,sum0 , levels=[1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300], colors=['#a0fffa','#00cdff','#0096ff', '#0069ff','#329600','#32ff00',
'#ffff00','#ffc800','#ff9600',
'#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5',
'#ff64ff', '#ffc8ff'], extend='max')

#ax.grid()
ax.set_title("", fontsize=16)
ax.set_xlabel('Lontitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('taiwanvvmL_20160107_SHL', fontsize=18)

divider = make_axes_locatable(ax) 
colorbar_axes = divider.append_axes("right", size="3%", pad=0.1) 

eletic = [1,2,6,10,15,20,30,40,50,70,90,110,130,150,200,300]
cb = plt.colorbar(a, cax=colorbar_axes, ticks=eletic)





