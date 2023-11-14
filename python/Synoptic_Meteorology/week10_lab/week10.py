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

lat = topo.variables['lat'][:]
lon = topo.variables['lon'][:]

to = topo.variables['TOPO'][:,:]
index = to.astype(int)

preci = precipitation.variables['sprec'][0,:,:]*300
print(np.max(preci))



for i in tqdm(range(1024), colour="green"):
    for j in range(1024):
        to[i,j] = zz[index[i,j]]


# create a figure using subplots
fig, ax = plt.subplots(1,1,figsize=[10,10],dpi=300)
# set aspect ratio
ax.set_aspect(1)
# draw
ele = ax.contourf(lon, lat, to, linewidths=0.5, levels=np.linspace(0,3800,39), colors=
['#ffffff', '#f2f2f2', '#e9e9e9', '#e5e5e5',
'#dcdcdc', '#d8d8d8', '#cfcfcf', '#cbcbcb',
'#c2c2c2', '#bebebe',  '#b6b6b6', '#b1b1b1',
'#adadad', '#a0a0a0', '#9c9c9c',
'#979797', '#8a8a8a', '#868686',
'#7d7d7d', '#707070', '#6c6c6c',
'#686868', '#5b5b5b', '#565656',
'#525252', '#454545', '#414141',
'#3d3d3d', '#383838',  '#303030', '#2b2b2b',
'#272727', '#232323',  '#1a1a1a', '#161616',
'#111111', '#0d0d0d',  '#040404', '#000000']
)
ax.contour(lon, lat, to, levels=0)
ax.contourf(lon,lat, preci, levels=[1,3,5,10,15,20,30,40,50,60,80,90,100,120,145,175], colors=['#a0fffa','#00cdff','#0096ff',
'#0069ff','#329600','#32ff00',
'#ffff00','#ffc800','#ff9600',
'#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5',
'#ff64ff', '#ffc8ff'])
ax.grid()
ax.set_title("Topography in TaiwanVVM", fontsize=16)
ax.set_xlabel('Lontitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)

divider = make_axes_locatable(ax) 
colorbar_axes = divider.append_axes("right", size="5%", pad=0.1) 

eletic = np.linspace(0,3800,20)
cb = plt.colorbar(ele, cax=colorbar_axes, ticks=eletic)
cb.set_label(label='Elevation [m]',fontsize=14)







