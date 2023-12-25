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

for i in range(0,180,1):
    num.append(f"{i:03}")

for i in range(180):
    name = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.C.Surface-000'+num[i]+'.nc'
    filelist.append(name)

preci_dict = {}

i = 0
for file_name in filelist:
    precipitation = nc.Dataset(file_name)
    preci_dict[i] = precipitation.variables['sprec'][0,62:1000,180:835]*300
    i = i+1

preci_dict = np.array(list(preci_dict.values()))
precipitation_by_hour = np.empty([15,938,655])

k = 0
for k in range(15):
    precipitation_by_hour[k,:,:] = np.sum(preci_dict[11*k:11*k+11,:,:], axis=0)
    k += 1

peak_time_map = np.empty([938,655])

for x in range(938):
    for y in range(655):
        peak_time_map[x,y] = np.argmax(precipitation_by_hour[:,x,y], axis=0) - 1

# create a figure using subplots
fig, ax = plt.subplots(1,1,figsize=[10,10],dpi=300)
# set aspect ratio
ax.set_aspect(1)
# draw
ax.contour(lon, lat, to, levels=0, linewidths=1.5, colors='k', zorder=2)
ax.contour(lon, lat, to, levels=[200,1000,2500], linewidths=1, colors='k', zorder=3)
a = ax.contourf(lon, lat, peak_time_map, levels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], colors=['#0000c8', '#0000f6', '#0014ff', '#0068ff', '#0094ff', '#02e8f4',
'#26ffd1', '#6aff8d', '#8dff6a', '#f4f802', '#ffab00','#ff8200', '#ff3400', '#f60b00', '#960000'], zorder=1)


#ax.grid()
ax.set_title("", fontsize=16)
ax.set_xlabel('Lontitude', fontsize=14)
ax.set_ylabel('Latitude', fontsize=14)
ax.set_title('taiwanvvmL_20160107_SHL       05LT', fontsize=18)

divider = make_axes_locatable(ax) 
colorbar_axes = divider.append_axes("right", size="3%", pad=0.1) 

eletic = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
cb = plt.colorbar(a, cax=colorbar_axes, ticks=eletic)

plt.savefig('rain_peak.png')





