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
from windrose import WindroseAxes
from metpy.units import units
import metpy.calc as mpcalc
from matplotlib import cm

a = 169
b = a + 12

topo = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/TOPO.nc')
print(topo)

zz = np.loadtxt('fort.98', skiprows=188, usecols=(1,))

#121-122E, 24-25N
lat = topo.variables['lat'][574:799]
lon = topo.variables['lon'][512:768]

to = topo.variables['TOPO'][512:768,574:799]
index = to.astype(int)

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

uwind_index = np.zeros([256,225])
vwind_index = np.zeros([256,225])

wind = nc.Dataset(filelist1[0])
u_dict = wind.variables['u'][0,:,512:768,574:799]
v_dict = wind.variables['v'][0,:,512:768,574:799]

#u_dict = np.array(list(u_dict.values()))
#v_dict = np.array(list(v_dict.values()))


print(u_dict.shape)

for i in range(256):
    for j in range(225):
        to[i,j] = zz[index[i,j]]
        uwind_index[i,j] = u_dict[index[i,j]+1,i,j]
        vwind_index[i,j] = v_dict[index[i,j]+1,i,j]

u = uwind_index.ravel()
v = vwind_index.ravel()

wind_speed = mpcalc.wind_speed(u*units('m/s'),v*units('m/s'))
wind_direction = mpcalc.wind_direction(u*units('m/s'),v*units('m/s'))

ax = WindroseAxes.from_ax()
ax.bar(wind_direction.magnitude, wind_speed.magnitude, normed='True', opening=0.8, edgecolor="white", cmap=cm.winter, bins=np.arange(0, 14, 2))
ax.set_legend(title='Wind Speed [m/s]')
plt.title('14-15 LT', fontsize = 24)
plt.savefig('windrose_3.png')