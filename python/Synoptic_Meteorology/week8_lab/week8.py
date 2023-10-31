import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors



rootgrp = nc.Dataset('ERA5_T_2016_01_07.nc')
MSLP = nc.Dataset('ERA5_MSL_2016_01_07.nc')
U = nc.Dataset('ERA5_U_2016_01_07.nc')
V = nc.Dataset('ERA5_V_2016_01_07.nc')
print(U)


lat = rootgrp.variables['lat'][:]
lon = rootgrp.variables['lon'][:]
plev = rootgrp.variables['plev'][:]


# find and plot the filled contour of global temperature at first time step at 850 hPa level
t = rootgrp.variables['ta'][0,0,:,:] # first time step (0), all lat/lon (:)
print(t.shape)

psl = MSLP.variables['psl'][0,:,:]/100
u = U.variables['ua'][0,0,:,:]
v = V.variables['va'][0,0,:,:]
print(u.shape)

#print(lon)

cmap = ListedColormap(['#a0fffa','#00cdff','#0096ff','#0069ff',
'#329600','#32ff00','#ffff00','#ffc800',
'#ff9600','#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5','#ff64ff', '#ffc8ff'])
norm = plt.Normalize(1, 150)
c = np.linspace(0, 150, 12)

CS=plt.contour(lon,lat,t,cmap=cmap,label='test',extend='both')
a=plt.contour(lon,lat,psl,colors='black')
x = np.linspace(90,160,281)
x = x[::7]
y = np.linspace(-10,55,261)
y = y[::7]

for i in range (261):
   for j in range (281):
      if psl[i,j]<=1000:
        u[i,j]=np.nan
        v[i,j]=np.nan

u = u[::7,::7]
v = v[::7,::7]

print(u)
X,Y = np.meshgrid(x,y)
wind=plt.barbs(X,Y,u,v,linewidth=0.15,length=4, barbcolor="black")
plt.clabel(a,  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=9)

plt.colorbar(CS)
plt.show()
