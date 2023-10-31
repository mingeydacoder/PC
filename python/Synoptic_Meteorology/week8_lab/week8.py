import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
import datetime
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from tqdm import tqdm



rootgrp = nc.Dataset('ERA5_T_2016_01_07.nc')
MSLP = nc.Dataset('ERA5_MSL_2016_01_07.nc')
U = nc.Dataset('ERA5_U_2016_01_07.nc')
V = nc.Dataset('ERA5_V_2016_01_07.nc')
precipitation = nc.Dataset('GPM.daily.Asia.2016.01.07.nc')
print(precipitation)


lat = rootgrp.variables['lat'][:]
lon = rootgrp.variables['lon'][:]
plev = rootgrp.variables['plev'][:]

t = rootgrp.variables['ta'][0,0,:,:] # first time step (0), all lat/lon (:)
print(t.shape)

psl = MSLP.variables['psl'][0,:,:]/100
u = U.variables['ua'][0,0,:,:]
v = V.variables['va'][0,0,:,:]
preci = precipitation.variables['preci'][0,:,:]
#print(preci.shape)

PRECIlon = precipitation.variables['longitude'][:]
PRECIlat = precipitation.variables['latitude'][:]
Plon_d,Plat_d = np.meshgrid(PRECIlon, PRECIlat)



cmap = ListedColormap(['#a0fffa','#00cdff','#0096ff','#0069ff',
'#329600','#32ff00','#ffff00','#ffc800',
'#ff9600','#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5', '#ffc8ff'])
norm = plt.Normalize(1, 150)
c = np.linspace(0, 150, 12)

#CS=plt.contour(lon,lat,t,cmap=cmap,label='test',extend='both')
pr=plt.contourf(Plon_d,Plat_d,preci,extend='max',levels=[1,3,5,10,15,20,30,40,50,60,80,90,100,120,145,175],colors=['#a0fffa','#00cdff','#0096ff','#0069ff','#329600','#32ff00','#ffff00','#ffc800','#ff9600','#ff0000','#c80000','#a00000','#96009b','#c800d2','#ff00f5','#ffc8ff'])
p=plt.contour(lon,lat,psl,colors='black')
x = np.linspace(90,160,281)
x = x[::7]
y = np.linspace(-10,55,261)
y = y[::7]

for i in tqdm(range (261),colour='green'):
   for j in range (281):
      if psl[i,j]<=1000:
        u[i,j]=np.nan
        v[i,j]=np.nan

u = u[::7,::7]
v = v[::7,::7]

#print(u)
X,Y = np.meshgrid(x,y)
wind=plt.barbs(X,Y,u,v,linewidth=0.15,length=4, barbcolor="black")
plt.clabel(p,
           inline=1,
           fmt='%1.1f',
           fontsize=9)

plt.colorbar(pr,ticks=[1,3,5,10,15,20,30,40,50,60,80,90,100,120,145,175])
plt.show()
