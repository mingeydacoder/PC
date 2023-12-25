import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from tqdm import tqdm
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import pandas as pd

rootgrp = nc.Dataset('/Users/chenyenlun/Documents/GPM/gpm/GPM.daily.Asia.2019.09.30.nc')
lat = rootgrp.variables['latitude'][:]
lon = rootgrp.variables['longitude'][:]
preci = rootgrp.variables['preci'][0,:,:]
print(preci)


box = [89.05, 160.95, -10.95, 55.95]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

cmap = ListedColormap(['#a0fffa','#00cdff','#0096ff','#0069ff',
'#329600','#32ff00','#ffff00','#ffc800',
'#ff9600','#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5', '#ffc8ff'])
level = [1,3,5,10,15,20,30,40,50,60,80,90,100,150,200,250]
norm = matplotlib.colors.BoundaryNorm(level, 15)



precif = ax.contourf(lon, lat, preci, cmap=cmap, norm=norm, levels=level, extend='max')
precibar = plt.colorbar(precif,ticks=[1,3,5,10,15,20,30,40,50,60,80,90,100,150,200,250])
plt.savefig("Precipitable water and Thickness¡.png")
ax.set_xticks(np.arange(89.05, 160.95, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-10.95, 55.95, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)