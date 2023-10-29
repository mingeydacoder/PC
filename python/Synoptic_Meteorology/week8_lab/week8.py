import numpy as np 
import netCDF4 as nc
import matplotlib.pyplot as plt
import datetime
import basemap


rootgrp = nc.Dataset('ERA5_T_2016_01_07.nc')
MSLP = nc.Dataset('ERA5_MSL_2016_01_07.nc')

lat = rootgrp.variables['lat'][:]
lon = rootgrp.variables['lon'][:]
plev = rootgrp.variables['plev'][:]


print(MSLP)

# find and plot the filled contour of global temperature at first time step at 850 hPa level
t = rootgrp.variables['ta'][0,0,:,:] # first time step (0), all lat/lon (:)
print(t.shape)

psl = MSLP.variables['psl'][0,:,:]

CS=plt.contour(lon,lat,t)
a=plt.contour(lon,lat,psl,colors='black')
plt.colorbar(CS,orientation='vertical')
plt.title('T at 850 hPa')
plt.show()