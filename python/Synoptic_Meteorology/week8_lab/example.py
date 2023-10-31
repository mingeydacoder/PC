import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import datetime
from mpl_toolkits.basemap import Basemap

event = np.loadtxt('2016_2018_weather_event.csv', delimiter = ',', skiprows = 1)

date = np.asarray(event[:,0])
SWF = np.asarray(event[:,12])
SSWF = np.asarray(event[:,13])

WSWF_DATE = np.where(SWF - SSWF ==1, date, 0)

#print(WSWF_DATE)

surf=nc.Dataset("era5_average_sp_2001_2019.nc")
surf_pre=surf.variables['asp'][40:301,120:401]


rootgrp = nc.Dataset('ERA5/U/2016/ERA5_U_2016_01_01.nc')

#print(rootgrp.variables['ua'][:])
usum = np.zeros((261,281))
#print(usum.shape)

vsum, tsum, qsum, zsum, wsum = usum.copy(), usum.copy(), usum.copy(), usum.copy, usum.copy

rootgrp2 = nc.Dataset('ERA5/MSL/2016/ERA5_MSL_2016_01_01.nc')
MSLsum = np.zeros((261,281))
num=np.full([670,720],207)

#print(MSLsum.shape)

rootgrp3 = nc.Dataset('dailyIMERG/2016/GPM.daily.Asia.2016.01.01.nc')
GMPlat = rootgrp3.variables['latitude'][:]
GMPlon = rootgrp3.variables['longitude'][:]
#print(GMPlat.shape, GMPlon.shape)

PRECIsum = np.zeros((670,720))
#print(precisum)

WSWFD = np.unique(WSWF_DATE)
WSWFdate = WSWFD[WSWFD !=0]
#print(WSWFdate.shape)

#print(str(WSWFdate[0])[:4],str(WSWFdate[0])[4:6],str(WSWFdate[0])[6:8])

for i in range(207):
    U = nc.Dataset('ERA5/U/'+str(WSWFdate[i])[:4]+'/ERA5_U_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    u = U.variables['ua'][0,0,:,:]
    usum = usum + np.array(u)
    V = nc.Dataset('ERA5/V/'+str(WSWFdate[i])[:4]+'/ERA5_V_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    v = V.variables['va'][0,0,:,:]
    vsum = vsum + np.array(v)
    T = nc.Dataset('ERA5/T/'+str(WSWFdate[i])[:4]+'/ERA5_T_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    t = T.variables['ta'][0]
    tsum = tsum + np.array(t)
    Q = nc.Dataset('ERA5/Q/'+str(WSWFdate[i])[:4]+'/ERA5_Q_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    q = Q.variables['hus'][0]
    qsum = qsum + np.array(q)
    Z = nc.Dataset('ERA5/Z/'+str(WSWFdate[i])[:4]+'/ERA5_Z_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    z = Z.variables['geopotential'][0,:,:,:]
    zsum = zsum + np.asarray(z)
    MSL = nc.Dataset('ERA5/MSL/'+str(WSWFdate[i])[:4]+'/ERA5_MSL_'+str(WSWFdate[i])[:4]+'_'+str(WSWFdate[i])[4:6]+'_'+str(WSWFdate[i])[6:8]+'.nc')
    msl = MSL.variables['psl'][0,:,:]
    MSLsum = MSLsum + np.array(msl)
    PRECI = nc.Dataset('dailyIMERG/'+str(WSWFdate[i])[:4]+'/GPM.daily.Asia.'+str(WSWFdate[i])[:4]+'.'+str(WSWFdate[i])[4:6]+'.'+str(WSWFdate[i])[6:8]+'.nc')
    preci= PRECI.variables['preci'][0,:,:]
    PRECIsum = PRECIsum + np.array(preci)

umean, vmean, tmean, qmean, zmeam = usum/207*1.94  , vsum/207*1.94  , t/207, q/207, z/207
MSLmean = MSLsum/207
PRECImean = PRECIsum/num



MSLlat = MSL.variables['lat'][:]
MSLlon = MSL.variables['lon'][:]

PRECIlon = PRECI.variables['longitude'][:]
PRECIlat = PRECI.variables['latitude'][:]

Ulat = MSLlat[::6]
Ulon = MSLlon[::6]

MSLlon_d,MSLlat_d = np.meshgrid(MSLlon, MSLlat)
Ulon_d,Ulat_d = np.meshgrid(Ulon, Ulat)
Plon_d,Plat_d = np.meshgrid(PRECIlon, PRECIlat)

fig, ax = plt.subplots(figsize = (12, 8))
m = Basemap(llcrnrlat = 5, llcrnrlon=90, urcrnrlat = 55, urcrnrlon=160)
m.drawcoastlines(color = 'grey')

Pr=m.contourf(Plon_d,Plat_d,PRECImean,extend='max',levels=[0.5,1.0,3,5,7,9,11,13,15,20,25,30,35,40,45,50,55],colors=['#a0fffa','#00cdff','#0096ff','#0069ff','#329600','#32ff00','#ffff00','#ffc800','#ff9600','#ff0000','#c80000','#a00000','#96009b','#c800d2','#ff00f5','#ff64ff','#ffc8ff'])
plt.colorbar(Pr,orientation='vertical',extend='max',ticks=[0.5,1.0,3,5,7,9,11,13,15,20,25,30,35,40,45,50,55]).set_label('[mm/day]', labelpad=-40,y=1.05,rotation =0)



ct = ax.contour(MSLlon_d, MSLlat_d, MSLmean // 100, colors='k')
ax.clabel(ct, inline=1, fontsize=10)

for i in range (261):
   for j in range (281):
      if surf_pre[i,j]<=100000:
        umean[i,j]=np.nan
        vmean[i,j]=np.nan
umean=umean[::6,::6]
vmean=vmean[::6,::6]

wind = m.barbs(Ulon_d,Ulat_d,umean,vmean,length=4,barbcolor='w')
parallels = np.arange(5,65,10)
meridians = np.arange(90,170,10)
m.drawparallels(parallels,color='none',labels=[1,0,0,0],fontsize=10)
m.drawmeridians(meridians,color='none',labels=[0,0,0,1],fontsize=10)

plt.text(90, 57, "WSWF ERA5 1000mb", fontsize=13)
plt.text(90, 55.5, "MSLP[hPa], 1000mb Wind[knot], GPM rain[mm/day]", fontsize=13)
plt.text(145, 57, "from 2016-01-01", fontsize=13)
plt.text(145, 55.5, "to 2018-12-31", fontsize=13)


plt.savefig('Surface weather chart and Precipitation.png')
#plt.show()
plt.close

#2


lonERA, latERA, levERA = np.asarray(rootgrp['lon']), np.asarray(rootgrp['lat']), np.asarray(rootgrp['plev'])

g = 9.8
zHeight = zmean[1] / g
thickness = zmean[7] / g - zmean[0] / g
PW = np.zeros([qmean.shape[1], qmean.shape[2]])
for k in range(11):
    PW += (-1 / g) * (qmean[k] + qmean[k+1]) * (levERA[k+1] - levERA[k]) * 0.5

fig, ax = plt.subplots(figsize = (12, 8))
m=Basemap(llcrnrlat=5,llcrnrlon=90,urcrnrlat=55,urcrnrlon=160)
m.drawcoastlines(color='grey')

ctf = ax.contourf(lonERA, latERA, PW, cmap=cmap, norm=norm, levels=clevel, extend='max')
cb = plt.colorbar(ctf, fraction=0.034, ticks=clevel)


ct1 = ax.contour(lonERA, latERA, zHeight, colors='k')
ax.clabel(ct1, inline=1, fontsize=10)

ct2 = ax.contour(lonERA, latERA, thickness, colors='#941A80')
ax.clabel(ct2, inline=1, fontsize=10)

ax.barbs(lonERA[::5], latERA[::5], umean[1, ::5, ::5], vmean[1, ::5, ::5], length=5, barbcolor='#6067C8', flagcolor='#6067C8', linewidth=0.5)

plt.show()