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

----------------------------------------------------

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter


def calc_qs(T_input, P_input):
    Lv, Rv = 2.5e6, 461.
    es = 611 * np.exp(Lv / Rv * (1 / 273 - 1 / T_input))
    qs_out = 0.622 * es / (P_input - 0.378 * es)
    return qs_out


def calc_distance(lat1, lon1, lat2, lon2):
    R = 6378137. # m
    dLat = lat2 * np.pi / 180 - lat1 * np.pi / 180
    dLon = lon2 * np.pi / 180 - lon1 * np.pi / 180
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d


########################## read file & data processing ##########################
# read event date
root = "/home/teachers/fortran_ta/data/LSM2021/weather/"
event = pd.read_csv(root + '2016_2018_weather_event.csv')
NE = np.asarray(event['NE'])
SNE = np.asarray(event['SNE'])
date = np.asarray(event['yyyymmdd'])

NE_date = np.where(NE == 1, date, 0)
SNE_date = np.where(SNE == 1, date, 0) # actually if it's SNE, it will be NE


# read ERA5, GPM, and mean variables
# ERA5
tmp = xr.open_dataset(root + "ERA5/U/2016/ERA5_U_2016_01_01.nc")
lonERA, latERA, levERA = np.asarray(tmp['lon']), np.asarray(tmp['lat']), np.asarray(tmp['plev'])
uSum = np.zeros(np.asarray(tmp['ua'])[0].shape)                             # construct empty array
vSum, TSum, qSum, wSum, zSum = uSum.copy(), uSum.copy(), uSum.copy(), uSum.copy(), uSum.copy() # construct empty array
MSLSum = np.zeros(np.asarray(xr.open_dataset(root + "ERA5/MSL/2016/ERA5_MSL_2016_01_01.nc")['psl'])[0].shape) # construct empty array
del tmp

# GPM
tmp = xr.open_dataset(root + "dailyIMERG/2016/GPM.daily.Asia.2016.01.01.nc")
lonGPM, latGPM = np.asarray(tmp['longitude']), np.asarray(tmp['latitude'])
preciSum = np.zeros(np.asarray(tmp['preci'])[0].shape)


# loop to sum and mean the variables
nonZeroCount = 0
for day in NE_date:
    if day != 0:
        # ERA5 variables
        uSum += np.asarray(xr.open_dataset(root + "ERA5/U/" +
              f"{str(day)[:4]}/ERA5_U_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['ua'])[0]
        vSum += np.asarray(xr.open_dataset(root + "ERA5/V/" +
                f"{str(day)[:4]}/ERA5_V_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['va'])[0]
        TSum += np.asarray(xr.open_dataset(root + "ERA5/T/" +
                f"{str(day)[:4]}/ERA5_T_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['ta'])[0]
        qSum += np.asarray(xr.open_dataset(root + "ERA5/Q/" +
                f"{str(day)[:4]}/ERA5_Q_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['hus'])[0]
        wSum += np.asarray(xr.open_dataset(root + "ERA5/W/" +
                f"{str(day)[:4]}/ERA5_W_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['w_NON_CDM'])[0]
        zSum += np.asarray(xr.open_dataset(root + "ERA5/Z/" +
                f"{str(day)[:4]}/ERA5_Z_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['geopotential'])[0]
        MSLSum += np.asarray(xr.open_dataset(root + "ERA5/MSL/" +
                f"{str(day)[:4]}/ERA5_MSL_{str(day)[:4]}_{str(day)[4:6]}_{str(day)[6:8]}.nc")['psl'])[0]

        # GPM variable
        preciSum += np.asarray(xr.open_dataset(root + "dailyIMERG/" +
                f"{str(day)[:4]}/GPM.daily.Asia.{str(day)[:4]}.{str(day)[4:6]}.{str(day)[6:8]}.nc")['preci'])[0]

        nonZeroCount += 1

uMean, vMean, TMean = uSum / nonZeroCount, vSum / nonZeroCount, TSum / nonZeroCount
qMean, wMean, zMean = qSum / nonZeroCount, wSum / nonZeroCount, zSum / nonZeroCount
MSLMean, preciMean = MSLSum / nonZeroCount, preciSum / nonZeroCount


# Topography filter
# ERA5
levERA3d = np.swapaxes(np.broadcast_to(levERA, (len(lonERA), len(latERA), len(levERA))), 0, 2)
# original lon: 60~180, lat:-20~60, we want lon: 90~160, lat: -10~55. resolution: 0.25 degree
aa = np.asarray(xr.open_dataset(root + "ERA5/era5_average_sp_2001_2019.nc")['lon'])
climatologicMSLP = np.asarray(xr.open_dataset(root + "ERA5/era5_average_sp_2001_2019.nc")['asp'])[40:-20, 120:-80]
topo_filter = levERA3d > climatologicMSLP
uMean[topo_filter], vMean[topo_filter], TMean[topo_filter] = np.nan, np.nan, np.nan
qMean[topo_filter], wMean[topo_filter], zMean[topo_filter] = np.nan, np.nan, np.nan

'''
########################## output result & read result ##########################
# output
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/uMean.npy", uMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/vMean.npy", vMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/TMean.npy", TMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/qMean.npy", qMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/wMean.npy", wMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/zMean.npy", zMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/MSLMean.npy", MSLMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/preciMean.npy", preciMean.flatten())
np.save("/home/B08/b08209006/synoptic_meteorology/wk9/output/shape.npy", np.concatenate((uMean.shape, vMean.shape, TMean.shape, qMean.shape,
                                                                                    wMean.shape, zMean.shape, MSLMean.shape, preciMean.shape)))


# read mean file
shape = np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/shape.npy")
uMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/uMean.npy"), (shape[0], shape[1], shape[2]))
vMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/vMean.npy"), (shape[3], shape[4], shape[5]))
TMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/TMean.npy"), (shape[6], shape[7], shape[8]))
qMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/qMean.npy"), (shape[9], shape[10], shape[11]))
wMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/wMean.npy"), (shape[12], shape[13], shape[14]))
zMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/zMean.npy"), (shape[15], shape[16], shape[17]))
MSLMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/MSLMean.npy"), (shape[18], shape[19]))
preciMean = np.reshape(np.load("/home/B08/b08209006/synoptic_meteorology/wk9/output/preciMean.npy"), (shape[20], shape[21]))
'''
########################## Calculate variables ##########################
# Graph2: Precipitable water and Thickness
# 925 hPa geopotential height = geopotential / g
# plev = [100000.  92500.  90000.  85000.  80000.  70000.  60000.  50000.  40000. 30000.  20000.  10000.]
g = 9.81
zHeight = zMean[1] / g
thickness = zMean[7] / g - zMean[0] / g

# precipitable water
PW = np.zeros([qMean.shape[1], qMean.shape[2]])
for k in range(11):
    PW += (-1 / g) * (qMean[k] + qMean[k+1]) * (levERA[k+1] - levERA[k]) * 0.5


# Graph3: Humidity and Convergence
qs = calc_qs(TMean[5], 70000.)
RH = qMean[5] / qs * 100


R = 6378137
R_lat = R * np.cos(np.radians(latERA))
dx = R_lat * np.radians(0.25)
dy = R * np.radians(0.25)

length = 2 * np.pi * R * np.cos(np.radians(latERA))
dx = length / (360. / (lonERA[-1] - lonERA[-2]))
dy = 2 * np.pi * R / (360. / (latERA[-1] - latERA[-2]))

# dy = calc_distance(latERA[:-2], lonERA[0], latERA[2:], lonERA[0])
# dy = np.swapaxes(np.broadcast_to(dy, (279, 259)), 0, 1)
# dx = []
# for i in range(len(latERA)):
#     dx.append(calc_distance(latERA[i], lonERA[0], latERA[i], lonERA[2]))
# dx = np.asarray(dx)[1:-1]
# dx = np.swapaxes(np.broadcast_to(dx, (279, len(dx))), 0, 1)


# du = (uMean[5, :, 2:] - uMean[5, :, :-2])[1:-1, :]
# dv = (vMean[5, 2:, :] - vMean[5, :-2, :])[:, 1:-1]

# divergence = du / dx + dv / dy
# divergence[divergence > -3e-6] = np.nan
# divergence[divergence <= -3e-6] = 1


def diff_4th(x, h):
    y_prime = (x[:-4] - 8*x[1:-3] + 8*x[3:-1] - x[4:]) / (12 * h)

    y_prime = np.insert(y_prime, 0, (-25 * x[1] + 48 * x[2] - 36 * x[3] + 16 * x[4] - 3 * x[5]) / (12 * h))
    y_prime = np.insert(y_prime, 0, (-25 * x[0] + 48 * x[1] - 36 * x[2] + 16 * x[3] - 3 * x[4]) / (12 * h))

    y_prime = np.append(y_prime, (25 * x[-2] - 48 * x[-3] + 36 * x[-4] - 16 * x[-5] + 3 * x[-6]) / (12 * h))
    y_prime = np.append(y_prime, (25 * x[-1] - 48 * x[-2] + 36 * x[-3] - 16 * x[-4] + 3 * x[-5]) / (12 * h))
    return y_prime


du_dx = np.zeros([261, 281])
for i in range(len(dx)):
    du_dx[i, :] = diff_4th(uMean[5, i, :], dx[i])

dv_dy = np.zeros([261, 281])
for i in range(len(lonERA)):
    dv_dy[:, i] = diff_4th(vMean[5, :, i], dy)


divergence = du_dx + dv_dy

divergence[divergence > -3e-6] = np.nan
divergence[divergence <= -3e-6] = 1

nonZeroCount = 240
########################## Plot ##########################
# Graph1: Surface weather chart and Precipitation
colormap = ['#a0fffa', '#00cdff', '#0096ff', '#0069ff', '#329600', '#32ff00', '#ffff00',
            '#ffc800', '#ff9600', '#ff0000', '#c80000', '#a00000', '#96009b','#c800d2',
            '#ff00f5', '#ff64ff', '#ffc8ff']
cmap = matplotlib.colors.ListedColormap(colormap)
clevel = [0.5, 1., 3., 5., 7., 9., 11., 13., 15., 20., 25., 30., 35., 40., 45., 50., 55.]
norm = matplotlib.colors.BoundaryNorm(clevel, 16)

box = [90, 160, 5, 55]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.6)

# shading: precipitation
ctf = ax.contourf(lonGPM, latGPM, preciMean, cmap=cmap, norm=norm, levels=clevel, extend='max')
cb = plt.colorbar(ctf, fraction=0.034, ticks=clevel)

# contour: surface pressure
ct = ax.contour(lonERA, latERA, MSLMean / 100, colors='k')
ax.clabel(ct, inline=1, fontsize=10)

# barb: 1000hPa wind [m/s to knot]
ax.barbs(lonERA[::5], latERA[::5], uMean[0, ::5, ::5] * 1.944, vMean[0, ::5, ::5] * 1.944, length=5, barbcolor='white', flagcolor='white', linewidth=0.5)

# labels
plt.text(90, 57, "NE ERA5 1000mb", fontsize=13)
plt.text(90, 55.5, "MSLP[hPa], 1000mb Wind[knot], GPM rain[mm/day]", fontsize=13)
plt.text(155, 59, f"[ {nonZeroCount} ]", fontsize=13)
plt.text(147.5, 57, "from 2016-01-01", fontsize=13)
plt.text(147.5, 55.5, "to     2018-12-31", fontsize=13)
plt.text(162, 56.5, "[mm/day]", fontsize=13)

# ticks setting
ax.set_xticks(np.arange(90, 160+10, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 55+10, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.savefig("../graph/Surface weather chart and Precipitation.png")
# plt.show()
plt.close()


# Graph2: Precipitable water and Thickness
colormap = ['#FFEE99','#FFCC65','#FF9932','#F5691C', '#FC3D3D','#D60C1E']
cmap = matplotlib.colors.ListedColormap(colormap)
clevel = [40, 50, 60, 70, 80, 90]
norm = matplotlib.colors.BoundaryNorm(clevel, 5)

box = [90, 160, 5, 55]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8)

# shading: precipitable water
ctf = ax.contourf(lonERA, latERA, PW, cmap=cmap, norm=norm, levels=clevel, extend='max')
cb = plt.colorbar(ctf, fraction=0.034, ticks=clevel)

# contour: 925 geopotential height
ct1 = ax.contour(lonERA, latERA, zHeight, colors='k')
ax.clabel(ct1, inline=1, fontsize=10)

# contour: 500~1000 thickness
ct2 = ax.contour(lonERA, latERA, thickness, colors='#941A80')
ax.clabel(ct2, inline=1, fontsize=10)

# barb: 925hPa wind [m/s to knot]
ax.barbs(lonERA[::5], latERA[::5], uMean[1, ::5, ::5] * 1.944, vMean[1, ::5, ::5] * 1.944, length=5, barbcolor='#6067C8', flagcolor='#6067C8', linewidth=0.5)

# labels
plt.text(90, 59, f"NE ERA5 925mb", fontsize=13)
plt.text(90, 57.3, "ERA5 Height[gpm], Wind[knot]", fontsize=13)
plt.text(90, 55.5, "1000-500 Thick[gpm], P-wat" + r"[kg/$m^2$]", fontsize=13)
plt.text(155, 59, f"[ {nonZeroCount} ]", fontsize=13)
plt.text(147.5, 57.3, "from 2016-01-01", fontsize=13)
plt.text(147.5, 55.5, "to     2018-12-31", fontsize=13)
plt.text(162, 56.5, r"[kg/$m^2$]", fontsize=13)

# ticks setting
ax.set_xticks(np.arange(90, 160+10, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 55+10, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.savefig("../graph/Precipitable water and Thickness.png")
# plt.show()
plt.close()



# Graph3: Humidity and Convergence
colormap = ['#96FFFF','#0797FA', '#0166FF']
cmap = matplotlib.colors.ListedColormap(colormap)
clevel = [70, 80, 90]
norm = matplotlib.colors.BoundaryNorm(clevel, 2)

box = [90, 160, 5, 55]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8)

# shading: RH
ctf = ax.contourf(lonERA, latERA, RH, cmap=cmap, norm=norm, levels=clevel, extend='max')
cb = plt.colorbar(ctf, fraction=0.034, ticks=clevel)

# scatter: divergence
xx, yy = np.meshgrid(lonERA, latERA)
ax.scatter(xx[divergence == 1], yy[divergence == 1], 0.1, color='#B2E55C')

# streamplot:
ax.streamplot(lonERA, latERA, uMean[5], vMean[5], density=2, color='k')

# labels
plt.text(90, 57.3, "NE ERA5 700mb", fontsize=13)
plt.text(90, 55.5, "Stream Line, RH[%], Divergence" + r"[$\leq -3\times 10^{-6} s^{-1}$]", fontsize=13)
plt.text(155, 59, f"[ {nonZeroCount} ]", fontsize=13)
plt.text(147.5, 57.3, "from 2016-01-01", fontsize=13)
plt.text(147.5, 55.5, "to     2018-12-31", fontsize=13)
plt.text(162, 56.5, "RH [%]", fontsize=13)

# ticks setting
ax.set_xticks(np.arange(90, 160+10, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 55+10, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.savefig("../graph/Humidity and Convergence.png")
# plt.show()
plt.close()