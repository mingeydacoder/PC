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

rootgrp = nc.Dataset('/Users/chenyenlun/Documents/GPM/gpm/GPM.daily.Asia.2019.10.03.nc')
MSLP = nc.Dataset('/Users/chenyenlun/Documents/MSL/msl/ERA5_MSL_2019_09_23.nc')
U = nc.Dataset('/Users/chenyenlun/Documents/U/u/ERA5_U_2019_09_23.nc')
V = nc.Dataset('/Users/chenyenlun/Documents/V/v/ERA5_V_2019_09_23.nc')
T = nc.Dataset('/Users/chenyenlun/Documents/T/t/ERA5_T_2019_09_23.nc')
Q = nc.Dataset('/Users/chenyenlun/Documents/Q/q/ERA5_Q_2019_09_23.nc')
Z = nc.Dataset('/Users/chenyenlun/Documents/Z/z/ERA5_Z_2019_09_23.nc')


lat = rootgrp.variables['latitude'][:]
lon = rootgrp.variables['longitude'][:]
plev = T.variables['plev'][:]
elat = MSLP.variables['lat'][:]
elon = MSLP.variables['lon'][:]
preci = rootgrp.variables['preci'][0,:,:]
psl = MSLP.variables['psl'][0,:,:]/100
u = U.variables['ua'][0,:,:,:]
v = V.variables['va'][0,:,:,:]
t = T.variables['ta'][0,:,:,:]
z = Z.variables['geopotential'][0,:,:,:]
q = Q.variables['hus'][0,:,:,:]

box = [95, 150, 5, 40]
'''

#plot precipitation

plt.figure()
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=1)

cmap = ListedColormap(['#a0fffa','#00cdff','#0096ff','#0069ff',
'#329600','#32ff00','#ffff00','#ffc800',
'#ff9600','#ff0000','#c80000','#a00000',
'#96009b','#c800d2','#ff00f5', '#ffc8ff'])
level = [1,5,20,40,60,80,100,125,150,175,200,225,250,350,400]
norm = matplotlib.colors.BoundaryNorm(level, 15)



precif = ax.contourf(lon, lat, preci, cmap=cmap, norm=norm, levels=level, extend='max')
precibar = plt.colorbar(precif,ticks=level)
ax.set_xticks(np.arange(95, 150, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 40, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_title('Precipitation [mm/day] on 09/28')
plt.savefig('preci0928.png')


#plot pressure and wind

plt.figure()
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=1)

x = np.linspace(95,150,281)
x = x[::6]
y = np.linspace(5,40,261)
y = y[::6]
X,Y = np.meshgrid(x,y)
u0 = u[0,::6,::6]
v0 = v[0,::6,::6]

p=plt.contour(elon,elat,psl,colors='black')
plt.clabel(p, inline=1, fmt='%1.1f', fontsize=9)
wind=plt.barbs(X,Y,u0,v0,linewidth=0.15,length=4, barbcolor="black")

ax.set_xticks(np.arange(95, 150, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 40, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_title('Mean Sea Level Pressure and wind on 10/03')
plt.savefig('p_wind1003.png')


#plot Relative Humidity and Convergence

def calc_qs(T_input, P_input):
    Lv, Rv = 2.5e6, 461.
    es = 611 * np.exp(Lv / Rv * (1 / 273 - 1 / T_input))
    qs_out = 0.622 * es / (P_input - 0.378 * es)
    return qs_out

R = 6378137
R_lat = R * np.cos(np.radians(elat))
dx = R_lat * np.radians(0.25)
dy = R * np.radians(0.25)

length = 2 * np.pi * R * np.cos(np.radians(elat))
dx = length / (360. / (elon[-1] - elon[-2]))
dy = 2 * np.pi * R / (360. / (elat[-1] - elat[-2]))

def diff_4th(x, h):
    y_prime = (x[:-4] - 8*x[1:-3] + 8*x[3:-1] - x[4:]) / (12 * h)

    y_prime = np.insert(y_prime, 0, (-25 * x[1] + 48 * x[2] - 36 * x[3] + 16 * x[4] - 3 * x[5]) / (12 * h))
    y_prime = np.insert(y_prime, 0, (-25 * x[0] + 48 * x[1] - 36 * x[2] + 16 * x[3] - 3 * x[4]) / (12 * h))

    y_prime = np.append(y_prime, (25 * x[-2] - 48 * x[-3] + 36 * x[-4] - 16 * x[-5] + 3 * x[-6]) / (12 * h))
    y_prime = np.append(y_prime, (25 * x[-1] - 48 * x[-2] + 36 * x[-3] - 16 * x[-4] + 3 * x[-5]) / (12 * h))
    return y_prime

du_dx = np.zeros([261, 281])
for i in range(len(dx)):
    du_dx[i, :] = diff_4th(u[5, i, :], dx[i])

dv_dy = np.zeros([261, 281])
for i in range(len(elon)):
    dv_dy[:, i] = diff_4th(v[5, :, i], dy)

vorticity = du_dx + dv_dy

vorticity[vorticity > -3e-5] = np.nan
vorticity[vorticity <= -3e-5] = 1

qs = calc_qs(t[5], 70000.)
RH = q[5] / qs * 100

box = [95, 150, 5, 40]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

colormap3 = ['#96FFFF','#0797FA', '#0166FF']
cmap3 = matplotlib.colors.ListedColormap(colormap3)
clevel3 = [70, 80, 90]
norm3 = matplotlib.colors.BoundaryNorm(clevel3, 2)

#plot relative humidity
rhf = ax.contourf(elon, elat, RH, cmap=cmap3, levels=clevel3, norm=norm3, extend='max')
cb = plt.colorbar(rhf, fraction=0.034, ticks=clevel3)

#plot streamline
ax.streamplot(elon, elat, u[5], v[5], density=2, color='k', linewidth=0.5)

#plot vorticity
xx, yy = np.meshgrid(elon, elat)
ax.scatter(xx[vorticity == 1], yy[vorticity == 1], 0.2, color='#B2E55C')


ax.set_xticks(np.arange(95, 150, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 40, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_title('Relative humidity and Convergence on 10/03',fontsize=16)
plt.text(152, 43.7, "RH [%]", fontsize=13)
plt.savefig("relative humidity_convergence1003.png")



#plot precipitable water and thickness

g = 9.81

z = Z.variables['geopotential'][0,:,:,:]
q = Q.variables['hus'][0,:,:,:]

zHeight = z[1]/g # Geopotential Height at 925 hPa
thickness = z[7]/g - z[0]/g
pw = np.zeros([q.shape[1],q.shape[2]])
for k in tqdm(range(11),colour='blue',ncols=80):
    sleep(0.05)
    pw += (-1/g) * (q[k] + q[k+1]) * (plev[k+1] - plev[k])

colormap2 = ['#FFEE99','#FFCC65','#FF9932','#F5691C', '#FC3D3D','#D60C1E']
cmap2 = matplotlib.colors.ListedColormap(colormap2)
clevel2 = [50, 60, 70, 80, 90, 100]
norm = matplotlib.colors.BoundaryNorm(clevel2, 5)

box = [95, 150, 5, 40]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

precif = ax.contourf(elon, elat, pw, cmap=cmap2, norm=norm, levels=clevel2, extend='max')
precibar = plt.colorbar(precif, fraction=0.034, ticks=clevel2)

gh = ax.contour(elon, elat, zHeight, colors='k')
ax.clabel(gh, inline=1, fontsize=10)

thick = ax.contour(elon, elat, thickness, colors='#941A80')
ax.clabel(thick, inline=1, fontsize=10)


ax.set_xticks(np.arange(95, 150, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 40, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_title('Precipitable water [kg/m$^2$] and Thickness on 09/23')
plt.savefig('thickness0923.png')

'''

#plot relative vorticity

R = 6378137
R_lat = R * np.cos(np.radians(elat))
dx = R_lat * np.radians(0.25)
dy = R * np.radians(0.25)

length = 2 * np.pi * R * np.cos(np.radians(elat))
dx = length / (360. / (elon[-1] - elon[-2]))
dy = 2 * np.pi * R / (360. / (elat[-1] - elat[-2]))

def diff_4th(x, h):
    y_prime = (x[:-4] - 8*x[1:-3] + 8*x[3:-1] - x[4:]) / (12 * h)

    y_prime = np.insert(y_prime, 0, (-25 * x[1] + 48 * x[2] - 36 * x[3] + 16 * x[4] - 3 * x[5]) / (12 * h))
    y_prime = np.insert(y_prime, 0, (-25 * x[0] + 48 * x[1] - 36 * x[2] + 16 * x[3] - 3 * x[4]) / (12 * h))

    y_prime = np.append(y_prime, (25 * x[-2] - 48 * x[-3] + 36 * x[-4] - 16 * x[-5] + 3 * x[-6]) / (12 * h))
    y_prime = np.append(y_prime, (25 * x[-1] - 48 * x[-2] + 36 * x[-3] - 16 * x[-4] + 3 * x[-5]) / (12 * h))
    return y_prime

dv_dx = np.zeros([261, 281])
for i in range(len(dx)):
    dv_dx[i, :] = diff_4th(v[6, i, :], dx[i])

du_dy = np.zeros([261, 281])
for i in range(len(elon)):
    du_dy[:, i] = diff_4th(u[6, :, i], dy)

vorticity = dv_dx - du_dy
print(vorticity)


box = [95, 150, 5, 40]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

colormap3 = ['#FCEAA4','#FBDB82','#FDC45C','#FCA643','#F48830',
'#E76D25','#D0521C','#AF3F15','#8D3111','#772C10']
cmap3 = matplotlib.colors.ListedColormap(colormap3)
clevel3 = [1e-5, 3e-5, 5e-5, 8e-5, 11e-5 , 14e-5, 17e-5, 20e-5 ,30e-5, 40e-5]
norm3 = matplotlib.colors.BoundaryNorm(clevel3, 9)

vor = ax.contourf(elon, elat, vorticity, cmap=cmap3, norm=norm3, levels=clevel3, extend='max')
cb = plt.colorbar(vor, fraction=0.034, ticks=clevel3)

ax.set_xticks(np.arange(95, 150, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(5, 40, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_title('Relative vorticity [m$^-1$] on 09/23',fontsize=16)
plt.savefig("vorticity0923.png")

