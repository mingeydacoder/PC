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




rootgrp = nc.Dataset('ERA5_T_2016_01_07.nc')
MSLP = nc.Dataset('ERA5_MSL_2016_01_07.nc')
U = nc.Dataset('ERA5_U_2016_01_07.nc')
V = nc.Dataset('ERA5_V_2016_01_07.nc')
precipitation = nc.Dataset('GPM.daily.Asia.2016.01.07.nc')
print(precipitation)


lat = rootgrp.variables['lat'][:]
lon = rootgrp.variables['lon'][:]
plev = rootgrp.variables['plev'][:]

t = rootgrp.variables['ta'][0,:,:,:] # first time step (0), all lat/lon (:)
print(t.shape)

psl = MSLP.variables['psl'][0,:,:]/100
u = U.variables['ua'][0,:,:,:]
v = V.variables['va'][0,:,:,:]
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


#plot Mean Sea Level Pressure and Precipitation
pr=plt.contourf(Plon_d,Plat_d,preci,extend='max',levels=[1,3,5,10,15,20,30,40,50,60,80,90,100,120,145,175],colors=['#a0fffa','#00cdff','#0096ff','#0069ff','#329600','#32ff00','#ffff00','#ffc800','#ff9600','#ff0000','#c80000','#a00000','#96009b','#c800d2','#ff00f5','#ffc8ff'])
p=plt.contour(lon,lat,psl,colors='black')
x = np.linspace(90,160,281)
x = x[::7]
y = np.linspace(-10,55,261)
y = y[::7]

for i in tqdm(range (261),colour='green',ncols=80):
   sleep(0.001)
   for j in range (281):
      if psl[i,j]<=1000:
        u[0,i,j]=np.nan
        v[0,i,j]=np.nan

u0 = u[0,::7,::7]
v0 = v[0,::7,::7]

X,Y = np.meshgrid(x,y)
wind=plt.barbs(X,Y,u0,v0,linewidth=0.15,length=4, barbcolor="black")
plt.clabel(p, inline=1, fmt='%1.1f', fontsize=9)
plt.colorbar(pr,ticks=[1,3,5,10,15,20,30,40,50,60,80,90,100,120,145,175])
plt.title("Mean Sea Level Pressure and Precipitation on 2016/01/07", fontsize=10)
plt.text(163, 57.5, "[mm/day]", fontsize=10)
plt.show()


#plot Precipitable water and Thickness

Z = nc.Dataset('ERA5_Z_2016_01_07.nc')
Q = nc.Dataset('ERA5_Q_2016_01_07.nc')
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
clevel2 = [40, 50, 60, 70, 80, 90]
norm = matplotlib.colors.BoundaryNorm(clevel2, 5)

box = [90, 160, -10, 55]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

precif = ax.contourf(lon, lat, pw, cmap=cmap2, norm=norm, levels=clevel2, extend='max')
precibar = plt.colorbar(precif, fraction=0.034, ticks=clevel2)

gh = ax.contour(lon, lat, zHeight, colors='k')
ax.clabel(gh, inline=1, fontsize=10)

print(zHeight.shape)

thick = ax.contour(lon, lat, thickness, colors='#941A80')
ax.clabel(thick, inline=1, fontsize=10)



u1 = u[1,::7,::7]
v1 = v[1,::7,::7]

ax.barbs(X,Y,u1,v1,linewidth=0.2,length=5, barbcolor="blue")


ax.set_xticks(np.arange(90, 160, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-10, 55, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.text(90, 59.7, "2016.01.07 ERA5 reanalysis", fontsize=13)
plt.text(90, 57.5, "PW [kg/m$^2$](color shading), 925mb Wind [kt] (blue barbs) ", fontsize=13)
plt.text(90, 55.5, "925mb Z [m] (black contours), 1000-500mb Depth [m] purple contours", fontsize=13)
plt.savefig("Precipitable water and Thickness¡.png")

'''
#plot Relative Humidity and Convergence

def calc_qs(T_input, P_input):
    Lv, Rv = 2.5e6, 461.
    es = 611 * np.exp(Lv / Rv * (1 / 273 - 1 / T_input))
    qs_out = 0.622 * es / (P_input - 0.378 * es)
    return qs_out

R = 6378137
R_lat = R * np.cos(np.radians(lat))
dx = R_lat * np.radians(0.25)
dy = R * np.radians(0.25)

length = 2 * np.pi * R * np.cos(np.radians(lat))
dx = length / (360. / (lon[-1] - lon[-2]))
dy = 2 * np.pi * R / (360. / (lat[-1] - lat[-2]))

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
for i in range(len(lon)):
    dv_dy[:, i] = diff_4th(v[5, :, i], dy)

divergence = du_dx + dv_dy

divergence[divergence > -3e-5] = np.nan
divergence[divergence <= -3e-5] = 1

qs = calc_qs(t[5], 70000.)
RH = q[5] / qs * 100
print(RH)

box = [90, 160, -10, 55]
fig, ax = plt.subplots(figsize = (12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent(box, crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', alpha=0.8, color='k', linewidth=0.5)

colormap3 = ['#96FFFF','#0797FA', '#0166FF']
cmap3 = matplotlib.colors.ListedColormap(colormap3)
clevel3 = [70, 80, 90]
norm3 = matplotlib.colors.BoundaryNorm(clevel3, 2)

#plot relative humidity
rhf = ax.contourf(lon, lat, RH, cmap=cmap3, levels=clevel3, norm=norm3, extend='max')
cb = plt.colorbar(rhf, fraction=0.034, ticks=clevel3)

#plot streamline
ax.streamplot(lon, lat, u[5], v[5], density=2, color='k', linewidth=0.5)

#plot divergence
xx, yy = np.meshgrid(lon, lat)
ax.scatter(xx[divergence == 1], yy[divergence == 1], 0.2, color='#B2E55C')


plt.text(90, 57.3, "700mb Wind (black streamlines)", fontsize=13)
plt.text(90, 55.5, "700mb Divergence" + r"[$\leq -3\times 10^{-6} s^{-1}$]" + "(green hatches)", fontsize=13)
plt.text(90, 59.1, "700mb RH[%] (color shading)", fontsize=13)
plt.text(90, 60.9, "2016.01.07 ERA5 reanalysis", fontsize=13)
plt.text(163, 56.5, "RH [%]", fontsize=13)

ax.set_xticks(np.arange(90, 160, 10),crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-10, 55, 10),crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=False)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
plt.savefig("Humidity and Convergence.png")

#ax.set_title('Relative Humidity and Divergence on 2016/01/07',fontsize=16)
'''







