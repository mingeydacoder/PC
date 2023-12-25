import numpy as np 
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
from time import sleep
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
from tqdm import tqdm
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
import matplotlib.gridspec as gridspec

topo = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/TOPO.nc')
print(topo)

zt = np.linspace(0,5900,60)
qvto = np.loadtxt('fort1.98', skiprows=66, usecols=(5,))
pressure = np.loadtxt('fort1.98', skiprows=66, usecols=(3,))/100
#pressure = units.Quantity(pressure, 'mbar')

lat = topo.variables['lat'][62:1000]
lon = topo.variables['lon'][180:835]
to = topo.variables['TOPO'][62:1000,180:835]

index = to.astype(int)

#mixing ratio data

x, y = 600, 750

num = f"{73:03}"
num2 = f"{97:03}"
num3 = f"{121:03}"
num4 = f"{145:03}"
num5 = f"{181:03}"


name = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num+'.nc'
name2 = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num2+'.nc'
name3 = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num3+'.nc'
name4 = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num4+'.nc'
name5 = '/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000'+num5+'.nc'



test = nc.Dataset('/home/teachers/fortran_ta/data/LSM2023/TaiwanVVM/winter/taiwanvvmL_20160107_SHL/archive/taiwanvvmL_20160107_SHL.L.Thermodynamic-000100.nc')
print(test)


data = nc.Dataset(name)
data2 = nc.Dataset(name2)
data3 = nc.Dataset(name3)
data4 = nc.Dataset(name4)
data5 = nc.Dataset(name5)

qv = data.variables['qv'][0,:,x,y]*1000
qv2 = data2.variables['qv'][0,:,x,y]*1000
qv3 = data3.variables['qv'][0,:,x,y]*1000
qv4 = data4.variables['qv'][0,:,x,y]*1000
qv5 = data5.variables['qv'][0,:,x,y]*1000

specific_humidity = np.array(((qv/1000)/(1+(qv/1000)))*1000)
specific_humidity2 = np.array(((qv2/1000)/(1+(qv2/1000)))*1000)
specific_humidity3 = np.array(((qv3/1000)/(1+(qv3/1000)))*1000)
specific_humidity4 = np.array(((qv4/1000)/(1+(qv4/1000)))*1000)
specific_humidity5 = np.array(((qv5/1000)/(1+(qv5/1000)))*1000)

print(specific_humidity)


theta =  data.variables['th'][0,:,x,y]
theta2 =  data2.variables['th'][0,:,x,y]
theta3 =  data3.variables['th'][0,:,x,y]
theta4 =  data4.variables['th'][0,:,x,y]
theta5 =  data5.variables['th'][0,:,x,y]
theta =  units.Quantity(theta, 'kelvin')
theta2 =  units.Quantity(theta2, 'kelvin')
theta3 =  units.Quantity(theta3, 'kelvin')
theta4 =  units.Quantity(theta4, 'kelvin')
theta5 =  units.Quantity(theta5, 'kelvin')

temp = units.Quantity(mpcalc.temperature_from_potential_temperature(pressure[0:60]*units.mbar, theta).magnitude-273.15, 'degC')
temp2 = units.Quantity(mpcalc.temperature_from_potential_temperature(pressure[0:60]*units.mbar, theta2).magnitude-273.15, 'degC')
temp3 = units.Quantity(mpcalc.temperature_from_potential_temperature(pressure[0:60]*units.mbar, theta3).magnitude-273.15, 'degC')
temp4 = units.Quantity(mpcalc.temperature_from_potential_temperature(pressure[0:60]*units.mbar, theta4).magnitude-273.15, 'degC')
temp5 = units.Quantity(mpcalc.temperature_from_potential_temperature(pressure[0:60]*units.mbar, theta5).magnitude-273.15, 'degC')



saturation_mixing_ratio = mpcalc.saturation_mixing_ratio(pressure[0:60]*units.hPa, temp).to('g/kg')
saturation_mixing_ratio2 = mpcalc.saturation_mixing_ratio(pressure[0:60]*units.hPa, temp2).to('g/kg')
saturation_mixing_ratio3 = mpcalc.saturation_mixing_ratio(pressure[0:60]*units.hPa, temp3).to('g/kg')
saturation_mixing_ratio4 = mpcalc.saturation_mixing_ratio(pressure[0:60]*units.hPa, temp4).to('g/kg')
saturation_mixing_ratio5 = mpcalc.saturation_mixing_ratio(pressure[0:60]*units.hPa, temp5).to('g/kg')

print(saturation_mixing_ratio)

hm = mpcalc.moist_static_energy(zt*units.meters, temp, specific_humidity*units('g/kg'))
hm2 = mpcalc.moist_static_energy(zt*units.meters, temp2, specific_humidity2*units('g/kg'))
hm3 = mpcalc.moist_static_energy(zt*units.meters, temp3, specific_humidity3*units('g/kg'))
hm4 = mpcalc.moist_static_energy(zt*units.meters, temp4, specific_humidity4*units('g/kg'))
hm5 = mpcalc.moist_static_energy(zt*units.meters, temp5, specific_humidity5*units('g/kg'))

hms = mpcalc.moist_static_energy(zt*units.meters, temp, saturation_mixing_ratio)
hms2 = mpcalc.moist_static_energy(zt*units.meters, temp2, saturation_mixing_ratio2)
hms3 = mpcalc.moist_static_energy(zt*units.meters, temp3, saturation_mixing_ratio3)
hms4 = mpcalc.moist_static_energy(zt*units.meters, temp4, saturation_mixing_ratio4)
hms5 = mpcalc.moist_static_energy(zt*units.meters, temp5, saturation_mixing_ratio5)


#plot
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(1, 5, wspace=0.3)

ax1 = plt.subplot(gs[0,0])
ax1.plot(hm,zt)
ax1.plot(hms,zt)
ax1.set_title('06LT')
ax1.set_xlabel('MSE [kJ/kg]')
ax1.set_ylabel('height [m]')
ax1.grid()

ax2 = plt.subplot(gs[0,1])
ax2.plot(hm2,zt)
ax2.plot(hms2,zt)
ax2.set_title('08LT')
ax2.set_xlabel('MSE [kJ/kg]')
ax2.tick_params(labelleft=False)   
ax2.grid()

ax3 = plt.subplot(gs[0,2])
ax3.plot(hm3,zt)
ax3.plot(hms3,zt)
ax3.set_title('10LT')
ax3.set_xlabel('MSE [kJ/kg]')
ax3.tick_params(labelleft=False)  
ax3.grid()

ax4 = plt.subplot(gs[0,3])
ax4.plot(hm4,zt)
ax4.plot(hms4,zt)
ax4.set_title('12LT')
ax4.set_xlabel('MSE [kJ/kg]')
ax4.tick_params(labelleft=False)  
ax4.grid()

ax5 = plt.subplot(gs[0,4])
ax5.plot(hm5,zt)
ax5.plot(hms5,zt)
ax5.set_title('15LT')
ax5.set_xlabel('MSE [kJ/kg]')
ax5.tick_params(labelleft=False)  
ax5.legend(['hm$_env$','hms$_env$'])
ax5.grid()

plt.savefig('week14_2.png')



