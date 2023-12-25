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

zt = np.linspace(0,3900,40)
qvto = np.loadtxt('fort1.98', skiprows=66, usecols=(5,))

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

qv = data.variables['qv'][0,1:41,x,y]*1000
qv2 = data2.variables['qv'][0,1:41,x,y]*1000
qv3 = data3.variables['qv'][0,1:41,x,y]*1000
qv4 = data4.variables['qv'][0,1:41,x,y]*1000
qv5 = data5.variables['qv'][0,1:41,x,y]*1000


theta =  data.variables['th'][0,1:41,x,y]
theta2 =  data2.variables['th'][0,1:41,x,y]
theta3 =  data3.variables['th'][0,1:41,x,y]
theta4 =  data4.variables['th'][0,1:41,x,y]
theta5 =  data5.variables['th'][0,1:41,x,y]
print(theta5)

def dtheta_dz_max(theta,dz):
    x = np.empty(40)
    for i in range(39):
        x[i+1] = (theta[i+1]-theta[i])/dz
    x[0] = 0
    index = np.argmax(x)
    result = np.max(x)
    return(index,result,x)

def dqv_dz_min(qv,dz):
    x = np.empty(40)
    for i in range(39):
        x[i+1] = (qv[i+1]-qv[i])/dz
    x[0] = 0
    index = np.argmin(x)
    result = np.min(x)
    return(index,result,x)

a = dqv_dz_min(qv5,100)
print(a)
b = dtheta_dz_max(theta5,100)
print(b)

#plot
fig = plt.figure(figsize=(15,10))
gs = gridspec.GridSpec(1, 5, wspace=0.3)

ax1 = plt.subplot(gs[0,0])
ax1.plot(qv,zt)
ax1.set_title('06LT')
ax1.set_xlabel('qv [g/kg]', color='#1f77b4')
ax1.set_ylabel('height [m]')
ax1.axhline(y=480, color='r', linestyle='--')
ax1.axhline(y=2200, color='green', linestyle='--', dashes=(4.5, 3))
ax1.axhline(y=2200, color='lightblue', linestyle='--', dashes=(4, 3))
ax1.tick_params(axis='x', labelcolor='#1f77b4')
bx1 = ax1.twiny()
bx1.set_xlabel('$\\theta & \\theta_e$',color='green')
bx1.tick_params(axis='x', labelcolor='green')
bx1.plot(theta,zt,color = 'green')

ax2 = plt.subplot(gs[0,1])
ax2.plot(qv2,zt)
ax2.set_title('08LT')
ax2.set_xlabel('qv [g/kg]', color='#1f77b4')
ax2.axhline(y=500, color='r', linestyle='--')
ax2.axhline(y=2200, color='green', linestyle='--', dashes=(4.5, 3))
ax2.axhline(y=2200, color='lightblue', linestyle='--', dashes=(4, 3))
ax2.tick_params(labelleft=False, axis='x', labelcolor='#1f77b4')   
bx2 = ax2.twiny()
bx2.set_xlabel('$\\theta & \\theta_e$',color='green')
bx2.tick_params(axis='x', labelcolor='green')
bx2.plot(theta2,zt,color = 'green')

ax3 = plt.subplot(gs[0,2])
ax3.plot(qv3,zt)
ax3.set_title('10LT')
ax3.set_xlabel('qv [g/kg]', color='#1f77b4')
ax3.axhline(y=400, color='r', linestyle='--')
ax3.axhline(y=2200, color='green', linestyle='--', dashes=(4.5, 3))
ax3.axhline(y=2200, color='lightblue', linestyle='--', dashes=(4, 3))
ax3.tick_params(labelleft=False, axis='x', labelcolor='#1f77b4')
bx3 = ax3.twiny()
bx3.set_xlabel('$\\theta & \\theta_e$',color='green')
bx3.tick_params(axis='x', labelcolor='green')
bx3.plot(theta3,zt,color = 'green')

ax4 = plt.subplot(gs[0,3])
ax4.plot(qv4,zt)
ax4.set_title('12LT')
ax4.set_xlabel('qv [g/kg]', color='#1f77b4')
ax4.axhline(y=370, color='r', linestyle='--')
ax4.axhline(y=2600, color='green', linestyle='--', dashes=(4.5, 3))
ax4.axhline(y=2600, color='lightblue', linestyle='--', dashes=(4, 3))
ax4.tick_params(labelleft=False, axis='x', labelcolor='#1f77b4')
bx4 = ax4.twiny()
bx4.set_xlabel('$\\theta & \\theta_e$',color='green')
bx4.tick_params(axis='x', labelcolor='green')
bx4.plot(theta4,zt,color = 'green')

ax5 = plt.subplot(gs[0,4])
ax5.plot(qv5,zt,label='_nolegend_')
ax5.set_title('15LT')
bx5 = ax5.twiny()
bx5.set_xlabel('$\\theta & \\theta_e$',color='green')
ax5.axhline(y=400, color='r', linestyle='--')
ax5.axhline(y=2700, color='green', linestyle='--', dashes=(4.5, 3))
ax5.axhline(y=2700, color='lightblue', linestyle='--', dashes=(4, 3))
ax5.tick_params(labelleft=False, axis='x', labelcolor='#1f77b4')
ax5.legend(['$\\theta_{sfc}+0.5k$','$d\\theta/dz$ MAX','$dq_v/dz$ min'])
ax5.set_xlabel('qv [g/kg]', color='#1f77b4')
bx5.tick_params(axis='x', labelcolor='green')
bx5.plot(theta5,zt,color = 'green')



plt.savefig('week14.png')



