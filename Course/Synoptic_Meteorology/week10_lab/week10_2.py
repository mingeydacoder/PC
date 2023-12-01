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

winds = nc.Dataset('/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week10_lab/taiwanvvmL_20160107_SHL.L.Dynamic-000000.nc')
print(winds)
u = winds.variables['u'][0,:,0,0]
u = np.insert(u,0,0)
print(u)
v = winds.variables['v'][0,:,0,0]
v = np.insert(v,0,0)

data = np.loadtxt('fort1.98', skiprows=66, unpack='True')
p = data[3]/100*units.hPa


t = mpcalc.temperature_from_potential_temperature(p, data[2]*units.kelvin)
tc = t.to('degC')
ws = mpcalc.saturation_mixing_ratio(p, tc).to('g/kg')
RH = ((data[5]*1000)/ws)*100
td = t.magnitude-((100-RH.magnitude)/5)
td = td-273.15
thetae = mpcalc.equivalent_potential_temperature(p,tc,td*units.degC)

prof = mpcalc.parcel_profile(p,tc[0], td[0]*units.degC).to('degC')

fig = plt.figure(figsize=(10,10))
gs = gridspec.GridSpec(1, 2, width_ratios=[2,0.7], wspace=0.4)

skew = SkewT(fig,rotation=45,subplot=gs[0,0])
skew.plot(p,t,color='b')         
skew.plot(p,td,color='r')   
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius, linewidths=0.8)    
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius, linewidths=0.8) 
skew.plot(p,prof,'k')
skew.shade_cape(p,t,prof)
skew.shade_cin(p,t,prof)
skew.plot_barbs(p,u*1.944,v*1.944,xloc=1.15)
skew.ax.set_xlim([-30,30])

skew.ax.text(0.6, 0.95, 'PW: 27.4 mm\nCAPE: 15.24 m\u00b2/s\u00b2\nCIN: -27.36m \u00b2/s\u00b2\nBLH: 350.0 m', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top')
skew.ax.text(0.7, 1.035, 'initial profile', transform=skew.ax.transAxes,fontsize=15, verticalalignment='top')
skew.ax.text(0.005, 1.035, '47918 Ishgaki Island', transform=skew.ax.transAxes,fontsize=15, verticalalignment='top')

cape, cin = mpcalc.cape_cin(p, tc, td*units.degC , prof)
pw = mpcalc.precipitable_water(p,td*units.degC)

print('CAPE & CIN & PW:',cape, cin, pw)

#plot theta thetae
ax = plt.subplot(gs[0,1])
ax.plot(data[2],p,color="blue")
ax.plot(thetae,p, color='k')
ax.set_box_aspect(3.95)
ax.set_ylim(1050, 100)
ax.set_yscale('log')
ax.set_ylabel(' ')
ax.set_yticks([1000,900,800,700,600,500,400,300,200,100])
plt.yticks([]) 
ax.set_xlabel('[K]')
ax.grid()
plt.axhline(y = 980,color = 'r', linestyle = 'dashed') 
plt.legend(["$\\theta$","$\\theta_e$","BLH"],loc='upper left')

'''
左右圖皆為2016.01.07石垣島在00Z時的探空圖，但左圖為理想化之模擬資料。此探空圖之T及Td由fort.98中的theta計算而得。可以發現於地表及較高層時，模擬及實際觀測的結果差距
較小，中間部分兩者皆可以看出逆溫現象，唯實際觀測結果之露點溫度在700hPa處下降較多，theta及thetae也有相似的趨勢。風向方面模擬結果與觀測相去不遠，唯實際測量到的風速較大
。
'''

