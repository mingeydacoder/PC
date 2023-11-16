import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import os
import matplotlib.gridspec as gridspec

data = np.loadtxt('/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/hw5/IdealizedSounding.csv', delimiter=',',skiprows=1, unpack='true')


p = data[0]*units.hPa
t = (data[2]-273.15)*units.degC
tk = (data[2])*units.kelvin
td = (data[3]-273.15)*units.degC
theta = mpcalc.potential_temperature(p,tk)
print(theta)


prof = mpcalc.parcel_profile(p,t[0], td[0]).to('degC')


fig = plt.figure(figsize=(20,20))
gs = gridspec.GridSpec(1, 3)

skew = SkewT(fig,rotation=45,subplot=gs[0,0])
skew.plot(p,t,color='b')         
skew.plot(p,td,color='r')   
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius)    
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius) 
skew.plot(p,prof,'k')
skew.ax.set_xlim([-30,30])

ax = plt.subplot(gs[0, 1])
ax.plot(theta,p,color="blue")
ax.set_box_aspect(1.38)
ax.set_ylim(1050, 100)
ax.set_yscale('log')
ax.set_yticks([1000,900,800,700,600,500,400,300,200,100])




'''
skew.plot(p,t,color='b')         
skew.plot(p,td,color='r')         
skew.plot_barbs(p,u,v,xloc=1.057)    
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius)    
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius)  
skew.plot_mixing_lines(pressure=np.arange(100,1001,20)*units.hPa) 
skew.ax.set_xlim(-40,40)
skew.plot(p,prof,'k')
skew.shade_cape(p,t,prof)
skew.shade_cin(p,t,prof)

cape, cin = mpcalc.cape_cin(p, t, td, prof)
pw = mpcalc.precipitable_water(p,td)

print('CAPE & CIN & PW:',cape, cin, pw)

props = dict(boxstyle='round', alpha=0.5)

skew.ax.text(0.7, 0.95, 'PW: 23.9 mm\nCAPE: 807.27 m\u00b2/s\u00b2\nCIN: -42.17m \u00b2/s\u00b2\nBLH: 180.0 m', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top')
skew.ax.text(0.8, 1.035, '2016010700', transform=skew.ax.transAxes,fontsize=17, verticalalignment='top')
skew.ax.text(0.005, 1.035, '47918 Ishgaki Island', transform=skew.ax.transAxes,fontsize=17, verticalalignment='top')








'''         

import numpy as np

# 创建一个3D数组（示例数据）
# 假设arr是一个形状为 (n, m, p) 的3D数组
n, m, p = 3, 4, 5
arr = np.random.randint(1, 10, size=(n, m, p))

# 指定要沿着哪个维度计算总和
# 在这个例子中，我们选择沿着第二个维度计算总和（axis=1）
# 这将生成一个形状为 (n, p) 的2D数组
sum_2d = np.sum(arr, axis=0)

# 打印原始3D数组和计算总和后的2D数组
print("原始3D数组：\n", arr)
print("\n计算总和后的2D数组：\n", sum_2d)