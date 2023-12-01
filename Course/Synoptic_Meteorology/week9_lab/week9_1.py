import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import os

new_directory = "/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/week9_lab/01"
os.chdir(new_directory)
#read files
filelist = []

for i in range(101,132,1):
    name = '47918.2016_0'+str(i)+'_00.txt'
    filelist.append(name)


data_dict = {}

i = 101
for file_name in filelist:
    data = np.loadtxt(file_name, skiprows=5, unpack='true')
    data_dict[i] = data
    i = i+1

missing_value = 999.9

#0107 sounding at island Ishigaki

p = data_dict[107][0]*units.hPa
t = data_dict[107][2]*units.degC
td = data_dict[107][3]

for i in range (68):
    if (td[i] == missing_value):
        td[i] = np.nan

td = td*units.degC
ws = data_dict[107][7]*1.9438*units.knots
wd = data_dict[107][6]*units.degrees
u,v = mpcalc.wind_components(ws,wd)

prof = mpcalc.parcel_profile(p,t[0], td[0]).to('degC')


fig = plt.figure(figsize=(10,10))
skew = SkewT(fig,rotation=45)              
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
theta = data_dict[107][8][0:35]
thetae = data_dict[107][9][0:35]

for i in range (35):
    if (thetae[i] == missing_value):
        thetae[i] = np.nan

pp = p[0:35]


plt.figure(figsize=(4,10))
plt.grid()
plt.plot(theta,pp,color="blue")
plt.plot(thetae,pp,color="k")
plt.ylim([1020,100])
plt.ylabel("Pressure [hPa]")
plt.xlabel("[K]")
plt.axhline(y = 1000,color = 'r', linestyle = 'dashed') 
plt.legend(["$\\theta$","$\\theta_e$","BLH"])
'''







