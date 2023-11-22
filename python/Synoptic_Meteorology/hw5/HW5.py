import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import os
import matplotlib.gridspec as gridspec
from scipy.optimize import fsolve

data = np.loadtxt('/Users/chenyenlun/Desktop/Github/PC/python/Synoptic_Meteorology/hw5/IdealizedSounding.csv', delimiter=',',skiprows=1, unpack='true')

n = 16
p = data[0]*units.hPa
h = data[1]
t = (data[2]-273.15)*units.degC
tk = (data[2])*units.kelvin
td = (data[3]-273.15)*units.degC
theta = mpcalc.equivalent_potential_temperature(p,t,td)
prof = mpcalc.parcel_profile(p, t[0], td[0]).to('degC')

alpha = 1/273

p2 = np.zeros(n)

for i in range(n):
    p2[i] = data[0][i]*(1/(10**(1000/18400/(1+alpha*(data[2][i+1]-data[2][i])))))
#print(p2)

p2 = np.append(p2,data[0][17:27])

def lifting(t0,td0,h0):
    t2 = t0
    td2 = td0
    j = 0
    for j in range(10000):
        t2 = t2-0.1
        td2 = td2-0.02
        j +=1
        if td2>t2:
            break
    
    LCL = h0+10*(j-1)

    if LCL-h0<1000:
        t2 = td2 = t2-(1000-LCL)*0.006
    else:
        t2 = t0-9.8
        td2 = td0-2
    return(t2,td2)



#print(lifting(data[2][0],data[3][0],data[1][0]))

newt = np.zeros(n)
newtd = np.zeros(n)

for k in range(n):
    newt[k], newtd[k] = lifting(data[2][k],data[3][k],data[1][k])

#print(newt,'\n', newtd)

newt = np.append(newt,data[2][17:27])
newtd = np.append(newtd,data[3][17:27])


#lift = mpcalc.parcel_profile_with_lcl()


fig = plt.figure(figsize=(15,15))
gs = gridspec.GridSpec(1, 3, width_ratios=[2,0.7,2], wspace=0.2)

skew = SkewT(fig,rotation=45,subplot=gs[0,0])
skew.plot(p,t,color='b')         
skew.plot(p,td,color='r')   
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius, linewidths=0.8)    
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius, linewidths=0.8) 
skew.plot(p,prof,'k',linewidth=3)
#skew.shade_cape(p,t,prof)
#skew.shade_cin(p,t,prof)
skew.ax.set_xlim([-20,30])
plt.axhline(y = 940.55,color = 'r', linestyle = 'dashed') 


cape, cin = mpcalc.cape_cin(p, t, td , prof)
#pw = mpcalc.precipitable_water(p,td*units.degC)

print('CAPE & CIN:',cape, cin)

skew.ax.text(0.6, 0.95, 'CAPE: 0.0 m\u00b2/s\u00b2\nCIN: 0.0 m\u00b2/s\u00b2\nLCL: 940.55 hPa\nLFC: none\nEL: none', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top')
skew.ax.text(0.04, 0.1, 'LCL', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top', color='red')


#plot thetae
ax = plt.subplot(gs[0,1])
ax.plot(theta,p,color="blue")
ax.set_box_aspect(4.7)
ax.set_ylim(1050, 100)
ax.set_yscale('log')
ax.set_ylabel(' ')
ax.set_yticks([1000,900,800,700,600,500,400,300,200,100])
plt.yticks([]) 
ax.set_xlabel('$\\theta_e$ [K]')
ax.grid()

#plot lifted profile

prof2 = mpcalc.parcel_profile(p2*units.hPa, newt[0]*units.kelvin, newtd[0]*units.kelvin).to('degC')

skew2 = SkewT(fig,rotation=45,subplot=gs[0,2])
skew2.plot(p2,newt-273.15,color='b')         
skew2.plot(p2,newtd-273.15,color='r')  
skew2.plot(p2,prof2,'k', linewidth=3)
skew2.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius, linewidths=0.8)    
skew2.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius, linewidths=0.8) 
skew2.ax.set_xlim([-20,30])
skew2.ax.set_ylabel(' ')
cape2, cin2 = mpcalc.cape_cin(p2*units.hPa, newt*units.kelvin, newtd*units.kelvin , prof2)
skew2.ax.text(2.2, 0.95, 'CAPE: 479.34 m\u00b2/s\u00b2\nCIN: -210.20 m\u00b2/s\u00b2\nLCL: 881.54 hPa\nLFC: none\nEL: none', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top')
print(cape2, cin2)
plt.axhline(y = 881.54,color = 'r', linestyle = 'dashed') 

lcl = mpcalc.lcl(p2*units.hPa, newt*units.kelvin, newtd*units.kelvin)
skew2.ax.text(1.7, 0.11, 'LCL', transform=skew.ax.transAxes,fontsize=14, verticalalignment='top', color='red')
print(lcl)




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
