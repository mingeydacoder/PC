# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:54:33 2024

@author: TDRC
"""

'''
% Python script  thermo_profile.py
% This script shows simple example of reading in data and plotting a 
% monthly mean temperature profile versus pressure 

% Note that data file to be loaded must consist entirely of columns of data
% Edit data to remove headers before loading
% In the example there are 3 columns corresponding to pressure, 
% temperature, and number of soundings respectively. each column has M rows
% A(3,M) is the size of the array containing the data
% The data is at altitude interval of 0.5 km starting at z = 0 km
'''

# Import packages

import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units

A = np.loadtxt('tropical_temp.dat')  # load the data

# Plot temperature versus pressure 

plt.figure(figsize = [8,5],dpi = 200)
plt.subplot(1,2,1)

# Plot 2nd row along x-axis,  first row along y-axis.

plt.plot(A[:,1],A[:,0]) 

plt.ylim(1000,0)
plt.xlabel('Temperature (K)')
plt.ylabel('Pressure (mb)')
plt.title(' Temperature vs pressure: tropical sounding')
plt.tight_layout()

# Calculate haight

z = np.arange(0,0.5 * A.shape[0],0.5)

# Plot temperature versus altitude

plt.subplot(1,2,2)
plt.plot(A[:,1],z)

# Plot 2nd row along x-axis,  first row along y-axis.

plt.ylim(z[0],z[-1])
plt.xlabel('Temperature (K)')
plt.ylabel('Height (m)')
plt.title(' Temperature vs height: tropical sounding')
plt.tight_layout()
plt.show()

# Do it yourself after the tutoriol!!
# Plot the potential temperature profile versus pressure and hight via python script.
# Please complete your python plotting script after this line.
# If you want to do this with another language (matlab, julia, etc.), please inform TA first.

R = 287.43
Cp = 1005
Theta = A[:,1] * (1000/A[:,0])**(R/Cp)
Geopot = mpcalc.height_to_geopotential(z * units.m)/10

plt.figure(figsize = [7,5],dpi = 200)

plt.subplot(1,2,1)
plt.plot(Theta,A[:,0],color='red')
plt.plot(A[:,1],A[:,0],color='blue')
plt.ylim(1000,0)
plt.xlim(0,1500)
plt.grid()
plt.title('T and Theta vs P: tropical sounding')
plt.ylabel('hPa')
plt.xlabel('K')
plt.legend(['$\Theta$','T'])
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(Theta,Geopot,color='red')
plt.plot(A[:,1],Geopot,color='blue')
plt.ylim(0,40)
plt.xlim(0,1500)
plt.grid()
plt.title('T and Theta vs GPH: tropical sounding')
plt.ylabel('km')
plt.xlabel('K')
plt.legend(['$\Theta$','T'])
plt.tight_layout()

print(Geopot)