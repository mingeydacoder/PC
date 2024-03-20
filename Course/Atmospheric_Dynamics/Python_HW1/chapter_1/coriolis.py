# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 10:54:33 2024

@author: TDRC
"""

'''
% PYTHON script:  coriolis.py  (uses functions xprim1 and xprim2)
% Problems M1.1
% Script to compute constant angular momentum trajectories in spherical 
% coordinates with curvature terms included [eqs. (1.10a) and (1.11a)]
% (shown in Figure 1, and with curvature terms omitted [eqs. (1.12a)
% and (1.12b)] (shown in Figure 2).
% black diamond on plot marks initial position of particle
'''

# Import packages

import numpy as np
import scipy.integrate as inte
import matplotlib.pyplot as plt
from xprim import *

# Input parameters

print('Initial longitude is zero. Specify latitude and speed when asked.')
init_lat = float(input('Specify an initial latitude in degrees:  '))
u0 = float(input('Specify a zonal wind in m/s: ' ))
v0 = float(input('Specify a meridional wind in m/s: '))

a = 6.37e6                  # Radius of the Earth
omega = 7.292e-5            # Angular velocity of the Earth
lat0 = np.radians(init_lat)
runtime = float(input('Specify integration time in days  '))
time = runtime * 24 * 3600

# Solve ODE (curvature term version)

sol1 = inte.solve_ivp(xprim1,[0,time],[u0,v0,0,lat0],vectorized = True,args = [a,omega],t_eval = np.linspace(0, int(time), int(time + 1)))

# Convert long and lat to degrees

long = np.degrees(sol1.y[2])
lat = np.degrees(sol1.y[3])

# Plot figure 1

plt.figure(figsize = [6,4],dpi = 200)
plt.plot(long,lat)
plt.plot(long[0],lat[0],'kd')
plt.title('Trajectory with curvature terms')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.tight_layout()
plt.show()

# Solve ODE (without curvature term version)

sol2 = inte.solve_ivp(xprim2,[0,time],[u0,v0,0,lat0],vectorized = True,args = [a,omega],t_eval = np.linspace(0, int(time), int(time + 1)))

# Convert long and lat to degrees

long = np.degrees(sol2.y[2])
lat = np.degrees(sol2.y[3])

# Plot figure 2

plt.figure(figsize = [6,4],dpi = 200)
plt.plot(long,lat,'r')
plt.plot(long[0],lat[0],'kd')
plt.title('Trajectory without curvature terms')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.tight_layout()
plt.show()