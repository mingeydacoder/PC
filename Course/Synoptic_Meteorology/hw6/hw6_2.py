import numpy as np
import matplotlib.pyplot as plt
import metpy.calc as mecalc
from metpy.units import units
from scipy.integrate import simpson
import scipy.integrate as integrate
from numpy import trapz

data = np.loadtxt('TNNUA_20210402_0600_L4.txt', skiprows=14, unpack='true')

plt.figure(figsize=[6.4,6.4])

theta = mecalc.potential_temperature(data[4]*units.mbar, (data[5]+273.15)*units.kelvin)
plt.plot(theta, data[16])



plt.legend(['06'])
plt.ylim([0,3000])
plt.xlim([290,320])
plt.grid()
plt.xlabel('Potential temp [K]')
plt.ylabel('Height [m]')

f = lambda t: 0.13*np.sin((2*np.pi)/86400) 
area = integrate.quad(f, 0, 10)
print(area)

'''
area = trapz(theta, dx=5)
print("area =", area)
'''
