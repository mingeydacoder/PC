import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc

t, td, p, Theta = 20, 15 , 950, 295

theta = mpcalc.potential_temperature(500. * units.mbar, 263. * units.kelvin)
#print(theta)

temp = mpcalc.temperature_from_potential_temperature(850*units.mbar, Theta*units.kelvin)
#print(temp)

def actual_vapor_pressure(td):
    e = 6.11*10**(7.5*(td/(237.7+td)))
    return e

def sat_vapor_pressure(t):
    es = 6.11*10**(7.5*(t/(237.7+t)))
    return es

e = actual_vapor_pressure(td)
#print(e)

w = 621.97*(actual_vapor_pressure(td)/(p-actual_vapor_pressure(td)))
ws = 621.97*(sat_vapor_pressure(t)/(p-sat_vapor_pressure(t)))
#print(w,ws)

RH = w/ws
#print(RH)

m = mpcalc.mixing_ratio(25 * units.hPa, 1000 * units.hPa).to('g/kg')

# 氣塊舉升後各數值計算

t2 = mpcalc.dry_lapse(1000 * units.hPa, -10 * units.degC, 500 * units.hPa).to('degC')
print(t2)

td = 243.5