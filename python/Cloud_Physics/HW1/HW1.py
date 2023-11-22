#Cloud Physics Hw1
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc

T, P, cp, Rd, cw, Lv = 305, 1015, 1004, 287, 4187, 2.5*10**6
P1, P2 = 910, 800

#(1),(2),(3) calculation

e = 6.11*10**(7.5*(21/(237.7+21))) #actual water vapor pressure [hPa]
es = 6.11*10**(7.5*(32/(237.7+32))) #saturated water vapor pressure [hPa]
ws = 621.97*(e/(1015-e)) #water vapor mixing ratio [g/kg]

costant = (T/(P**(Rd/(cp+ws*cw))))*np.exp((ws*Lv)/(T*(cp+ws*cw))) #not reach saturation yet, Q = ws
print(costant)

plevs = [1015, 910, 800] * units.hPa
Tup = mpcalc.dry_lapse(plevs, 32*units.degC).to('degC') #(1)
print(Tup)

es1 = es*np.exp((-40700/8.3145)*((1/(22.62+273.15))-(1/T))) #not reach saturation yet, Q = ws, x = 0 #(3)
ws1 = 621.97*(e/(910-e)) #water vapor mixing ratio at 910 hPa [g/kg] #(2)
print("ws1 =", ws1)

#(4),(5),(6) calculation

es2 = es*np.exp((-40700/8.3145)*((1/(11.94+273.15))-(1/T))) #reach saturation,x = Q - ws #(6)
ws2 = 621.97*(e/(800-e)) #(5)
print("ws2 =", ws2)

Q = 20
for i in range (200):
    costant1 = ((11.94+273.15)/(P2**(Rd/(cp+Q*cw))))*np.exp((ws2*Lv)/((11.94+273.15)*(cp+Q*cw)))
    if abs(costant1-costant)/costant<=0.0001:
        print("Q = 20 +",0.01*i,"c =",costant1)
        break
    Q += 0.01

X2 = Q - ws2
print(X2)
