import numpy as np
import matplotlib.pyplot as plt

banqiao = np.loadtxt('OzoneSonda_202012180305.csv',delimiter=",",skiprows = 1749, unpack = "true")

Z = np.linspace(10,50,41)
Z1 = np.linspace(10,50,4662)
T = 270
n = 2000

na0 = 2.68*1e19
k1 = 1e-11
k2 = 1e-33
k3 = 1e-3
k4 = (8*(1e-12))*np.exp(-2060/T)
Co2 = 0.21
na = 1.8*1e18
t = np.linspace(0,n,n)

O = np.zeros(n)
O2 = Co2*na
O3 = np.zeros(n)
Ox = np.zeros(n)
O[0] = 0
O3[0] = 0
Ox[0] = 0

dOx = 2*k1*O2-2*k4*O*O3
dt = 1

for i in range(1000):
    dOx = 2*k1*O2-2*k4*O[i]*O3[i]
    Ox[i+1] = Ox[i] + dt*dOx
    O[i+1] = Ox[i+1]*(k3/(k3+k2*Co2*na**2))
    O3[i+1] = Ox[i+1]*(k2*Co2*na**2/(k3+k2*Co2*na**2))

#plt.plot(t,Ox)
plt.xlabel("time [t]")
plt.ylabel("$O_x$ concentration [mol/$cm^3$]")


#apply solar radiation
start = 0
end = n
interval = 50
time = np.arange(start, end, interval)
sine_values = np.sin(time)

print(len(sine_values))
plt.plot(time,sine_values)

