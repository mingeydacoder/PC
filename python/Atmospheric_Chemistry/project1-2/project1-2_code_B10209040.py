import numpy as np
import matplotlib.pyplot as plt

T = 270
n = 15000000

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

dt = 1

for i in range(14999999):
    dOx = 2*k1*O2-2*k4*O[i]*O3[i]
    Ox[i+1] = Ox[i] + dt*dOx
    O[i+1] = Ox[i+1]*(k3/(k3+k2*Co2*na**2))
    O3[i+1] = Ox[i+1]*(k2*Co2*na**2/(k3+k2*Co2*na**2))
    if (O3[i+1]-O3[i])/O3[i]<1e-8:
        print(i/86400)
        break


plt.plot(t/86400,O3)
plt.xlabel("time [days]")
plt.ylabel("$O_x$ concentration [molecules/$cm^3$]")


#apply solar radiation
start = 0
end = n
interval = 50
time = np.arange(start, end, interval)
sine_values = np.sin(time)


#plt.plot(time,sine_values)

