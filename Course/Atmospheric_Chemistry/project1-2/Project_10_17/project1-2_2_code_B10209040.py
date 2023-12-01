import numpy as np              
import numpy.ma as ma
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
steady = 8110150

O = np.zeros(n)
O2 = Co2*na
O3 = np.zeros(n)
Ox = np.zeros(n)
O[0] = 0
O3[0] = 0
Ox[0] = 0

dt = 1

#apply solar radiation
start = 0
end = n
sample_rate = 1
time = np.arange(start, end, 1/sample_rate)
frequency = 1/86400
amplitude = 1
theta = 0
sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
for i in range (end):
    if sinewave[i]<0:
        sinewave[i]=0


for i in range(steady,n-1):
    dOx = 2*k1*O2-2*k4*O[i]*O3[i]
    Ox[i+1] = Ox[i] + dt*dOx
    O[i+1] = Ox[i+1]*(k3/(k3+k2*Co2*na**2))
    O3[i+1] = Ox[i+1]*(k2*Co2*na**2/(k3+k2*Co2*na**2))
    
    if i>=steady:
        k1 = k1*sinewave[i-steady]
        k3 = k3*sinewave[i-steady]
            
plt.plot(t[steady:],Ox[steady:])



