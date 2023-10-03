import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import pandas as pd

data = np.loadtxt("Car_CO2.csv", delimiter=",", skiprows=1, unpack="true")

time = data[0]
co2 = data[1]

#sources and sinks in ppm
exhalation = 3*17/60*0.5*0.04/3500*1e6
inhalation = 3*17/60*0.5/3500
air_in = 1.5*420/3500
air_out = 1.5/3500

c = np.zeros(1800)
c[0] = 595
dt = 3

for i in range(0,1799):
    c[i+1] = c[i]+dt*((exhalation+air_in)-((inhalation*c[i])+(air_out*c[i])))
    if i>=1260:
        c[i+1] = c[i]+dt*((exhalation+20*air_in)-((inhalation*c[i])+(20*air_out*c[i])))
print(c)

plt.plot(time,co2)
plt.plot(time,c)
plt.xlim(0,5400)
plt.ylim(0,10000)
plt.show()