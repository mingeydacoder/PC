import numpy as np
import matplotlib.pyplot as plt

banqiao = np.loadtxt('OzoneSonda_202012180305.csv',delimiter=",",skiprows = 1749, unpack = "true")

Z = np.linspace(10,50,41)
Z1 = np.zeros(4662)

for i in range(4662):
    Z1[i] = banqiao[3][i]/1000

T = np.zeros(41)
for i in range(41):
    T[i] = 215
    if i>=10:
        T[i] = 215+2.5*(i-10)

na0 = 2.68*1e19
k1 = (5*1e-10)*np.exp((-1.2*1e2)*np.exp(-Z/7.4))
k2 = np.full(41, 1e-33) 
k3 = 10**((1/40*Z)-3.75)
k4 = (8*(1e-12))*np.exp(-2060/T)


#(a).1.
#f, ax = plt.subplots(2,2,figsize = (7.5,7.5))
ax[0,0].plot(k1,Z)
ax[0,0].set_xlabel('k1')
ax[0,0].set_ylabel('Z')

ax[1,0].plot(k3,Z)
ax[1,0].set_xlabel('k3')
ax[1,0].set_ylabel('Z')

ax[1,1].plot(k4,Z)
ax[1,1].set_xlabel('k4')
ax[1,1].set_ylabel('Z')

ax[0,1].plot(k2,Z)
ax[0,1].set_xlabel('k2')
ax[0,1].set_ylabel('Z')

#(a).2.a

O3 = (((k1*k2)/(k3*k4))**(0.5))*0.21*(na0**(1.5))*1000 #unit:molecule/cm^3
O3_banqiao = banqiao[1]/banqiao[0]/48/1000*(6.02*1e23)

print(O3)
plt.plot(O3,Z)
plt.plot(O3_banqiao,Z1)
plt.xlabel("O$_{3}$ concentration [molecules/$cm^3$]")
plt.ylabel("Height [km]")
plt.legend(["Chapman Theory","Sonda Observation"])



