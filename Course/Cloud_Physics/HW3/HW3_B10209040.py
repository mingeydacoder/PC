import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck



#constant
E, L, rho, a, b = 1, 2.5, 1e6, 8500, 1
r01, r02, r03, r04 = 1.0851*1e-3, 0.8905*1e-3, 2.4818*1e-3, 2.0897*1e-3
S = 0.6
one_over_Fk_plus_Fd = 120
dt = 1
dz = 1
n = 1000
R = np.zeros(n)
dRdZ = np.zeros(n)
dRdt = np.zeros(n)
z = np.linspace(0,n,n)
t = np.linspace(0,n,n)


#Bowen's model and Mason eq. with no updraft 

def rt(r0):
    Z = np.zeros(n)
    R[0] = r0*1e6
    for i in tqdm(range(n-1)):
        dRdt[i] = (S-1)*one_over_Fk_plus_Fd/R[i]
        R[i+1] = R[i]+dt*dRdt[i]
    return(R)

def rz(r0):
    Z = np.zeros(n)
    R[0] = r0*1e6
    for i in range(n-1):
        dRdt[i] = (1-S)*one_over_Fk_plus_Fd/R[i]
        R[i+1] = R[i]+dt*dRdt[i]
        Z[i+1] = Z[i]+dt*(-a*(R[i])**1)
            
            
    return(R, Z)

'''
rr, zz = rz(r01)
plt.ylim([-300,0])
plt.plot(t,zz)
plt.savefig('1.png')
'''


rr, zz = rz(r01)
for i in range(n):
    if np.abs(zz[i]*1e-6+300)<5:
        print(i,zz[i]*1e-6,rr[i]*1e-3,'A') 

rr1, zz1 = rz(r02)
for i in range(n):
    if np.abs(zz1[i]*1e-6+300)<5:
        print(i,zz1[i]*1e-6,rr1[i]*1e-3,'B') 

rr2, zz2 = rz(r03)
for i in range(n):
    if np.abs(zz2[i]*1e-6+300)<5:
        print(i,zz2[i]*1e-6,rr2[i]*1e-3,'C') 

rr3, zz3 = rz(r04)
for i in range(n):
    if np.abs(zz3[i]*1e-6+300)<5:
        print(i,zz3[i]*1e-6,rr3[i]*1e-3,'D') 





