import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck



#constant
E, L, rho, a, b = 1, 2.5, 1e6, 8500, 1
w1, w2 = 2, 4
r01, r02 = 15*1e-6, 30*1e-6
dt = 1
dz = 1
n = 1000
R = np.zeros(n)
dRdZ = np.zeros(n)
dRdt = np.zeros(n)
z = np.linspace(0,n,n)
t = np.linspace(0,n,n)


#Bowen's model

def zt(w, r0):
    Z = np.zeros(n)
    R[0] = r0
    for i in tqdm(range(n-1)):
        dRdt[i] = ((E*L)/(4*rho))*(a*R[i]**b)
        Z[i+1] = Z[i]+dt*(w-a*R[i]**1)
        R[i+1] = R[i]+dt*dRdt[i]
        #if z[i]<0:
            #print(t[i])
    return(Z)

def rz(w, r0):
    Z = np.zeros(n)
    R[0] = r0
    for i in tqdm(range(n-1)):
        Z[i+1] = Z[i]+dt*(w-a*R[i]**1)
        dRdZ[i] = ((E*L)/(4*rho))*(a*R[i]**b)*(1/(w-a*R[i]**1))
        R[i+1] = R[i]+(Z[i+1]-Z[i])*dRdZ[i]
            
            
    return(R, Z)

rr, zz = rz(w1,r01)
print(rr[790:810])
rr1, zz1 = rz(w1,r02)
print(rr1[620:640])
rr2, zz2 = rz(w2,r01)
print(rr2)
rr3, zz3 = rz(w2,r02)
print(rr3[790:810])


#plot

plt.figure()
plt.plot(t, zt(w1,r01))
plt.plot(t, zt(w1,r02))
plt.plot(t, zt(w2,r01))
plt.plot(t, zt(w2,r02))
plt.title("Bowen's model Z-t diagram")
plt.ylim([0,2500])
plt.xlabel('time [s]')
plt.ylabel('Height above cloud base [m]')
plt.legend(['w = 2 m/s, $r_0$ = 15 $\mu m$',
            'w = 2 m/s, $r_0$ = 30 $\mu m$',
            'w = 4 m/s, $r_0$ = 15 $\mu m$',
            'w = 4 m/s, $r_0$ = 30 $\mu m$'])

plt.figure()
plt.plot(rr, zz)
plt.plot(rr1, zz1)
plt.plot(rr2, zz2)
plt.plot(rr3, zz3)
#plt.axes().xaxis.set_minor_locator(tck.AutoMinorLocator())
plt.title("Bowen's model Z-r diagram")
plt.ylim([0,2500])
plt.xlabel('radius [m]')
plt.ylabel('Height above cloud base [m]')
plt.legend(['w = 2 m/s, $r_0$ = 15 $\mu m$',
            'w = 2 m/s, $r_0$ = 30 $\mu m$',
            'w = 4 m/s, $r_0$ = 15 $\mu m$',
            'w = 4 m/s, $r_0$ = 30 $\mu m$'])

