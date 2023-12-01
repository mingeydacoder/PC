import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as tck



#constant
E, L, rho, a, b = 1, 2.5, 1e6, 8500, 1
r01, r02, r03, r04 = 1.0851*1e-3, 0.8905*1e-3, 2.4818*1e-3, 2.0897*1e-3
dt = 1
dz = 1
n = 1000
R = np.zeros(n)
dRdZ = np.zeros(n)
dRdt = np.zeros(n)
z = np.linspace(0,n,n)
t = np.linspace(0,n,n)


#Bowen's model with no updraft speed

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

def rz(r0):
    Z = np.zeros(n)
    R[0] = r0
    for i in tqdm(range(n-1)):
        Z[i+1] = Z[i]+dt*(-a*R[i]**1)
        dRdZ[i] = -((E*L)/(4*rho))
        R[i+1] = R[i]+(Z[i+1]-Z[i])*dRdZ[i]
            
            
    return(R, Z)

rr, zz = rz(r01)
print(zz[0:5],rr[0:5])
plt.plot(rr[0:100],zz[0:100])

rr1, zz1 = rz(r02)
plt.plot(rr1[0:100],zz1[0:100])

rr2, zz2 = rz(r03)
plt.plot(rr2[0:100],zz2[0:100])

rr3, zz3 = rz(r04)
plt.plot(rr3[0:100],zz3[0:100])

plt.ylim([-350,0])
plt.xlim([0,0.003])

plt.legend(['A',
            'B',
            'C',
            'D'])

plt.savefig('test.png')


'''
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
plt.savefig('1.png')



plt.figure()
plt.plot(rr[0:3], zz[0:3])

plt.plot(rr1*1e3, zz1)
plt.plot(rr2*1e3, zz2)
plt.plot(rr3*1e3, zz3)


#plt.axes().xaxis.set_minor_locator(tck.AutoMinorLocator())
plt.title("Bowen's model Z-r diagram")
#plt.ylim([-350,0])
#plt.xlim([0.001,0.003])
plt.xlabel('radius [m]')
plt.ylabel('Height above cloud base [m]')
plt.legend(['A',
            'B',
            'C',
            'D'])
plt.savefig('2.png')
'''
