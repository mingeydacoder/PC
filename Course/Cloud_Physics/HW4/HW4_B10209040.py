import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as quad

#1.

#determined values

N0 = 1e4
lamb = 1
standard_gamma_j = 1
i1, i2, i3 = -3, 0, 3

#function

D = np.linspace(0,30,62)

def modified_gamma_size_distribution_func (i) :
    output = N0 * D ** i * np.exp(-1 * lamb * D ** standard_gamma_j)
    return output

#plot 

plt.figure()
plt.plot(D,modified_gamma_size_distribution_func(i1), color = 'r')
plt.yscale('log')
plt.grid(axis='y')
plt.title('i = -3')


plt.figure()
plt.plot(D,modified_gamma_size_distribution_func(i2), color = 'green')
plt.yscale('log')
plt.grid(axis='y')
plt.title('i = 0')


plt.figure()
plt.plot(D,modified_gamma_size_distribution_func(i3))
plt.yscale('log')
plt.grid(axis='y')
plt.title('i = 3')

#2.

#constant

N0_2 = 1e6
zeta = 2 * 1 * 0.01 * np.pi
dt = 5

nD_0 = N0 * D^3 * np.exp(-lamb * D)
dQ_dt = np.empty(120)





