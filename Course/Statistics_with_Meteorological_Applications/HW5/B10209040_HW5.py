import numpy as np
from sympy import symbols, solve

#

data = np.loadtxt('TS.txt')
close_data = data[-180:]
reshaped_close_data = close_data.reshape(15,12)
annual_average_close_data = np.mean(reshaped_close_data, axis=1)
far_data = data[:180]
reshaped_far_data = far_data.reshape(15,12)
annual_average_far_data = np.mean(reshaped_far_data, axis=1)

mu = np.mean(annual_average_far_data)
standard_deviation = np.std(far_data)
Z = 1.645
n = 15
xbar = symbols('xbar')
xbar_real = np.mean(annual_average_close_data)

equation =  (xbar - mu)/(standard_deviation - n**(0.5)) - Z
xbar_solution = solve(equation, xbar)

print(xbar_solution)
print(xbar_real)