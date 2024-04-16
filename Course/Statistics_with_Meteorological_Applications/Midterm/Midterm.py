import numpy as np
import matplotlib.pyplot as plt

#8.1
data = np.loadtxt('Ts_monthly_1901_2013.txt')
t = np.linspace(0,1355,1356)
print(t)
plt.plot(t,data)
plt.xlabel("month")
plt.ylabel("temperature[c]")
#8.2
fit = np.polyfit(t,data,1)
trendline = fit[0]*t + fit[1]
print(fit)
plt.plot(t,trendline)
#8.4
TS_SeasonalCycle = np.tile(np.nanmean(data.reshape((-1, 12)), axis=0), data.shape[0]//12)
TS_Anomaly = data - TS_SeasonalCycle

plt.plot(t,TS_Anomaly)
plt.xlabel("month")
plt.ylabel("temperature[c]")
#8.5
fit2 = np.polyfit(t,TS_Anomaly,1)
trendline2 = fit2[0]*t + fit2[1]
print(fit2)
plt.plot(t,trendline2)