import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import pandas as pd 

data = np.loadtxt('46757-2022062500.txt', delimiter=",", skiprows=2, unpack='true')

h = data[1]
p = data[2]
t = data[3]
td = data[5]
ws = data[6]
wd = data[7]

'''
def wsd2uv(ws, wd):
    wd = 270 - wd
    wd = wd /180 *np.pi
    x = ws * np.cos(wd)
    y = ws * np.sin(wd)
    return(x, y)
'''

u = -ws*np.sin(wd)
v = -ws*np.cos(wd)

fig = plt.figure(figsize=(10,10))
skew = SkewT(fig,rotation=45)              #叫出斜溫圖
skew.plot(p,t,color='b')         #PT線
skew.plot(p,td,color='r')         #PTd線
skew.plot_barbs(p,u,v,xloc=1.1)     #風標，繪製於圖右側
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius)    #繪製乾絕熱線
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius)    #繪製濕絕熱線
skew.ax.set_title('2020082100')
box = skew.ax.get_position()               #這四行繪製圖右側風標的竿子
ax2 = fig.add_axes([box.x1*1.07,box.y0*0.999,0,box.y0+box.height*0.835])
ax2.set_yticks()
ax2.set_xticks()



'''
#skew.plot(p,氣塊溫度,color='k')         #氣塊抬升線
#skew.plot(p,氣塊露點,color='k')    
#skew.plot_barbs(p,u,v,xloc=1.1)     #風標，繪製於圖右側
skew.plot_mixing_lines(P=np.arange(100,1001,20)*units.hPa)        #繪製混和比線
skew.ax.set_title()                        #可以輸入繪圖設定
skew.ax.set_xlabel()
...
box = skew.ax.get_position()               #這四行繪製圖右側風標的竿子
ax2 = fig.add_axes([box.x1*1.07,box.y0*0.999,0,box.y0+box.height*0.835])
ax2.set_yticks()
ax2.set_xticks()
'''
plt.show()

