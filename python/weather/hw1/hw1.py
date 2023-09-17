import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import pandas as pd 

data = np.loadtxt('46757-2022062500.txt', delimiter=",", skiprows=2, unpack='true')

h = data[1]
p = data[2]

fig = plt.figure(figsize=(10,10))
skew = SkewT(fig,rotation=45)              #叫出斜溫圖
skew.plot(壓力,環境溫度,color='b')         #PT線
skew.plot(壓力,環境露點,color='r')         #PTd線
skew.plot(壓力,氣塊溫度,color='k')         #氣塊抬升線
skew.plot(壓力,氣塊露點,color='k')    
skew.plot_barbs(壓力,u風,v風,xloc=1.1)     #風標，繪製於圖右側
skew.plot_dry_adiabats(t0=np.arange(-20,151,10)*units.celsius)    #繪製乾絕熱線
skew.plot_moist_adiabats(t0=np.arange(-20,46,5)*units.celsius)    #繪製濕絕熱線
skew.plot_mixing_lines(p=np.arange(100,1001,20)*units.hPa)        #繪製混和比線
skew.ax.set_title()                        #可以輸入繪圖設定
skew.ax.set_xlabel()
...
box = skew.ax.get_position()               #這四行繪製圖右側風標的竿子
ax2 = fig.add_axes([box.x1*1.07,box.y0*0.999,0,box.y0+box.height*0.835])
ax2.set_yticks()
ax2.set_xticks()
plt.show()

