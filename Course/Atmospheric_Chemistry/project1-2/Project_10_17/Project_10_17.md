# Atmospheric Chemistry Projects 10/17

## Project 1-2

![1-2-1](/Users/chenyenlun/Desktop/Github/PC/python/Atmospheric_Chemistry/project1-2/output1.png)

<center>Figure.(1)

在Figure.(1)中顯示出根據Chapman mechanism的模擬，$O_3$的濃度隨著時間的變化。若假設當$O_3$的變化率小於$10^{-8}$時為此物種的steady state，此時$t = 8110150$，也就是模擬開始約94日後。

## Project 1-3

(a.1) 四個反應常數的垂直分布特徵: 除k2為固定值外，其餘常數皆隨著高度及氣溫而有所變化。

![1-3-a-1](/Users/chenyenlun/Desktop/Github/PC/python/Atmospheric_Chemistry/project1-3/1.png)

<center>Figure.(2)

(a.2)

![1-3-a-2](/Users/chenyenlun/Desktop/Github/PC/python/Atmospheric_Chemistry/project1-3/3.png)

<center>Figure.(3)

(a.3) 根據 steady state 假設的臭氧濃度高度分布為一相對平滑的曲線。由Figure.(3)中可以看出在高度約為30-35公里處有高峰值，且在接近對流層頂部約50公里處濃度未降至低點。此分佈特徵與實際臭氧垂直分布略有落差，推測可能為程式中常數設定或是 steady state 計算有誤。若與板橋探空相比較，發現其臭氧濃度最高點發生在較低處，且整體濃度的數值較小。原因可能為這項假設較不周全，未考慮牽涉到光化反應的兩個常數k1及k3因sink terms的變化。所以導致模式結果較實際值高的現象。

(b.1)

![1-3-b-1](/Users/chenyenlun/Desktop/Github/PC/python/Atmospheric_Chemistry/project1-3/2.png)

<center>Figure.(4)
