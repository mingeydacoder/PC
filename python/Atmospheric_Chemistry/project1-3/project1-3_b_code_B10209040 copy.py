import numpy as np
import matplotlib.pyplot as plt

ALT = np.loadtxt('ALT_80N_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
VTO = np.loadtxt('VTO_50N_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
CWB = np.loadtxt('CWB_25N_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
PMO = np.loadtxt('PMO_05N_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
LAU = np.loadtxt('LAU_45S_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
SYO = np.loadtxt('SYO_70S_100m.csv',delimiter=",",skiprows = 1, unpack = "true")
SPO = np.loadtxt('SPO_90S_100m.csv',delimiter=",",skiprows = 1, unpack = "true")

ALTppm = ALT[2]/ALT[1]*1e5
VTOppm = VTO[2]/VTO[1]*1e5
CWBppm = CWB[2]/CWB[1]*1e5
PMOppm = PMO[2]/PMO[1]*1e5
LAUppm = LAU[2]/LAU[1]*1e5
SYOppm = SYO[2]/SYO[1]*1e5
SPOppm = SPO[2]/SPO[1]*1e5

ALTcon = ALTppm/48/1000000
VTOcon = VTOppm/48/1000000
CWBcon = CWBppm/48/1000000
PMOcon = PMOppm/48/1000000
LAUcon = LAUppm/48/1000000
SYOcon = SYOppm/48/1000000
SPOcon = SPOppm/48/1000000

print(ALTppm)

plt.subplot(1,2,1)
plt.plot(ALTppm,ALT[0]/1000)
plt.plot(VTOppm,VTO[0]/1000)
plt.plot(CWBppm,CWB[0]/1000)
plt.plot(PMOppm,PMO[0]/1000)
plt.plot(LAUppm,LAU[0]/1000)
plt.plot(SYOppm,SYO[0]/1000)
plt.plot(SPOppm,SPO[0]/1000)
plt.legend(["Alert (ALT)","Valentia (VTO)","Taipei (CWB)","Paramaribo (PMO)","Lauder (LAU)","Syowa (SYO)","South Pole (SPO)"])

plt.subplot(1,2,2)
plt.plot(ALTcon,ALT[0]/1000)
plt.plot(VTOcon,VTO[0]/1000)
plt.plot(CWBcon,CWB[0]/1000)
plt.plot(PMOcon,PMO[0]/1000)
plt.plot(LAUcon,LAU[0]/1000)
plt.plot(SYOcon,SYO[0]/1000)
plt.plot(SPOcon,SPO[0]/1000)
plt.legend(["Alert (ALT)","Valentia (VTO)","Taipei (CWB)","Paramaribo (PMO)","Lauder (LAU)","Syowa (SYO)","South Pole (SPO)"])


#plt.plot(ALTcon,ALT[0]/1000)