import csv 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

ase1 = []
with open('C:/Users/Anton.Putintsev/Desktop/HD6(LASING) 20.12.2019/Preliminary_data.txt', newline='') as inputfile:
    for row in csv.reader(inputfile, delimiter='\t'):
        ase1.append(row)
ase1=pd.DataFrame(ase1)
ase1=ase1.values
ase1=ase1.astype(np.float) 

x=ase1[:,0]
fig1=plt.figure()
ax = plt.subplot(111)

ax.plot(x, ase1[:,5],'Hr', markerfacecolor='red', marker="o", markeredgecolor="black", markersize=10, markeredgewidth=0.3, alpha=0.7)

plt.title('FWHM',size=20)
"""plt.plot(x,ase1[350:730,1]*500/391,'-r', x,ase2[350:730,1]*500/382,'-b', x,ase3[350:730,1]*500/364,'-g', x,ase4[350:730,1]*500/298,'-k', x,ase5[350:730,1]*500/312,'-m')"""
plt.xlabel('Pulse power, [mV]',size=16)
plt.ylabel('FWHM, [nm]',size=16)
plt.grid(linestyle='dotted')
plt.savefig('C:/Users/Anton.Putintsev/Desktop/HD6(LASING) 20.12.2019/fwhm.jpg') 
plt.show()