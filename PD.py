import csv 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
import numpy as np

data = []
with open('C:/Users/Anton.Putintsev/Desktop/HD6(LASING) 20.12.2019/Preliminary_data.txt', newline='') as inputfile:
    for row in csv.reader(inputfile, delimiter='\t'):
        data.append(row)
data=pd.DataFrame(data)
data=data.values
data=data.astype(np.float) 



"""def BS041_1(o):
    return o*0.9731-1.2474
def BS041_2(o):
    return o*18.283-38.575"""
    
def fit_1(o):
    return (o**0.5333)*(10**0.6225)
def fit_2(o):
    return (o**18.946)/10**38.039

x=data[:,0]
"""x1=np.arange(1.4,2.22,0.01)
x2=np.arange(2.14,2.3,0.01)"""

x1=np.arange(27,140,1)
x2=np.arange(120,175,1)

fig1=plt.figure()
ax = plt.subplot(111)
"""ax.plot(x, ase1[:,1], label='61.98 mV',color=)"""

ax.plot(x, data[:,1],'Hr', markerfacecolor='red', marker="o", markeredgecolor="black", markersize=10, markeredgewidth=0.3, alpha=0.7)
ax.plot(x1, fit_1(x1), '--k', x2, fit_2(x2), '--k', linewidth=0.8)
#, x1, B2080_1(x1))
#, '--k',x2, B2080_2(x2), '--k')

ax.set_xlim([25,220])
ax.set_ylim([10,100000])

ax.set_yscale('log')
ax.set_xscale('log',subsx =[2,3,4,5,6,7,8,9])
for axis in [ax.yaxis, ax.xaxis]:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
#ax.minorticks_off()
#ax.set_xticks([400, 600,1000, 2000])"""
ax.set_xticks([40,60,80,100,200])

#ax.legend()"""
plt.title('Power dependence HD6', size=20)
plt.xlabel('Pulse height, [mV]', size=16)
plt.ylabel('Intensity, [a.u.]', size=16)
plt.savefig('C:/Users/Anton.Putintsev/Desktop/HD6(LASING) 20.12.2019/pd_HD6_2.jpg') 
plt.show()
print(ase1[:,1].shape, x.shape)