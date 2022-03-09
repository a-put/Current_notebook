import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal
from scipy import fftpack
import seaborn as sns

dx=0.4
dy=0.4
N2=100
x=np.array([x*dx for x in range(-N2,N2)]) 
y=np.array([x*dx for x in range(-N2,N2)])


x0=0
y0=0
sigma = 30/(2*np.sqrt(2*np.log(2)))
def gauss(x,y,x0,y0):
    wv0=np.dot((np.exp(-(x-x0)**2/(2*sigma**2))).reshape(1,2*N2).T,(np.exp(-(y-y0)**2/(2*sigma**2))).reshape(1,2*N2))/(2*np.pi*sigma**2)
    return wv0
pump=10*gauss(x,y,x0,y0)

yy, xx = np.mgrid[slice(-N2*dy, N2*dy, dy), slice(-N2*dx, N2*dx, dx)]

plt.pcolormesh(xx,yy,pump)
plt.show()