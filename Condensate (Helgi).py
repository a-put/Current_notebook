import matplotlib.pyplot as plt
import numpy as np

TotalTime=2000
dt=0.1
Nt=np.int(TotalTime/dt)

""" Define Polariton parameters"""
hbar= 6.626*(10^-34)*(10**3)/(2*np.pi*1.6*10**(-19))*10**12 #       [meV ps]
p_mP = 0.32 # LPB effective mass                                    [meV/cm^2]
p_Gamma = 1/5.5 # POlariton decay rate                              [ps^-1]
p_GammaA = 0.2 # Active exciton reservoir decay rate 
p_GammaI = 0.002 # Inactive exciton reservoir decay rate 
p_alpha = 0.005 # Polariton-polariton interaction strength          [meV um^2]
p_g = 0.01 # Condesate-reservoir interaction strength               [meV um^2]
p_w = 2/(2*np.sqrt(2*np.log(2)))
p_W = 0.05 
p_R = 0.055 
p_noise = 1E-4 
p_P0 = 19

"""Creating the system"""
N=1 # Number of pump spots
N2=100 # Grid in 1d is 2*N2

dx=0.4
dy=0.4
p_a=31*dx

xmax=N2*dx
ymax=N2*dy

x=np.array([x*dx for x in range(-N2,N2)]) 
y=np.array([x*dx for x in range(-N2,N2)])

Mx=np.int(x.size)
My=np.int(y.size)
C=Mx*My # Size of the grid or number of nods

""" Create reciprocal space to calculate Laplaciann in FFT """
kx=np.zeros(Mx)
if Mx % 2==0:
    kx=[(l-Mx/2-1)*np.pi/xmax for l in range(Mx)]
    kx=np.repeat(np.array(kx).reshape(1,2*N2),My,axis=0)    # create My identical rows of x-grid
else:
    kx=[(l-Mx/2-1/2)*np.pi/xmax for l in range(Mx)]
    kx=np.repeat(np.array(kx).reshape(1,2*N2),My,axis=0)

ky=np.zeros(My)
if Mx % 2==0:
    ky=[(l-My/2-1)*np.pi/xmax for l in range(0,My)]
    ky=np.repeat(np.array(ky).reshape(1,2*N2),Mx, axis=0).T
else:
    ky=[(l-My/2-1/2)*np.pi/xmax for l in range(0,My)]
    ky=np.repeat(np.array(ky).reshape(1,2*N2),Mx, axis=0).T

kxmax=np.max(kx)

""" Create non-resonant pump """

x0=0
y0=0
def gauss(x,y,x0,y0):
    wv0=np.dot((np.exp(-(x-x0)**2)).reshape(1,2*N2).T,(np.exp(-(y-y0)**2)).reshape(1,2*N2))/(2*p_w**2)
    return wv0
pump=10000*gauss(x,y,x0,y0)

yy, xx = np.mgrid[slice(-N2*dy, N2*dy, dy), slice(-N2*dx, N2*dx, dx)]

#print(yy.shape)
#plt.pcolormesh(xx,yy,pump)
#plt.show()

""" Polariton dispersion """

Ekp=hbar*(kx**2 + ky**2)/(2*p_mP)
Ekp=np.fft.ifftshift(Ekp)

""" Decaying/damping boundaries in k-space and R-space """

xcut = 0.95*xmax
ycut = 0.85*ymax

p_ratr = 6

Decr=np.dot((1-np.tanh((np.abs(x-xcut))/(xmax/20))).reshape(1,2*N2).T,(1-np.tanh((np.abs(y-ycut))/(ymax/20))).reshape(1,2*N2))
Decr=p_ratr*(1-Decr/np.max(Decr))                            # My x Mx array

kxcut=0.9*kxmax
kr=np.sqrt(kx**2+ky**2)                                   # My x Mx array

p_ratk=10

Deck=p_ratk*(np.tanh((kr-kxcut)/(kxmax/20))+1)              # My x Mx array
Deck=np.fft.ifftshift(Deck)

""" Kinetic part of the Hamiltonian in the SSSFM"""

kinetic_factor=np.exp(-1j*(Ekp-1j*Deck)*dt/2)

""" Initial conditions """

XA=p_noise*np.random.rand(My,Mx)
XI=p_noise*np.random.rand(My,Mx)
psi=p_noise*(np.random.rand(My,Mx)+1j*np.random.rand(My,Mx))
psiq=p_noise*(np.random.rand(My,Mx)+1j*np.random.rand(My,Mx))

ti=0
final=np.zeros(Nt,dtype=object)
final[0]=np.abs(psi)

for k in range(1,Nt):
    n1=np.random.rand(My,Mx)
    n2=np.random.rand(My,Mx)
    XI= XI*np.exp(-(p_GammaI+p_W)*dt)+pump*dt
    XA= XA*np.exp(-(p_GammaA+p_R*np.abs(psi)**2)*dt)+p_W*XI*dt

    #Condensate update

    psi = psi + p_noise*(n1 + 1j*n2)/np.sqrt(2)
    psiq = np.fft.ifft2(kinetic_factor*np.fft.fft2(psi))
    psiq = psiq*np.exp(((p_alpha*np.abs(psiq)**2)/2 + p_g*(XA+XI)/2 + 1j*(p_R*XA - p_Gamma - Decr)/2)*(-1j*dt))
    psi = np.fft.ifft2(kinetic_factor*np.fft.fft2(psiq))

    ti=ti+1
    final[k]=np.abs(psi)

plt.pcolormesh(final[19000])
plt.show()

   
