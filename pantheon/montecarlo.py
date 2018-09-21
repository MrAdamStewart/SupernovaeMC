import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

## remove runtime warning for model
np.seterr(divide='ignore')

lightData="lcparam_full_long.txt"
name = np.loadtxt(lightData,dtype=str,usecols=0)
zcmb = np.loadtxt(lightData,dtype=float,usecols=1)
zhel = np.loadtxt('lcparam_full_long.txt', dtype=float, usecols=2)
dz = np.loadtxt('lcparam_full_long.txt', dtype=float, usecols=3)
mb = np.loadtxt('lcparam_full_long.txt', dtype=float, usecols=4)

H = 70 #H0
c = 3e8 #so
dH = (c / H) * 10 ** -3

#dH = (3/7) * 10 ** 4 #hubble distance in Mpc


omegaM = 0.28 #matter density
omegaL = 0.72 # lambda density
omegaK = 0 # curvature density


def int():
    ##integrates 100 times over z from 0 to 1
    ## returns array of integral values
    zz = 0
    eX = []
    while zz < 100:
        z = zz/1000
        E =  np.sqrt(omegaM * (1 + z) ** (3) + (omegaK * (1 + z) ** 2) + omegaL)
        ## integrates 1/E from 0 to z
        int = integrate.quad(lambda z: E ** (-1), 0, z)[0]
        zz+= 1
        eX.append(int)
    return eX

## call integral function
integrand = int()
## reproduce z values used in function
z = np.arange(0,1,0.01)
## luminosity distance = hubble distance * (1 + z) * integrand
dL = (dH) * (1 + z) * integrand
##distance modulus = 5 * log10(dL/10pc)
mu = 5 * np.log10(dL * 10 ** 5)
#model plot
plt.plot(z, mu, 'rx')
#plt.plot(z,dL,'rx')
#data plot
plt.plot(zcmb,mb,'bx')
plt.show()
