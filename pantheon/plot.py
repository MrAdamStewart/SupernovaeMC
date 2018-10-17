import matplotlib.pyplot as plt
import numpy as np
import os
import math
import matplotlib.mlab as mlab
import matplotlib.axes as axes
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

print('/n')

cwd = os.getcwd()

## ADJUST THIS
fname = cwd + '/runs/2018-10-16_0.022_11000.txt'
print('file name: ' + fname)
oCDM = True # else L-CDM
burn = 7000

data = np.loadtxt(fname, dtype='float', usecols=(0,1), skiprows=burn)

omegaM = data[:,0]
omegaL = data[:,1]
omegaK = 1 - omegaM - omegaL

if oCDM == True:
    print('Model: o-CDM')
else:
    print('Model: L-CDM')

samples = len(data)
print('Burn = ' + str(burn) + ' Samples = ' + str(samples))

meanM = np.mean(omegaM)
meanL = np.mean(omegaL)
meanK = np.mean(omegaK)

stdM = np.std(omegaM)
stdL = np.std(omegaL)
stdK = np.std(omegaK)

varM = np.var(omegaM)
varL = np.var(omegaL)
varK = np.var(omegaK)

covML = np.cov(omegaM, omegaL)

print('Omega_M: mean = ' + str(round(meanM,5)) + ' std = ' + str(round(stdM,5)) + ' var = ' + str(round(varM,5)))
print('Omega_L: mean = ' + str(round(meanL,5)) + ' std = ' + str(round(stdL,5)) + ' var = ' + str(round(varL,5)))
print('Omega_K: mean = ' + str(round(meanK,5)) + ' std = ' + str(round(stdK,5)) + ' var = ' + str(round(varK,5)))

N = 1000
M = np.linspace(0, 1, N)
L = np.linspace(-1, 2, N)
M, L = np.meshgrid(M,L)

mu = np.array([meanM, meanL])
Sigma = covML

pos = np.empty(M.shape + (2,))
pos[:, :, 0] = M
pos[:, :, 1] = L

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)
normalizedZ = (Z - np.min(Z))/(np.max(Z) - np.min(Z))

plt.figure(1)
plt.contourf(M, L, normalizedZ, [0.05, 0.45, 1], cmap=cm.winter)

x = np.linspace(0,5,10)
y1 = 1 - x
plt.plot(x,y1,'k')

y2 = 0.5 * x
plt.plot(x,y2,'k--')
plt.xlim(0,1)
plt.ylim(0,2)
plt.xlabel("$\Omega_{\mathrm{M}}$")
plt.ylabel("$\Omega_{\Lambda}$")
plt.axes()
plt.tick_params(direction='in')

step = np.arange(0,len(omegaM))

plt.figure(2)
plt.plot(step,omegaM)
plt.title('Random walk')
plt.ylabel("$\Omega_M$")
plt.xlabel('Steps (#)')

plt.figure(3)
plt.plot(step,omegaL)
plt.title('Random walk')
plt.ylabel("$\Omega_{\Lambda}$")
plt.xlabel('Steps (#)')

plt.figure(4)
plt.plot(step,omegaK)
plt.title('Random walk')
plt.ylabel("$\Omega_k$")
plt.xlabel('Steps (#)')

plt.show()
