import matplotlib .pyplot as plt
import numpy as np
import os
import math
import matplotlib.mlab as mlab
import matplotlib.axes as axes
from matplotlib import cm

print('/n')

cwd = os.getcwd()

## ADJUST THIS
fname = cwd + '/runs/strange_0.05_5000.txt'
print('file name: ' + fname)
oCDM = True # else L-CDM
burn = 200

data = np.loadtxt(fname, dtype='float', usecols=(0,1,2), skiprows=burn)

OmegaK = data[:,1]
OmegaL = data[:,0]
weight = data[:,2]

weightedOmegaK = OmegaK * weight
weightedOmegaL = OmegaL * weight

mode = 'strangeCDM'
print('Starting model: strangeCDM')

samples = len(data)
print('Burn = ' + str(burn) + ' Samples = ' + str(samples))

meanM = 0
meanL = sum(weightedOmegaL) / sum(weight)
meanK =sum(weightedOmegaK) / sum(weight)

stdM = 0
stdL = np.std(OmegaL)
stdK = np.std(OmegaK)

varM = 0
varL = np.var(OmegaL)
varK = np.var(OmegaK)

covML = 0
covKL = np.cov(weightedOmegaL,weightedOmegaK)

print('Omega_M: mean =  0  std = 0')
print('Omega_L: mean = ' + str(meanL) + ' std = ' + str(stdL))
print('Omega_K: mean = ' + str(meanK) + ' std = ' + str(stdK))

N = 1000
M = 0
L = np.linspace(-1, 3, N)
K = np.linspace(-1,1,N)
K, L = np.meshgrid(K,L)

mu = np.array([meanK, meanL])
Sigma = covKL

pos = np.empty(K.shape + (2,))
pos[:, :, 0] = K
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
plt.contourf(K, L, normalizedZ, [0.05, 0.45, 1], cmap=cm.winter)

x = np.linspace(0,5,10)
y1 = 1 - x
plt.plot(x,y1,'k')

y2 = 0.5 * x
plt.plot(x,y2,'k--')
plt.xlim(0,1.6)
plt.ylim(0,2.4)
plt.title('$o$CDM Constraints')
plt.xlabel("$\Omega_{\mathrm{c}}$")
plt.ylabel("$\Omega_{\Lambda}$")
plt.axes()

step = np.arange(0,len(OmegaK))
'''
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
'''
