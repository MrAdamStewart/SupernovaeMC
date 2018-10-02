import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import random
import math
import time



# module settings
np.set_printoptions(threshold=2000000)
pd.options.display.max_rows = 2000

'''
Data and covariance matrix importing
'''


# There has to be a simpler way to do this #
###DATA###
lightData="lcparam_full_long.txt"
print('Data: Pantheon')
print('Data File: ' + lightData)
print('\n' + 'Importing observational data...')

name = np.loadtxt(lightData,dtype=str,usecols=0)
zcmb = np.loadtxt(lightData,dtype=float,usecols=1)
zcmbArray = np.asarray(zcmb)
zhel = np.loadtxt(lightData, dtype=float, usecols=2)
dz = np.loadtxt(lightData, dtype=float, usecols=3)
mb = np.loadtxt(lightData, dtype=float, usecols=4)
dmb = np.loadtxt(lightData, dtype=float, usecols=5)
x1 = np.loadtxt(lightData, dtype=float, usecols=6)
dx1 = np.loadtxt(lightData, dtype=float, usecols=7)
color = np.loadtxt(lightData, dtype=float, usecols=8)
dcolor = np.loadtxt(lightData, dtype=float, usecols=9)
print('Import successful')

# Build pd data frame
print('\n' + 'Building data frame...')
snMatrix = np.column_stack((name, zcmb, zhel, dz, mb, dmb, x1, dx1, color, dcolor))
snDataArray = pd.DataFrame(snMatrix,index=np.arange(1,1049),
    columns=('Name','zCMB','zHel','dz','mb','dmb','x1','dx1','color','dcolor'))
print('Build successful')

# Display data frame
print('\n' + 'Total Samples: ' + str(len(snDataArray)))
print('\n')
print(snDataArray)

###COVARIANCE###

# Data file is vector
print('\n' + 'Importing covariance file...')
covarianceFile="sys_full_long.txt"
covarianceData = np.genfromtxt(covarianceFile,unpack=True,skip_header=1)
print('Import successful')

# Reshape to matrix
print('Building covariance matrix...')
# diagonalise magnitude errors
dStat = np.diag(dmb)
# C = cSys + dStat (covmat)
covarianceMatrix = np.reshape(covarianceData, (1048, 1048)) + dStat

print('Build successful')

# Display matrix info
print('--- Covariance Matrix ---')
print('Dimensions: ''Columns: ' + str(len(covarianceMatrix)) +'  Rows: ' + str(len(covarianceMatrix[0])))
'''
Model generation
'''

H = 70 #H0
c = 3e8 #so
hubDist = (c / H) * 10 ** -3

omegaK = 0 # curvature density

#lumDists = []

def muModel(z,omegaM,omegaL,omegaK):
    E =  np.sqrt(omegaM * (1 + z) ** (3) + (omegaK * (1 + z) ** 2) + omegaL)
    ## integrates 1/E from 0 to z
    integrand = integrate.quad(lambda z: E ** (-1), 0, z)[0]
    ## luminosity distance = hubble distance * (1 + z) * integrand
    lumDist = hubDist * (1 + z) * integrand
    #lumDists.append(lumDist)
    mu = 5 * np.log10(lumDist * 10 ** 5)

    return mu


def likelihood(zcmbArray,omegaM,omegaL,omegaK):
    # build model data
    muModelData = []
    for i in range(len(zcmbArray)):
        mui = muModel(zcmbArray[i],omegaM,omegaL,omegaK)
        muModelData.append(mui)
    # initial residuals for model
    residuals = mb - muModelData
    # weight of residuals using errors
    weight = []
    for i in range(len(dmb)):
        w = dmb[i] / sum(dmb)
        weight.append(w)
    # def scriptm
    scriptm = sum(residuals*weight)
    # add scriptm to model
    muModelData = muModelData + scriptm
    # final residuals with scriptm
    likeResiduals = np.matrix(mb - muModelData)
    # likelihood value
    likeValue = likeResiduals * np.linalg.inv(covarianceMatrix) * likeResiduals.getH()
    #print('likelihood:', round(float(likeValue),4))

    return likeValue


omegaMmin = 0
omegaMmax = 1.5
omegaLmin = 0
omegaLmax = 2
stepsize = 0.01

omegaMx = []
omegaLy = []


def start(omegaMmin,omegaMmax,omegaLmin,omegaLmax,stepsize,iterations):
    omegaM = random.uniform(omegaMmin,omegaMmax)
    omegaL = random.uniform(omegaLmin,omegaLmax)
    print('omegam',omegaM)
    print('omegal',omegaL)
    l = []
    for i in range(0,iterations):
        global like
        global tElapsed

        t1 = time.clock()

        omegaMx.append(round(omegaM,7))
        omegaLy.append(round(omegaL,7))

        if omegaM < 0:
            omegaM = 0.001
        if omegaL <  0:
            omegaL = 0.001 #... no negative densities.#

        like1 = likelihood(zcmbArray,omegaM,omegaL,omegaK)

        jump(omegaM,omegaL,stepsize)

        like2 = likelihood(zcmbArray,omegaMnew,omegaLnew,omegaK)

        l.append(like1)
        l.append(like2)

        if l[i] < l[i+1]:
            omegaM = omegaMnew
            omegaL = omegaLnew
        else:
            r = random.uniform(0,1)
            if r > l[i+1]/l[i]:
                pass
            else:
                omegaM = omegaMnew
                omegaL = omegaLnew

        if like2 < like1:
            like = like2
        else:
            like = like1

        tElapsed = str(round(time.clock() - t1, 2))

        print('Iteration: ' + str(i) + '     Time Taken: ' + tElapsed)


def jump(omegaM,omegaL,stepsize):
    global omegaMnew
    global omegaLnew
    jumpR = abs(random.gauss(0,stepsize))
    jumpTheta = random.uniform(0,2*math.pi)
    omegaMnew = omegaM + jumpR * math.sin(jumpTheta)
    omegaLnew = omegaL + jumpR * math.cos(jumpTheta)

    return omegaMnew,omegaLnew


iterations = 10000
stepsize = 0.02

#random start point in between these values
omegaM_min = 0
omegaM_max = 1

omegaL_min = 0
omegaL_max = 1

start(omegaMmin,omegaM_max,omegaL_min,omegaL_max,stepsize,iterations)

data = np.column_stack((omegaMx,omegaLy))
np.savetxt('densityparameters.txt', data, header='omega_M , omega_L', comments='#stepsize 0.02')

###acceptsnce rate ~ 0.3

plt.plot(omegaMx,omegaLy,'kx')
plt.show()
