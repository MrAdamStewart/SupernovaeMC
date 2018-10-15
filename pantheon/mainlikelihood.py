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


h = 0.7

hubDist = 3000 / h

omegaK = 0 # curvature density

#lumDists = []

def muModel(z,omegaM,omegaL):
    global omegaK
    zz = z
    omegaK = 1 - omegaM - omegaL
    #E =  np.sqrt(omegaM * (1 + zz) ** 3 + omegaK * (1 + zz) ** 2 + omegaL)
    ## integrates 1/E from 0 to z
    #print(1 / E)
    integral = integrate.quad(lambda zz: 1 / (np.sqrt(omegaM * (1 + zz) ** 3 + omegaK * (1 + zz) ** 2 + omegaL)), 0, z)[0]
    comovingDistance = hubDist * integral
    if omegaK == 0:
        comovingDistanceTransverse = comovingDistance
    if omegaK > 0:
        comovingDistanceTransverse = hubDist * 1 / math.sqrt(omegaK) * math.sinh(np.sqrt(omegaK) * comovingDistance / hubDist)
    if omegaK < 0:
        omegaKabs = abs(omegaK)
        comovingDistanceTransverse = hubDist * 1 / math.sqrt(omegaKabs) * math.sin(math.sqrt(omegaKabs) * comovingDistance / hubDist)

    angularDiameterDistance = comovingDistanceTransverse / (1 + z)

    luminosityDistance = (1  + z) ** 2 * angularDiameterDistance

    mu = 5 * np.log10(luminosityDistance * 1e5)

    return mu


#print(muModel(1,0.3,0.7))

def likelihood(zcmbArray,omegaM,omegaL):
    # build model data
    muModelData = []
    for i in range(len(zcmbArray)):
        mui = muModel(zcmbArray[i],omegaM,omegaL)
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
'''

likerange = [0] * 100
for i in range(0,100):
    omM = (0.01)  * i
    loglik = float(likelihood(zcmbArray,omM,0.7))
    likerange[i] = loglik
    print(i)

omegaMrange = np.arange(0,1,0.01)

plt.plot(omegaMrange,likerange,'kx')
plt.savefig('fig1.png')
plt.show()
'''
omegaMx = []
omegaLy = []


def start(omegaMmin,omegaMmax,omegaLmin,omegaLmax,stepsize,iterations):
    omegaM = random.uniform(omegaMmin,omegaMmax)
    omegaL = random.uniform(omegaLmin,omegaLmax)
    print('omegam',omegaM)
    print('omegal',omegaL)
    l = []
    for i in range(1,iterations):
        global like
        global tElapsed

        t1 = time.clock()

        like1 = likelihood(zcmbArray,omegaM,omegaL)
        loglike1 = np.exp((- 1 / 2) * like1)

        jump(omegaM,omegaL,stepsize)

        like2 = likelihood(zcmbArray,omegaMnew,omegaLnew)
        loglike2 = np.exp((- 1 / 2) * like2)

        likeratio = loglike2 / loglike1

        if likeratio > 1:
            like = loglike2
            omegaM = omegaMnew
            omegaL = omegaLnew
        else:
            r = random.uniform(0,1)
            if likeratio < r:
                like = loglike1
                pass
            else:
                like = loglike2
                omegaM = omegaMnew
                omegaL = omegaLnew

        if omegaM < 0:
            omegaM = 0.001

        omegaMx.append(round(omegaM,7))
        omegaLy.append(round(omegaL,7))

        tElapsed = str(round(time.clock() - t1, 2))
        print(str(i) + ':' + '  M:' + str(round(omegaM,5)) + ' L:' + str(round(omegaL,5)) +
        ' K:' + str(round(omegaK,4)) + '    likelihood: ' + str(like[0]))


def jump(omegaM,omegaL,stepsize):
    global omegaMnew
    global omegaLnew
    jumpR = abs(random.gauss(0,stepsize))
    jumpTheta = random.uniform(0,2*math.pi)
    omegaMnew = omegaM + jumpR * math.sin(jumpTheta)
    omegaLnew = omegaL + jumpR * math.cos(jumpTheta)

    return omegaMnew,omegaLnew

iterations = 10000
stepsize = 0.04

#random start point in between these values
omegaM_min = 0.2
omegaM_max = 0.4

omegaL_min = 0.5
omegaL_max = 0.8

start(omegaM_min,omegaM_max,omegaL_min,omegaL_max,stepsize,iterations)

data = np.column_stack((omegaMx,omegaLy))
np.savetxt('densityparameters.txt', data, header='omega_M , omega_L')

###acceptsnce rate ~ 0.3

plt.plot(omegaMx,omegaLy,'kx')
plt.xlim((0,1))
plt.ylim((-1,2))
plt.show()
