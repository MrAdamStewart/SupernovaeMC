import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
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
print('\n' + 'Building covariance matrix...')
# diagonalise magnitude errors
dStat = np.diag(dmb)
# C = cSys + dStat (covmat)
covarianceMatrix = np.reshape(covarianceData, (1048, 1048)) + dStat

print('Build successful')

# Display matrix info
print('\n' + 'Covariance Matrix')
print('Dimensions: ''Columns: ' + str(len(covarianceMatrix)) +'  Rows: ' + str(len(covarianceMatrix[0])))
'''
Model generation
'''

H = 70 #H0
c = 3e8 #so
hubDist = (c / H) * 10 ** -3

omegaM = 0.3 #matter density
omegaL = 0.7 # lambda density
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


def modelGenerate(zcmbArray,omegaM,omegaL,omegaK):
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
    print('\n')
    print('scriptm:',scriptm)
    # add scriptm to model
    muModelData = muModelData + scriptm

    # final residuals with scriptm
    likeResiduals = np.matrix(mb - muModelData)
    # likelihood
    likeValue = likeResiduals * np.linalg.inv(covarianceMatrix) * likeResiduals.getH()
    print('likelihood:', round(float(likeValue),4))

modelGenerate(zcmbArray,0.3,0.7,0)

###acceptsnce rate ~ 0.3
