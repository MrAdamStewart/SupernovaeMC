import numpy as np
import pandas as pd
import scipy.integrate as integrate
import random
import math
import time
import os
import array as arr
from datetime import date

# constants (from Planck 2018)
h = 0.674
hubDist = 3000 / h

# system info
todayDate = date.today()
cwd = os.getcwd()
# module settings
np.set_printoptions(threshold=2000000)
pd.options.display.max_rows = 2000

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
dmbvar = dmb ** 2

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
print('\n' + 'Importing covariance file...')
covarianceFile="sys_full_long.txt"
covarianceData = np.genfromtxt(covarianceFile,unpack=True,skip_header=1)
print('Import successful')
print('Building covariance matrix...')
# diagonalise magnitude errors
dStat = np.diag(dmbvar)

covarianceMatrix = np.reshape(covarianceData, (1048, 1048)) + dStat
print('Build successful')
print('--- Covariance Matrix ---')
print('Dimensions: ''Columns: ' + str(len(covarianceMatrix)) +'  Rows: ' + str(len(covarianceMatrix[0])))

def muModel(z,omegaM,omegaL):
    zz = z
    omegaK = 1 - omegaM - omegaL
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

def likelihood(zcmbArray,omegaM,omegaL):
    #data - modeldata
    muModelData = []
    for i in range(len(zcmbArray)):
        mui = muModel(zcmbArray[i],omegaM,omegaL)
        muModelData.append(mui)
    residuals = mb - muModelData
    #weight residuals using errors
    weight = []
    for i in range(len(dmb)):
        w = dmb[i] / sum(dmb)
        weight.append(w)
    #add scriptm to model
    scriptm = sum(residuals*weight)
    muModelData = muModelData + scriptm
    likeResiduals = np.matrix(mb - muModelData)
    #chi^2
    likeValue = likeResiduals * np.linalg.inv(covarianceMatrix) * likeResiduals.getH()

    return likeValue

#empty arrays to store results
omegaMArr =arr.array('d', [])
omegaLArr = arr.array('d', [])
omegaKArr = arr.array('d', [])
wAppend = arr.array('i', [])
likeArray = arr.array('d', [])
#for counting
count = arr.array('i', [])
count.append(0)

def start(omegaMmin,omegaMmax,omegaLmin,omegaLmax,stepsize,iterations):
    c = 0
    #omegaM = random.uniform(omegaMmin,omegaMmax)
    omegaL = random.uniform(omegaLmin,omegaLmax)
    omegaK = 1 - omegaL
    #omegaMArr.append(round(omegaM,5))
    omegaLArr.append(round(omegaL,5))
    omegaKArr.append(round(omegaK,5))
    weight = 1
    wAppend.append(int(weight))
    print('Random starting point:  L = ' + str(omegaL) + ', K =' + str(omegaK))
    i = -1
    while i < iterations + 1: #algorithm
        t1 = time.clock()
        i = i + 1
        weight = 1
        chiSquared1 = likelihood(zcmbArray,0,omegaL)
        logLike1 = np.exp((- 1 / 2) * chiSquared1)
        likeArray.append(chiSquared1)
        jump(0,omegaL,stepsize) # proposal
        chiSquared2 = float(likelihood(zcmbArray,0,omegaLnew))
        logLike2 = np.exp((- 1 / 2) * chiSquared2)
        likeRatio = logLike2/logLike1
        if likeRatio > 1: #MOVE TO PROPOSAL
            decision='ACCEPT'
            omegaM = 0
            omegaL = omegaLnew
            omegaK = 1 - omegaL
        if likeRatio < 1:
            decision='REJECT/ACCEPT'
            weight = weight + 1 #weight this point
            i = i + 1
            r = random.random() #psuedo-random float [0,1]
            #check ratio with r
            if likeRatio > r: #algorithm has approved proposal
                decision='ACCEPT'
                omegaM = 0
                omegaL = omegaLnew
                omegaK = 1 - omegaL
            else: #rejected proposal again
                    decision='REJECT'
                    weight = weight + 1 #weight this point
                    i = i + 1
        #ending sequence
        #omegaMArr.append(round(omegaM,5))
        omegaLArr.append(round(omegaL,5))
        omegaKArr.append(round(omegaK,5))
        wAppend.append(int(weight))
        t2 = time.clock()
        print('Iteration ' + str(i) + ': ' + ' weight=' + str(weight) + '  time:' + str(round(t2-t1,2))  + '  ' + str(chiSquared1) +  ' OmegaL = ' + str(round(omegaL,3)) + ' OmegaK = ' + str(round(omegaK,3)))

        c = c + 1
        #print('weight=' + str(weight) + 'chi^2=' + str(logLike1))


def jump(omegaM,omegaL,stepsize):
    global omegaMnew
    global omegaLnew
    jumpR = abs(random.gauss(0,stepsize))
    jumpTheta = random.uniform(0,2*math.pi)
    omegaMnew = 0
    omegaLnew = omegaL + jumpR * math.cos(jumpTheta)

    return omegaMnew,omegaLnew

#random start point in between these values
omegaM_min = 0.27
omegaM_max = 0.4
omegaL_min = 0.5
omegaL_max = 0.8

#start sim
iterations = 5000
stepsize = 0.05
t1 = time.clock()
start(omegaM_min,omegaM_max,omegaL_min,omegaL_max,stepsize,iterations)
t2 = time.clock()
print('Time Taken: ' + str(t2 - t1))
#save results
data = np.column_stack((omegaLArr,omegaKArr,wAppend))
outputName = cwd + "/runs/strange_" + str(stepsize) + "_" + str(iterations) + ".txt"
np.savetxt(outputName, data, header='omega_M = 0 , omega_L, omega_K' )
