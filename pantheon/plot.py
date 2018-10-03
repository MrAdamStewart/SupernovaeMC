import matplotlib.pyplot as plt
import numpy as np

omegaM = np.loadtxt('omegaM.txt',dtype=float)
omegaL = np.loadtxt('omegaL.txt',dtype=float)

fig = plt.figure()

print(np.mean(omegaL))
print(np.std(omegaL))
