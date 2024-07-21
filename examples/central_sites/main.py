import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 3 # exponential length
H = hamiltonians.honeycomb(L) # get the Hamiltonian

import matplotlib.pyplot as plt

ic = H.index_around_r()[0] # central index
ics = H.index_around_r(r=H.R[ic],dr=2.1) # central indexes

plt.scatter([H.R[ic][0]],[H.R[ic][1]],c="red",s=80)
xs = [H.R[i][0] for i in ics]
ys = [H.R[i][1] for i in ics]
plt.scatter(xs,ys,c="blue",s=40)
plt.scatter(H.R[:,0],H.R[:,1],c="black",s=20)

plt.axis("equal")

plt.show()

