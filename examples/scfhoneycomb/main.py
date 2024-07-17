import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np
import matplotlib.pyplot as plt

L = 3 # exponential length
H = hamiltonians.honeycomb(L,periodic=True) # get the Hamiltonian
exit()

# get the SCF object
SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object
#SCF.solve(info=True,use_qtci=True,use_kpm=True) # solve the SCF

U = 2.0 # interaction
SCF = H.get_SCF_Hubbard(U=U) # get the SCF object

R,M = SCF.H0.R,SCF.H0.H # position and hopping

M = M.todense()

for i in range(M.shape[0]):
    for j in range(M.shape[0]):
        t = M[i,j]
        if abs(t)>0.:
            plt.plot([R[i,0],R[j,0]],[R[i,1],R[j,1]],c="black",marker='o')
plt.axis("equal")

plt.show()
