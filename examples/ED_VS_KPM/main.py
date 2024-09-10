import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 10 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    length = 2**L
    omega = np.pi*2.*np.sqrt(2.)/length*8
    return 1. + 0.2*np.cos(omega*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object

ip = 0

import matplotlib.pyplot as plt

for use_qtci in [True,False]: # with and without QTCI
    for use_kpm in [True,False]: # with and without KPM
        SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object
        SCF.solve(use_qtci=use_qtci,
                use_kpm = use_kpm,
                info_qtci = True,
                maxite = 20,
                delta= 1e-3,
                use_dynamical_qtci = True,
                backend = "C++",
                chiral_AF = True, # use symmetry for chiral models
                ) # solve the SCF
        # now plot the magnetization
        label = ""
        if use_kpm: label += "KPM"
        else: label += "ED"
        if use_qtci: label += " QTCI"
        else: label += " full"
        plt.plot(SCF.H0.R[:,0],SCF.Mz*SCF.U,label=label)


plt.legend()

plt.show()
