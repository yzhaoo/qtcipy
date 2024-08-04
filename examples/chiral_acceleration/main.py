import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 4 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    return 1. + 0.2*np.cos(np.pi*2.*np.sqrt(2.)*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])

H.modify_hopping(f)

# generate two scf objects
U = 4.0
SCF_ED = H.get_SCF_Hubbard(U=U) # generate a selfconsistent object
SCF_QT = H.get_SCF_Hubbard(U=U) # generate a selfconsistent object

SCF_ED.solve(use_qtci=False,use_kpm = False,info=True) # full solution
SCF_QT.solve(use_qtci=False,use_kpm = True,
#        qtci_maxm=10,
        info=True,
        backend = "C++",
        delta=1e0,
#        qtci_tol = 1e-2,
        chiral_AF = True, # use symmetry for chiral models
        ) # solve the SCF

x = range(H.H.shape[0])
import matplotlib.pyplot as plt

plt.scatter(x,SCF_ED.Mz,label="ED")
plt.scatter(x,SCF_QT.Mz,label="QTCI_KPM")
plt.legend()
plt.show()


