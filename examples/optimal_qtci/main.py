import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 12 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    omega = np.pi*2.*np.sqrt(2)/20 # frequency
    return 1. + 0.2*np.cos(omega*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

if False: # if it has to be computed
    SCF.solve(
            info=True,
            use_qtci=True,use_kpm = True,
            info_qtci = True,
            qtci_maxm = 20, # bond dimension to use as initial guess
            backend = "C++",
            qtci_accumulative = True,
            qtci_tol = 1e-2, # error in quantics
            chiral_AF = True, # use symmetry for chiral models
            ) # solve the SCF
    SCF.save()

SCF = SCF.load() # relaod the mean field

from qtcipy import qtcirecipes

kwargs = qtcirecipes.optimal_maxm(SCF.Mz,qtci_error=1e-2)
print("Maxm")
print(kwargs)
kwargs = qtcirecipes.optimal_accumulative(SCF.Mz,qtci_error=1e-2)
print("Accumulative")
print(kwargs)
print("Best flags")
kwargs = qtcirecipes.optimal_qtci(SCF.Mz,qtci_error=1e-2)
print(kwargs)
