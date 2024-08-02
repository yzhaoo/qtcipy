import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 10 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    omega = np.pi*2.*np.sqrt(2.)/20
    return 1. + 0.2*np.cos(omega*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object

#m = SCF.estimate_qtci_maxm(f)
#m = 10
#print(m) ; exit()


#SCF.solve(info=True) # solve the SCF
SCF.solve(info=True,
        use_qtci=True,use_kpm = True,
        info_qtci = True,
#        maxite = 1,
        qtci_maxm = 2, # bond dimension to use as initial guess
#        maxite = 1,
        backend = "C++",
#        qtci_recursive = True, # use a recursive mode, to enforce tol
#        qtci_pivot1 = 3,
#        qtci_fullPiv = False,
#        qtci_accumulative = True,
#        qtci_tol = 1e-2, # error in quantics
        chiral_AF = True, # use symmetry for chiral models
        ) # solve the SCF

#print(SCF.qtci_kwargs)
SCF.save()
SCF = SCF.load()

print(SCF.qtci_kwargs)

#print("Loading from file")
#SCF.solve(info=True,use_qtci=True,use_kpm=False) # solve the SCF

