import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 6 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

def f(r):
    return 1. + 0.2*np.cos(np.pi*2.*np.sqrt(2.)*r[0]) #+0.2*np.cos(np.pi*2.*np.sqrt(3.)*r[0])
H.modify_hopping(f)

# get the SCF object
SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object

#m = SCF.estimate_qtci_maxm(f)
#m = 10
#print(m) ; exit()


#SCF.solve(info=True) # solve the SCF
SCF.solve(info=True,use_qtci=True,
#        qtci_maxm = m,
        use_kpm=True) # solve the SCF
#SCF.save()
#SCF = SCF.load()
#print("Loading from file")
#SCF.solve(info=True,use_qtci=True,use_kpm=False) # solve the SCF

