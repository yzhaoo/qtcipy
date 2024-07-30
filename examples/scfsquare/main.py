import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 5 # exponential length
H = hamiltonians.square(L,periodic=True) # get the Hamiltonian

# get the SCF object

def fhop(r):
    omega = np.pi*2.*np.sqrt(2)/8
    return 0.2*(np.cos(omega*r[0]) + np.cos(omega*r[1]))

H.modify_hopping(fhop)
#H.dim = 2
SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object
SCF.solve(info=True,use_qtci=True,
        info_qtci=True,
        use_kpm=True,
        backend="Julia"
        ) # solve the SCF
#SCF.save()
#SCF = SCF.load()
#print("Loading from file")
#SCF.solve(info=True,use_qtci=True,use_kpm=False) # solve the SCF

