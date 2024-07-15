import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 6 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

# get the SCF object
SCF = H.get_SCF_Hubbard(U=2.0)
SCF.solve(info=True,use_qtci=True,use_kpm=True)

