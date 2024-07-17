import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 3 # exponential length
H = hamiltonians.square(L) # get the Hamiltonian

# get the SCF object
SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object
SCF.solve(info=True,use_qtci=True,use_kpm=True) # solve the SCF
#SCF.save()
#SCF = SCF.load()
#print("Loading from file")
#SCF.solve(info=True,use_qtci=True,use_kpm=False) # solve the SCF

