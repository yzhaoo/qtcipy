import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 16 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

# get the SCF object
SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object
H.add_onsite(lambda r: np.cos(2*np.pi*r[0]*np.sqrt(2)))
out = SCF.estimate_time()
print("Factor of evaluations CPU",out[1])
print("Time in quantics CPU",out[0])


out = SCF.estimate_time(kpm_cpugpu="GPU")
print("Factor of evaluations GPU",out[1])
print("Time in quantics GPU",out[0])


