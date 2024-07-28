import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 20 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

import time
H.get_dos_i(i=100)
t0 = time.time()
H.get_dos_i(i=100)
t1 = time.time()
print("Time",t1-t0)



