import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 10 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian
#d = H.get_density_i()
SCF = H.get_SCF_Hubbard(U=2.0) # get the SCF object
#es,ds = H.get_dos_i(w=np.linspace(-5.,5.,1000),i=2**L//2)
#SCF.solve()
es,ds = SCF.get_dos_i(w=np.linspace(-5.,5.,1000),i=2**L//2,delta=3e-2)
#print(d)


import matplotlib.pyplot as plt

plt.plot(es,ds)

plt.show()



exit()
H.add_onsite(lambda r: np.cos(0.01*r[0]))
SCF = H.get_SCF_Hubbard(U=1.0) # get the SCF object
SCF = SCF.solve(info=True,use_qtci=False,use_kpm=True) # solve the SCF problem
print(SCF.log["SCF_error"])
