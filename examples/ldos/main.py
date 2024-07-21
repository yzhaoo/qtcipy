import os ; import sys
sys.path.append(os.getcwd()+"/../../src")
sys.path.append(os.environ["PYQULAROOT"])

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 14 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

d = [H.get_ldos(e=1.95,i=i) for i in range(H.H.shape[0])]
dc = H.get_ldos(e=1.95) # compute all with quantics

import matplotlib.pyplot as plt

plt.plot(range(len(d)),d,label="explicit")
plt.scatter(range(len(dc)),dc,c="red",label="QTC")
plt.legend()
plt.xlabel("Site")
plt.ylabel("LDOS")

plt.tight_layout()
plt.show()
