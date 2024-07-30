import os ; import sys
sys.path.append(os.environ["QTCIPYROOT"])
sys.path.append(os.environ["PYQULAROOT"])


from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 6 # exponential length
H = hamiltonians.honeycomb(L,periodic=True) # get the Hamiltonian


from pyqula.strain import graphene_buckling

omega = np.pi*2./10*np.sqrt(2.) # periodicity

# first moire
fhop0 = graphene_buckling(omega=omega,dt=0.2) # function for buckled lattices
fhop = lambda r,dr: fhop0(r,dr) - 1.0 # redefine
H.modify_hopping(fhop,use_dr = True) # modify the hoppings, once

# second moire
fhop0 = graphene_buckling(omega=omega/7,dt=0.1) # function for buckled lattices
fhop = lambda r,dr: fhop0(r,dr) - 1.0 # redefine
H.modify_hopping(fhop,use_dr = True) # modify the hoppings, again


from qtcipy import parallel

import time


def execute(): # function to execute
    SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object
    SCF.solve(maxite = 1,
        chiral_AF = True,
        use_qtci=True,
        use_kpm=True,
        backend="C++",
        norb = 2,
#        kpm_cpugpu="GPU",
        qtci_maxm = 10,
        qtci_tol = None,
        ) # solve the SCF


t0 = time.time()
parallel.cores = 1
execute()
t1 = time.time()
dt_se = t1-t0



t0 = time.time()
parallel.cores = 2
execute()
t1 = time.time()
dt_pa = t1-t0
print("Time spent in serial execution",dt_se)
print("Time spent in parallel execution",dt_pa)
