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

SCF = H.get_SCF_Hubbard(U=2.0) # generate a selfconsistent object

from qtcipy import parallel

parallel.cores = 2

SCF.solve(info=True,
        maxite = 1,
        chiral_AF = True,
        info_qtci=True,
        use_qtci=True,
        use_kpm=True,
        backend="C++",
        norb = 2,
        qtci_maxm = 10,
        qtci_tol = None,
        ) # solve the SCF



