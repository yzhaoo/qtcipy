import os ; import sys
sys.path.append(os.getcwd()+"/../../src")

from qtcipy.tbscftk import hamiltonians
import numpy as np

L = 12 # exponential length
H = hamiltonians.chain(L) # get the Hamiltonian

import time

H.get_dos_i(kpm_cpugpu="GPU") # run once to precompile

# run on the CPU, with single precission

print("System size,",H.H.shape[0])
print("Timings for single evaluation of KPM")


H.get_dos_i(kpm_prec="single") # run once to precompile
t0 = time.time() # do the timing
es,ds = H.get_dos_i(kpm_prec="single")
t1 = time.time() # do the timing
ts = t1-t0 # store time
print("Single prec. CPU time = ",np.round(t1-t0,4))


# run on the GPU, with double precission

H.get_dos_i(kpm_prec="double") # run once to precompile
t0 = time.time() # do the timing
ed,dd = H.get_dos_i(kpm_prec="double")
t1 = time.time() # do the timing
td = t1-t0
print("Double prec. CPU time = ",np.round(t1-t0,4))



# run on the GPU, with single precission

H.get_dos_i(kpm_cpugpu="GPU") # run once to precompile
t0 = time.time() # do the timing
eg,dg = H.get_dos_i(kpm_cpugpu="GPU")
t1 = time.time() # do the timing
tg = t1-t0
print("GPU time = ",np.round(t1-t0,4))





import matplotlib.pyplot as plt

plt.plot(es,ds,label="single prec. CPU, time="+str(np.round(ts,3)),
        markersize=12,marker="o")
plt.plot(ed,dd,label="double prec. CPU, time="+str(np.round(td,3)),
        markersize=8,marker="o")
plt.plot(eg,dg,label="GPU, time="+str(np.round(tg,3)),
        markersize=4,marker="o")
plt.ylabel("DOS")
plt.xlabel("Energy")
plt.legend()
plt.tight_layout()
plt.savefig("comparison.png")
exit()

plt.show()

