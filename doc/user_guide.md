
# Selfconsistent tight binding calculations

## Creating the Hamiltonian

The Hamiltonian can create from one of the sublibraries as

```python
from qtcipy.tbscftk import hamiltonians
H = hamiltonians.chain(8) # get the Hamiltonian
H = hamiltonians.honeycomb(4,periodic=True) # get the Hamiltonian
```

where the number denotes the lateral size of the system in log scale

## Creating the selfconsistent object

The object to perform selfconsistent calculations can be reated as

```python
SCF = H.get_SCF_Hubbard(U=3.0) # generate a selfconsistent object
```


where U is the value of the Hubbard interaction. By default, the interaction
is taken as zero at the edge of the system.


## Performing a selfconsistent calculation

The selfconsistent mean field can be computed as 

```python
SCF.solve(use_qtci=True,use_kpm = True,chiral_AF = True)
```

The quantics tensor corss interpolation method is activated by setting
```python
use_qtci=True,use_kpm = True.
```

The conventional exact diagonalization mode is set by activating

```python
use_qtci=False,use_kpm = False
```

For systems with chiral symmetry at half filling, acceleration can be activated with

```python
chiral_AF = True
```

### Mixing in a selfconsistent calculation

The mixing between iterations can be controlled with

```python
SCF.solve(mix=0.5)
```

Sometimes the quantics tensor cross may have instabilities in one
iteration. For this reason, the mixing is by default on the
"failsafe" mode, meaning that if it detects that one iteration fluctuates
it goes back close to the previous solution. If you want to use
a conventional mixing, you can activate the option with

```python
SCF.solve(mixing_strategy="plain")
```

## Selfconsisten magnetization

The selfconsistent magnetization can be extracted as 
```python
Mz = SCF.Mz
```

with the positions of sites obtained from
```python
H.R
```


## Computing the non interacting density of states

The density of states in site index can be computed as
```python
(energies,dos) = H.get_dos(i=index,delta=1e-2)
```

where delta is the smearing.

If you want to compute the DOS in a list of sites indexes = [i1,i2,..,iN]
using the stochastic trace method, you can do it as
```python
(energies,dos) = H.get_dos(i=indexes,delta=1e-2,ntries=5)
```

with ntries the number of stochastic vectors tried. In practice,
10 vectors is enough for large enough systems.

## Computing the interacting and interacting density of states

The density of states in site index can be computed as
```python
(energies,dos) = SCF.get_dos(i=index,delta=1e-2)
```

where delta is the smearing.

If you want to compute the DOS in a list of sites indexes = [i1,i2,..,iN]
using the stochastic trace method, you can do it as
```python
(energies,dos) = SCF.get_dos(i=indexes,delta=1e-2,ntries=5)
```

with ntries the number of stochastic vectors tried. In practice,
10 vectors is enough for large enough systems.


## The quantics tensor cross in the mean field

The mean field is parametericed by quantics tensor cross at each step.
By default, the QTCI architecture
is optimized until a certain threshold in the SCF,
and afterwards the same architecture is used to finish the 
selfconsistent loop. After a selfconsistent calculation, the architecture
of the optimized QTCI is stored in 

```python
SCF.qtci_kwargs
```

If you want to switch of the dynamical optimization of the QTCI architecture,
and rather provide fixed values at the beginning, you can do so as

```python
SCF.solve(use_qtci=True,use_kpm = True,mix=0.5,use_dynamical_qtci=False)
```


## Log with the performance of the mean field
After performing a mean-field calculation you may want to see how the
selfconsisstent error, the performance of the QTCI architecture evolved.
You can access those varaibles from the SCF.log dictionary, in particular

```python
SCF.log["SCF_error"] # error in the SCF steps
SCF.log["QTCI_eval"] # fraction of space evaluated by the QTCI
```

## Loading and saving

If you want to save a selfconsistent calculation for later use, you do it as

```python
SCF.save() # error in the SCF steps
```

If you want to load a selfconsistent calculation, you can do it as

```python
SCF = SCF.load() # error in the SCF steps
```

## GPU acceleration

If you want to use GPU acceleration in any part of the code that has it,
you have to give as optional argument to the relevant function

```python
h.get_dos(kpm_cpugpu="GPU")
```

