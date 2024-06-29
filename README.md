#  qtcipy
Python library to perform quantics tensor cross interpolation. The library
is built on top of [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl?tab=readme-ov-file)

# Examples

The folder examples contains several use cases of the library

Some of the examples use the library for electronic structure [pyqula](https://github.com/joselado/pyqula)

# Installation

You need to have Julia installed in your computer, and the libraries
QuanticsTCI, QuanticsGrids, PyCall that you can be installed with
using Pkg; Pkg.add("QuanticsTCI") ; Pkg.add("PyCall") ; Pkg.add("QuanticsGrids")

Your Python distribution need to have Julia installed, that can be done with
pip install julia

Julia needs to be in your PATH, as the code will use the output of "which julia"

