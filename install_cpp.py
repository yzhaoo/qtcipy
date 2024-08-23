import os ; import sys
sys.path.append(os.getcwd()+"/src") # add the path of the library

from qtcipy import install


install.install_xfac() # install the Python libraries locally
#install.install_pylibs() # install the Python libraries locally


# for triton, install also a local julia as
# pip install --target YOUR_PATH julia

# you need to install the C++ version from 
# module load cmake
# module load openblas
# git clone https://github.com/tensor4all/xfac
# compile as
# cmake -S . -B build -D CMAKE_BUILD_TYPE=Release -D XFAC_BUILD_PYTHON=ON -D OPENBLAS_PROVIDES_LAPACK=true
# cd build/python
# make

