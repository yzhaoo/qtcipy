import os ; import sys
sys.path.append(os.getcwd()+"/src") # add the path of the library

from qtcipy import install

#install.install() # install the library

install.install_pylibs() # install the Python libraries locally


# for triton, install also a local julia as
# pip install --target YOUR_PATH julia

