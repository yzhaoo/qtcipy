import os ; import sys
sys.path.append(os.getcwd()+"/src") # add the path of the library

from qtcipy import install

install.install() # install the julia library
