

def install():
    """Install all the required Julia libraries"""
    import os
    import subprocess
    def get_julia_path():
        result = subprocess.run(['which', 'julia'], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()
    julia_path = get_julia_path()
    # install all the required julia libraries
    path = os.path.dirname(os.path.realpath(__file__))
    os.system(julia_path+" "+path+"/install.jl")



# install required python libraries

import subprocess
import sys
import os

def install_package(package, target_path=None):
    """
    Install a Python package using pip, with an optional target path.
    """
    command = [sys.executable, "-m", "pip", "install", package]
    if target_path:
        command.extend(["--target", target_path])
    try:
        subprocess.check_call(command)
        print(package," installed successfully")
        if target_path:
            print("Installed to target path: {target_path}")
    except subprocess.CalledProcessError as e:
        print("Failed to install",package)



def install_pylibs():
    """Install Python libraries"""
    path = os.path.dirname(os.path.realpath(__file__)) # this location
    pylibpath = path + "/pylib"
    install_package("julia",target_path=pylibpath)




