
## use the local Julia if needed

import os
try: # try using global Julia
#    import julia as jl
    from julia import Julia
except:
    path = os.path.dirname(os.path.realpath(__file__))
    import sys
    sys.path.append(path+"/pylib") # add the local julia
    print("Using the local Julia library for Python")
    import julia as jl






from julia import Julia
Julia(compiled_modules=False)  # Initialize Julia

from julia import Main

import subprocess

def get_julia_path():
    result = subprocess.run(['which', 'julia'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').strip()

julia_path = get_julia_path()


os.environ["JULIA_BINDIR"] = julia_path



# Define a Julia function within Python
path = os.path.dirname(os.path.realpath(__file__))


# now execute the Julia code
Main.eval(open(path+"/interpolate.jl").read())


def restart():
    global Main
    del Main # remove this
    from julia import Main
    print("Julia session will be restarted")
    Main.eval(open(path+"/interpolate.jl").read())




# this may make the code stable
os.environ["PYTHONFAULTHANDLER"] = "0"



