

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



