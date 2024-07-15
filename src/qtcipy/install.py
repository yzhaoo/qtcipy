

def install():
    """Install all the required Julia libraries"""
    from julia import Julia
    Julia(compiled_modules=False)  # Initialize Julia
    from julia import Main
    import os
    import subprocess
    def get_julia_path():
        result = subprocess.run(['which', 'julia'], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8').strip()
    julia_path = get_julia_path()
    os.environ["JULIA_BINDIR"] = julia_path
    # Define a Julia function within Python
    path = os.path.dirname(os.path.realpath(__file__))

    Main.eval(open(path+"/install.jl").read())



