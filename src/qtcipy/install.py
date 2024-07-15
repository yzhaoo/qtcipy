

def install():
    """Install all the required Julia libraries"""
    from .juliasession import Main
    Main.eval("using Pkg")
    Main.eval("Pkg.add('QuanticsTCI')")
    Main.eval("Pkg.add('PyCall') ")
    Main.eval("Pkg.add('TensorCrossInterpolation') ")
    Main.eval("Pkg.add('QuanticsGrids')")



