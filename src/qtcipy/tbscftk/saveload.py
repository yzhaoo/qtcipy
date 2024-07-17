import pickle

# function to save and load a Hamiltonian and SCF object


def load_SCF(filename = "SCF.scf"):
    """Function to load an SCF"""
    objs = load(filename) # load this file (a dictionary)
    SCF = dict2SCF(objs) # transform dictionary into an object
    return SCF


def save_SCF(SCF,filename = "SCF.scf"):
    """Function to load an SCF"""
    objs = SCF2dict(SCF) 
    save(objs,filename) # save this file (a dictionary)




def SCF2dict(SCF):
    """Transform an SCF into a dictionary"""
    objs = dict() # initialize
    objs["H0"] = H2dict(SCF.H0) # initial Hamiltonian
    objs["H"] = [SCF.H[0],SCF.H[1]] # mean-field Hamiltonian (matrices)
    objs["U"] = SCF.U
    objs["log"] = SCF.log
    objs["Mz"] = SCF.Mz
    objs["MF"] = SCF.MF
    return objs



def dict2SCF(objs):
    """Create an SCF object from a dictionary"""
    from .scf import SCF_Hubbard
    H0 = dict2H(objs["H0"]) # initial Hamiltonian
    SCF = SCF_Hubbard(H0) # create the object
    SCF.U = objs["U"] # Hubbard
    SCF.H = [objs["H"][0],objs["H"][1]] # mean field Hamil.
    SCF.log = objs["log"]
    SCF.Mz = objs["Mz"]
    SCF.MF = objs["MF"]
    return SCF







def load_H(SCF,filename = "SCF.scf"):
    """Function to save an SCF"""
    objs = load(filename) # load this file (a dictionary)
    H = dict2H(objs) # generate Hamiltonian
    return H


def dict2H(objs):
    """Transform a dictionary into a Hamiltonian"""
    from .hamiltonians import Hamiltonian
    R = objs["R"] # positions
    AB = objs["AB"] # positions
    H = objs["H"] # Hamiltonian
    dim = objs["dim"] # dimension
    Hout = Hamiltonian(dim=dim,H=H,R=R,AB=AB) # Hamiltonian
    return Hout # return Hamiltonian



def H2dict(H):
    """Transform a Hamiltonian into a dictionary"""
    objs = dict() # dictionary
    objs["R"] = H.R # positions
    objs["AB"] = H.AB # positions
    objs["H"] = H.H # Hamiltonian
    objs["dim"] = H.dim  # dimension
    return objs # return dictionary



def load(input_file):
    """Load the hamiltonian"""
    with open(input_file, 'rb') as input:
      return pickle.load(input)



def save(self,output_file):
  """ Write an object"""
  with open(output_file, 'wb') as output:
    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


