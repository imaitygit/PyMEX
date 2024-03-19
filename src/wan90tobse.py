#-----------------------------|
#email: i.maity@imperial.ac.uk|
#Author: Indrajit Maity       |
#-----------------------------|

# To DO 
# + Add the option to always save data files 

import numpy as np
import sys
from generic_func import *
from read_inp import *
from constants import *
from functools import partial
print_f = partial(print, flush=True)
from cyfunc import grad_Hk_opt, Hk_opt

#MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

#|=========================|
#| WANNIER902BSE conversion|
#|=========================|

class WAN2BSE(object):
  """
  Hamiltonian, Coefficients, Structure;
  Everything based on Wannier90 I/O; 
  """
  # data attributes
  def __init__(self, win_f, uk_f, hr_f, wsvec_f, wf_f):
    """
    win_f: Wannier90 input file
    uk_f: Wannier90 generated u matrices
    hr_f: Wannier90 hamiltonian
    wsvec_f: Wannier90 WS cell wrap (R, T vectors) 
    wf_f: Wannier90 output with the location of WF
    """
    self.win_f = win_f
    self.uk_f = uk_f 
    self.hr_f = hr_f
    self.wsvec_f = wsvec_f
    self.wf_f = wf_f


  def get_kgrid(self):
    """
    Returns the k-grid used in the simulations
    An equivalent R-grid in real-space is chosen throughout
    this version.
    """
    f = open(self.win_f, "r")
    lines = f.readlines()
    f.close()

    for i in range(len(lines)):
      if "mp_grid" in lines[i].casefold():
        return np.array([int(lines[i].split()[2]),\
                         int(lines[i].split()[3]),\
                         int(lines[i].split()[4])])
    

  def get_atom(self):
    """
    Reads the Wannier 90 *.wout file and extracts
    the positions of the atoms in angstroms.
    """
    f = open(self.win_f, "r")
    lines = f.readlines()
    f.close()

    # Find the atomic positions
    for i in range(len(lines)):
      if "begin atoms_cart" in lines[i].casefold():
        idx_1 = i+2
      elif "end atoms_cart" in lines[i].casefold():
        idx_2 = i
    atoms = np.empty((idx_2-idx_1, 4), dtype=object)
    for i in range(len(lines)):
      if "begin atoms_cart" in lines[i].casefold():
        for j in range(i+2, i+2+idx_2-idx_1):
          for k in range(4):
            if k == 0:
              atoms[j-i-2,k] = str(lines[j].split()[k])
            else:
              atoms[j-i-2,k] = eval(lines[j].split()[k])
    del lines        
    return atoms


  # Methods
  def map_WF(self):
    """
    An algorithm to guess the WANNIER function location
    based on the atomic positions for all the sites 
    and all the wannier band index. Useful for reducing memory
    """
    # map indices and positions. Details of indices are below
    # 0- atom indices; maximum of this index denotes number of atoms
    # 1-WF starting index for the above atom;
    # 2-WF ending index for the above atom;
    # 3-5-Position of the atoms and WF (atom-centered WF)
    atoms = self.get_atom()
    rloc = self.get_WF_loc()
    map_wf = np.empty((atoms.shape[0],6), dtype=object)

    # Find the WFs for wach atom
    for i in range(atoms.shape[0]):
      # For each atom find the WF centers that fall within
      # 0.5 Angstrom for example; And exploit that multiple
      # Wannier functions could be at the same atom to reduce
      # memory consumption;
      tmp = []
      for j in range(rloc.shape[0]):
        dist = rloc[j]-atoms[i,1:]
        # 0.5 Angstrom: Ad-hoc but should be a reasonable estimate
        # Since we used atom-centered orbitals
        if np.linalg.norm(dist) < 0.5:
          tmp.append(j)
      map_wf[i,1] = np.min(np.array(tmp))
      # +1: Because in python you only loop over 0->n-1
      map_wf[i,2] = np.max(np.array(tmp))+1
      map_wf[i,3:] = atoms[i,1:]
    # Order them based on entries at column 1
    sorted_ind = np.argsort(map_wf[:, 1])
    # Reorder the array based on the sorted indices
    map_wf_r = map_wf[sorted_ind]
    map_wf_r[:,0] = np.arange(atoms.shape[0])
    #if rank == root:
    #  print_f("Before mapping", map_wf)
    #  print_f("Mapped", map_wf_r)
    return map_wf_r


  def get_WF_loc(self):
    """
    Extracts the location of the Wannier functions
    (in Angstroms)
    """
    f = open(self.wf_f, "r")
    lines = f.readlines()
    f.close()

    # Find the number of Wannier Functions
    for i in range(len(lines)):
      if "number of wannier functions" in lines[i].casefold():
        num_wf = int(lines[i].split()[6])

    # The locations of WF in angstroms
    rloc = np.zeros((num_wf, 3), dtype=float)
    for i in range(len(lines)):
      if "initial state" in lines[i].casefold():
        for j in range(i+1, i+1+num_wf):
          for k in range(3):
            # IM: Bug-fixed for (x,y,z) large
            tmp = lines[j].replace("(", "").replace(",", "")
            rloc[j-i-1][k] = tmp.split()[5+k]
    del lines
    return rloc


  # Lattice parameters extraction
  def get_lattice(self):
    """
    Extracts real-space lattice vectors
    """
    f = open(self.win_f, "r")
    lines = f.readlines()
    f.close()

    # Initialize
    A = np.zeros((3, 3), dtype=float) 
    for i in range(len(lines)):
      if "Begin Unit_Cell_Cart" in lines[i]:
        if "bohr" in lines[i+1].casefold():
          for j in range(i+2, i+2+3):
            for k in range(3):
              A[j-i-2][k] = eval(lines[j].split()[k])
          del lines
          return A*Bohr_to_Ang

        elif "ang" in lines[i+1].casefold():
          for j in range(i+2, i+2+3):
            for k in range(3):
              A[j-i-2][k] = eval(lines[j].split()[k])
          del lines
          return A

        else:
          print_f("Unrecognized lattice units in %s file"\
                  %(self.win_f))
          print_f("Exiting...")
          sys.exit()


  # Reciprocal lattice vectors extraction
  def get_reciprocal(self):
    """
    Get reciprocal lattice vectors
    """
    A = self.get_lattice()
    # Since B_i.A_j = 2\pi*\delta_{ij}
    B = 2*np.pi*np.linalg.inv(A.T)
    return B

  
  # get k-points list from win
  def get_kpoints(self):
    """
    K-points lists from the Wannier90 input
    """
    f = open(self.win_f, "r")
    lines = f.readlines()
    f.close()

    # number of k-points: uniform grid
    for i in range(len(lines)):
      if "mp_grid" in lines[i]:
        nk = int(eval(lines[i].split()[2])*\
                 eval(lines[i].split()[3])*\
                 eval(lines[i].split()[4]))

    # k in angstrom -1
    k_c = np.zeros((nk, 3), dtype=float)
    k = np.zeros((nk, 3), dtype=float)
    for i in range(len(lines)):
      if "begin kpoints" in lines[i].casefold():
        for j in range(i+1, i+1+nk):
          for m in range(3):
            k_c[j-i-1][m] = eval(lines[j].split()[m])
          k[j-i-1] = np.dot(k_c[j-i-1], self.get_reciprocal()) 
    del lines
    return k


  def get_bandpath(self, bandpath):
    """
    Band-structure kpoint path
    """
    # only for benchmarking and therefore, 
    # kept hard-coded;
    f = open(bandpath, "r")
    lines = f.readlines()
    f.close()

    # k in crystal coordinates but
    # along high-symmetry paths
    k = np.zeros((len(lines), 3), dtype=float)
    for i in range(len(lines)):
      for j in range(3):
        k[i][j] = eval(lines[i].split()[j])
    del lines
    return k


  # Hamiltonian in real-space
  def get_Hr(self):
    """
    Construction of the Hamitonian (H_n1,n2(r))
    from the hr_f file
    """
    # read the hr_f
    f = open(self.hr_f,"r")
    lines = f.readlines()
    f.close()
  
    # Number of Wannier functions
    N_w = int(lines[1].split()[0])
    # Number of Wigner-Seitz grid points
    N_ws = int(lines[2].split()[0])
    # skip lines
    skip = linecounter(15, N_ws) + 3
    
    # Construct Hamiltonian
    h_r_ab = np.zeros((N_ws, N_w, N_w), dtype=complex)
    r_ws = np.zeros((N_ws, 3), dtype=float)

    # Formatted 
    for i in range(skip, skip+(int(N_ws*N_w**2.)),int(N_w**2)):
      t1 = int((i - skip)/(N_w**2.0))
      for j in range(i, i+int(N_w**2.), N_w):
        for k in range(j, j+N_w):
          t2 = int((j - i)/(N_w))
          t3 = int(k - j)
          h_r_ab[t1][t3][t2] =\
                        float(lines[k].split()[5]) +\
                        1j*float(lines[k].split()[6])
      for l in range(3):
        r_ws[t1][l] = float(lines[k].split()[l])

    del lines
    return r_ws, h_r_ab


  def get_wsvec(self):
    """
    Extract WS-vectors by R+ tilde{R_mnl} as in WANNIER90 docs.
    IM: NEEDS attention of this is correct
    """
    f = open(self.wsvec_f, "r")
    lines = f.readlines()[1:]
    f.close()
 
    r_ws_mod = []
    for i in range(len(lines)):
      if len(lines[i].split()) == 5:
        tmp = lines[i].split()
        tmp_r_ws = np.array([float(tmp[0]), float(tmp[1]),\
                             float(tmp[2])])
        nline = int(lines[i+1].split()[0])
        # Setting nline=1
        # Forcefully, to avoid major rewriting of size aspects
        # WAN90 User manual: Some very minor symmetry breaking
        # during band interpolation may be observed but not 
        # a huge deal breaker;
        for j in range(i+2, i+2+1):
          tmp = lines[j].split()
          tmp_r_ws_ = tmp_r_ws +\
          np.array([float(tmp[0]), float(tmp[1]), float(tmp[2])])
          r_ws_mod.append(tmp_r_ws_)
    del lines  
    return np.array(r_ws_mod) 

  
#  # Hamiltonian in reciprocal space 
#  def get_Hk(self, path=False):
#    """
#    Lattice Fourier transform of the Hamiltonian
#    Python version
#    """
#    A = self.get_lattice()
#    B = self.get_reciprocal()
#    if path == True:
#      k = self.get_bandpath("kpoints")
#    else:
#      k = self.get_kpoints()
#    r_ws, h_r_ab = self.get_Hr()
#    #r_ws = self.get_wsvec()
#    # Initialize for one k
#    h_k_ab = np.zeros((np.shape(k)[0],np.shape(h_r_ab)[1],\
#                       np.shape(h_r_ab)[2]), dtype=complex)
#    for i in range(np.shape(k)[0]):
#      for m in range(np.shape(h_r_ab)[1]):
#        for n in range(np.shape(h_r_ab)[2]):
#          tmp = 0.0
#          for j in range(np.shape(h_r_ab)[0]):
#            r_j = np.matmul(r_ws[j], A)
#            k_j = np.matmul(k[i], B)
#            tmp = tmp + (h_r_ab[j][m][n]* np.exp(1j*\
#                         np.dot(k_j, r_j)))
#          h_k_ab[i][m][n] = tmp
#      #if i%10 == 0:
#      #  print_f("Computed %d k-points"%(i))
#    return h_k_ab       


  def get_Hk(self, path=False):
    """
    H_k cythonised version
    -- Not tested -- 
    """
    A = self.get_lattice()
    B = self.get_reciprocal()
    if path == True:
      k = self.get_bandpath("kpoints")
    else:
      k = self.get_kpoints()

    # Real-space hamiltonain
    r_ws, h_r_ab = self.get_Hr()
    #r_ws = self.get_wsvec()

    # Grad_Hk calculations
    arr = np.zeros((np.shape(k)[0],np.shape(h_r_ab)[1],\
                       np.shape(h_r_ab)[2]), dtype=complex) 
    if rank == root:
      print_f()
      print_f("%d kpoints found"%(k.shape[0]))
      print_f("Computing Gradient analytically")
      print_f()
    h_k_ab = Hk_opt(k, h_r_ab, r_ws,\
                                A, B, arr)
    return np.moveaxis(h_k_ab, 0, -1)


  def get_grad_Hk(self, path=False):
    """
    TO DO
      + Bug fixes for Gamma points
        (getting zero for gamma points)
      + ADD MPI to speed things up
    TESTED [?]
    Derivative of H_k with respect to k vector
    """
    A = self.get_lattice()
    B = self.get_reciprocal()
    if path == True:
      k = self.get_bandpath("kpoints")
    else:
      k = self.get_kpoints()

    # Real-space hamiltonain
    r_ws, h_r_ab = self.get_Hr()
    #r_ws = self.get_wsvec()

    # Grad_Hk calculations
    arr = np.zeros((np.shape(k)[0],np.shape(h_r_ab)[1],\
                       np.shape(h_r_ab)[2]), dtype=complex) 
    # Gamma point only calculations
    if k.shape[0] == 1:
      if rank == root:
        print_f("Gamma point calculations.")
        print_f("Computing Gradient numerically")
      grad_h_k_ab = grad_Hk_opt(k, h_r_ab, r_ws,\
                                    A, B, arr)
    # Multiple k-points can be done efficiently
    else:
      if rank == root:
        print_f()
        print_f("%d kpoints found"%(k.shape[0]))
        print_f("Computing Gradient analytically")
        print_f()
      grad_h_k_ab = grad_Hk_opt(k, h_r_ab, r_ws,\
                                A, B, arr)
    return np.moveaxis(grad_h_k_ab, 0, -1)

  
  def get_bands(self):
    """
    Plot band-structure along a given path
    """
    h_k_ab = self.H_k(path=True)
    E = np.zeros((np.shape(h_k_ab)[0], np.shape(h_k_ab)[1]),\
                  dtype=float)
    k = np.array([i for i in range(np.shape(h_k_ab)[0])])
    for i in range(np.shape(h_k_ab)[0]):
      E[i], eigvec = np.linalg.eigh(h_k_ab[i])
    for i in range(np.shape(h_k_ab)[1]):
      plt.plot(k, E[:,i])
    plt.show()



  def get_C(self, check_unitary=True):
    """
    Reads the U matrices from WANNIER90 code
    The data format in hard-coded here.
    @input
      check_unitary: check if matrices are unitary
                     default is True
    @output
      C_nm_k: Coefficients for LCAO calculations.
    """
    # This particular method loads the file
    # and can be a huge memory bottleneck
    try:
      f = open(self.uk_f, "r") 
      lines = f.readlines()
      f.close()
    except UnicodeDecodeError:
      print_f("%s file format unrecognized"%(self.uk_f))
      print_f("Exiting...")
      sys.exit()

    # Wannier90 output format (strictly followed)
    # nkp: k-points; 
    # nwann: number of Wannier bands;
    # nbnd: number of Bloch bands;
    nkp = eval(lines[1].split()[0])
    nwann = eval(lines[1].split()[1])
    nbnd = eval(lines[1].split()[2])

    # Construct U_k_mn matrices and store kpoints
    kp = np.zeros((nkp, 3))
    U_k_mn = []
    C_k_nm = []

    # loop over k-points
    for i in range(3, len(lines), (nwann*nbnd)+2):
      # k-points
      for j in range(3):
        kp[int(i/((nwann*nbnd)+2))][j] = eval(lines[i].split()[j])
      # Elements of U^k matrices
      # Note: Wannier90 prints column-major, FORTRAN-style
      tmp = []
      for j in range(i+1, (nwann*nbnd)+i+1, 1):
        tmp.append(eval(lines[j].split()[0]) + 1j*\
                   eval(lines[j].split()[1]))
      #if i == 3:
      #  U_0_mn = np.reshape(tmp, (nbnd, nwann), order="F")
      #  print(U_0_mn)
      #  print("Is it unitary: ", is_unitary(U_0_mn))

      if check_unitary == True:
        if is_unitary(np.reshape(tmp, (nbnd, nwann),\
                           order="F")) != True:
          print_f("Matrix is not-unitary.")
          print_f("Problematic Wannierization")
          sys.exit()
      M = np.reshape(tmp, (nbnd, nwann), order="F")
      U_k_mn.append(M)
      C_k_nm.append(np.matrix(M).H)
    U_k_mn = np.array(U_k_mn)
    C_k_nm = np.array(C_k_nm)
    #print_f("size of U_k_mn: ", U_k_mn.shape)
    #print_f("size of C_k_nm: ", C_k_nm.shape)
  
    # Move-axis for BSE code
    C_nm_k = np.moveaxis(C_k_nm, 0, -1)

    del lines
    return C_nm_k  


#print("---OOP---")
#WAN2BSE = WAN2BSE("WSe2.win", "WSe2_u.mat", "WSe2_hr.dat", "WF.wout")
#WAN2BSE.get_grad_Hk()

