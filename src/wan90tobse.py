#-----------------------------|
#email: i.maity@imperial.ac.uk|
#Author: Indrajit Maity       |
#-----------------------------|

# To DO 
# + Add the option to always save data files 

import numpy as np
import sys
from generic_func import *
#from read_inp import *
from constants import *
from functools import partial
from collections import namedtuple
print_f = partial(print, flush=True)

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
    Reads the Wannier90 *.win file and extracts
    the positions of the atoms in angstroms.

    If lattice is provided, fractional coordinates
    (begin atoms_frac) are converted to Cartesian.
    """
    with open(self.win_f, "r") as f:
      lines = f.readlines()

    atoms = []

    i = 0
    while i < len(lines):
      line_lower = lines[i].casefold()

      # Atoms in Cartesian coordinates
      if "begin atoms_cart" in line_lower:
        i += 2
        while i < len(lines) and "end atoms_cart" not in lines[i].casefold():
          parts = lines[i].split('!')[0].split()
          if len(parts) >= 4:
            atom_name = parts[0]
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            atoms.append([atom_name] + coords)
          i += 1

      # Atoms in fractional coordinates
      # -------------------------------
      elif "begin atoms_frac" in line_lower:
        lattice = self.get_lattice()
        i += 1
        while i < len(lines) and "end atoms_frac" not in lines[i].casefold():
          parts = lines[i].split('!')[0].split()
          if len(parts) >= 4:
            atom_name = parts[0]
            frac_coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            cart_coords = frac_coords @ lattice
            atoms.append([atom_name] + list(cart_coords))
          i += 1
      else:
        i += 1

    del lines
    return np.array(atoms, dtype=object)


  def map_WF(
    self,
    using_mlwf: bool = False,
    mode: str = "atom_centered",
    cutoff: float = 0.5,
  ):
    """
    Map Wannier functions to spatial centers.

    Parameters
    ----------
    using_mlwf : bool
      Whether MLWF positions are used.
    mode : str
      'atom_centered' | 'nearest_atom' | 'wf_centered'
    cutoff : float
      Distance cutoff for atom_centered mode (Angstrom).
    """
    atoms = self.get_atom()              # (Nat, 4) [id, x, y, z]
    wf_pos = self.get_WF_loc(using_mlwf) # (Nwf, 3)

    if mode == "atom_centered":
      return self._map_atom_centered(atoms, wf_pos, cutoff)

    elif mode == "nearest_atom":
      return self._map_nearest_atom(atoms, wf_pos)

    elif mode == "wf_centered":
      return self._map_wf_centered(wf_pos)

    else:
      raise ValueError(f"Unknown WF mapping mode: {mode}")


  def _map_wf_centered(self, wf_pos):
    """
    Map Wannier functions treating each WF as its own center.

    Output format per row:
      0 : sequential WF center index
      1 : WF start index
      2 : WF end index (exclusive)
      3 : x position
      4 : y position
      5 : z position
    """
    Nwf = wf_pos.shape[0]
    map_wf = np.empty((Nwf, 6), dtype=object)

    for i in range(Nwf):
      map_wf[i, 0] = i          # WF center index
      map_wf[i, 1] = i          # WF start index
      map_wf[i, 2] = i + 1      # WF end index (exclusive)
      map_wf[i, 3:] = wf_pos[i] # WF coordinates

    # Safety check: WF start indices must increase monotonically
    assert np.all(np.diff(map_wf[:, 1].astype(int)) > 0), \
      "WF start indices are not strictly increasing!"

    return map_wf


  def _map_atom_centered(self, atoms, wf_pos, cutoff):
    """
    Map Wannier functions to atoms (atom-centered WFs).

    Output format per row:
      0 : atom index
      1 : WF start index
      2 : WF end index (exclusive)
      3 : x position
      4 : y position
      5 : z position
    """
    map_wf = np.empty((atoms.shape[0], 6), dtype=object)

    for i in range(atoms.shape[0]):
      tmp = []
      for j in range(wf_pos.shape[0]):
        dist = wf_pos[j] - atoms[i, 1:]
        if np.linalg.norm(dist) < cutoff:
          tmp.append(j)

      if len(tmp) == 0:
        raise RuntimeError(
          f"No Wannier functions found near atom {i} within cutoff {cutoff}"
        )

      tmp = np.array(tmp)
      map_wf[i, 1] = tmp.min()
      map_wf[i, 2] = tmp.max() + 1
      map_wf[i, 3:] = atoms[i, 1:]

    # Order rows by WF start index
    sorted_ind = np.argsort(map_wf[:, 1])
    map_wf_r = map_wf[sorted_ind]
    map_wf_r[:, 0] = np.arange(atoms.shape[0])

    return map_wf_r


  def _map_nearest_atom(self, atoms, wf_pos):
    """
    Map Wannier functions to the nearest atom.

    Output format per row:
      0 : atom center index
      1 : WF start index
      2 : WF end index (exclusive)
      3 : x position
      4 : y position
      5 : z position
    """
    atoms_xyz = atoms[:, 1:].astype(float)
    wf_pos = wf_pos.astype(float)

    map_wf = np.empty((atoms.shape[0], 6), dtype=object)

    for i in range(atoms.shape[0]):
      tmp = []
      for j in range(wf_pos.shape[0]):
        dist = wf_pos[j] - atoms_xyz[i]
        # Assign WF j to the nearest atom i if it's the closest one
        # Compare distance to all atoms
        dists_to_all_atoms = np.linalg.norm(wf_pos[j] - atoms_xyz, axis=1)
        if np.argmin(dists_to_all_atoms) == i:
          tmp.append(j)

      if len(tmp) == 0:
        raise RuntimeError(
          f"No Wannier functions assigned to atom {i} as nearest atom."
        )

      tmp = np.array(tmp)
      map_wf[i, 1] = tmp.min()
      map_wf[i, 2] = tmp.max() + 1
      map_wf[i, 3:] = atoms_xyz[i]

    # Order rows by WF start index
    sorted_ind = np.argsort(map_wf[:, 1])
    map_wf_r = map_wf[sorted_ind]
    map_wf_r[:, 0] = np.arange(atoms.shape[0])

    return map_wf_r


  def get_WF_loc(self, using_mlwf: bool = False):
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
    if using_mlwf:
      for i in range(len(lines)):
        if "final state" in lines[i].casefold():
          for j in range(i+1, i+1+num_wf):
            for k in range(3):
              # IM: Bug-fixed for (x,y,z) large
              tmp = lines[j].replace("(", "").replace(",", "")
              rloc[j-i-1][k] = tmp.split()[5+k]
      del lines
    else:
      for i in range(len(lines)):
        if "initial state" in lines[i].casefold():
          for j in range(i+1, i+1+num_wf):
            for k in range(3):
              # IM: Bug-fixed for (x,y,z) large
              tmp = lines[j].replace("(", "").replace(",", "")
              rloc[j-i-1][k] = tmp.split()[5+k]
      del lines      
    return rloc


  def get_lattice(self):
    """
    Extracts real-space lattice vectors
    """
    with open(self.win_f, "r") as f:
      lines = f.readlines()

    A = np.zeros((3, 3), dtype=float)
    for i, line in enumerate(lines):
      if "unit_cell_cart" in line.lower():
        j = i + 1
        while j < len(lines) and not lines[j].strip():
          j += 1
        units_line = lines[j].strip().lower()

        if "bohr" in units_line:
          factor = Bohr_to_Ang
        elif "ang" in units_line:
          factor = 1.0
        else:
          print_f(f"Unrecognized lattice units in {self.win_f}")
          print_f(f"Exiting...")
          comm.Abort(1)

        # Read the next 3 lines as lattice vectors
        vec_count = 0
        k = j + 1
        while vec_count < 3 and k < len(lines):
          line_clean = lines[k].split('!')[0].strip()
          parts = line_clean.split()
          if len(parts) >= 3:
            A[vec_count] = [float(parts[0]), float(parts[1]), float(parts[2])]
            vec_count += 1
          k += 1

        del lines
        return A * factor

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

  # get k-points_reciprocal list from win
  def get_kpoints_reciprocal(self):
    """
    K-points lists from the Wannier90 input
    (in reciprocal space)
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
    for i in range(len(lines)):
      if "begin kpoints" in lines[i].casefold():
        for j in range(i+1, i+1+nk):
          for m in range(3):
            k_c[j-i-1][m] = eval(lines[j].split()[m]) 
    del lines
    return k_c    


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
    return r_ws, h_r_ab


  def get_Rimproved(self):
    """
    Clean namedtuple approach with float64
    """
    VectorData = namedtuple('VectorData', ['R_base', 'T_vec'])
    with open(self.wsvec_f) as f:
      lines = f.readlines()[1:]

    data = []
    i = 0
    while i < len(lines):
      parts = lines[i].split()
      if len(parts) == 5:
        R_base = np.array(parts[:5], dtype=np.float64)
        nline = int(lines[i+1])
        T_vec = np.array([lines[i+2+j].split()[:3] for j in range(nline)], dtype=np.float64)
        data.append(VectorData(R_base, T_vec))
        i += 2 + nline
      else:
        i += 1
    return data    


  def get_Hk_and_grad_Hk(self, method="dft",
                               kpt = None,
                               minimum_dist_replica=True,
                               Ez=None,
                               map_wf=None):
    """
    H_k pure python 
    """
    A = self.get_lattice()
    B = self.get_reciprocal()
    if method == "dft":
      k = self.get_kpoints()
    elif method == "tb":
      if kpt is None:
        print_f(f"No kpt found for rank: {rank}!")
        comm.Abort(1)
      if kpt.ndim == 1:
        k = np.array([kpt])

    # Real-space hamiltonain
    r_ws, h_r_ab = self.get_Hr()
    num_k = len(k) 
    num_r = h_r_ab.shape[0]; num_a = h_r_ab.shape[1] 
    num_b = h_r_ab.shape[2]
    h_k_ab = np.zeros((num_k, num_a, num_b), dtype=complex)
    grad_h_k_ab = np.zeros((num_k, num_a, num_b, 3), dtype=complex)

    if minimum_dist_replica:
      R_improved = self.get_Rimproved()
      R_ = [r.R_base for r in R_improved]
      R_reshaped = np.array(R_).reshape((*h_r_ab.shape, -1))
      Tvecs_reshaped = np.empty(h_r_ab.shape, dtype=object)
      for i, idx in enumerate(np.ndindex(h_r_ab.shape)):
        Tvecs_reshaped[idx] = R_improved[i].T_vec

      for kpt in range(num_k): 
        kvec = k[kpt]
        for a in range(num_a):
          for b in range(num_b):
            tmp1 = 0.0 + 0.0j
            tmp2 = np.zeros((3), dtype=complex)
            for i in range(num_r):
              N_abR = Tvecs_reshaped[i,a,b].shape[0]
              Rvec = crys2ang(A, R_reshaped[i,a,b,:3])
              for j in range(N_abR):
                T_abR = crys2ang(A, Tvecs_reshaped[i,a,b][j])
                Vec = Rvec + T_abR
                tmp_ = (h_r_ab[i,a,b] * np.exp(1j * np.dot(k[kpt], 
                                                Vec))/ N_abR)
                tmp1 = tmp1 + tmp_
                tmp2[:] = tmp2[:] + (tmp_*Vec*1j)
            h_k_ab[kpt,a,b] = tmp1
            grad_h_k_ab[kpt, a, b,:] = tmp2
    else:
      for kpt in range(num_k):
        kvec = k[kpt]
        for a in range(num_a):
          for b in range(num_b):
            tmp1 = 0.0 + 0.0j
            tmp2 = np.zeros((3), dtype=complex)
            for i in range(num_r):
              Rvec = np.matmul(r_ws[i], A)
              tmp = (h_r_ab[i,a,b]* np.exp(1j *
                            np.dot(kvec, Rvec)))
              tmp1 = tmp1 + tmp
              tmp2[:] = tmp2[:] + (tmp*Rvec*1j)
            h_k_ab[kpt,a,b] = tmp1
            grad_h_k_ab[kpt, a, b,:] = tmp2

    # Addition of vertical external field term (if Ez is not None)
    if Ez is not None:
      pos_wf = map_wf[:, 3:6].astype(float)
      ind_wf = map_wf[:, 0:3].astype(int)
      z_orbital = np.repeat(pos_wf[:, 2], ind_wf[:, 2] - ind_wf[:, 1])
      if z_orbital.shape[0] != num_a:
        print_f("Error: Mismatch in number of Wannier functions and orbital mapping.")
        print_f("Exiting...")
        comm.Abort(1)
        
      z0 = np.mean(z_orbital)
      di = np.diag_indices(num_a)
      h_k_ab[:, di[0], di[1]] += -Ez * (z_orbital - z0)
    return np.moveaxis(h_k_ab, 0, -1),\
           np.moveaxis(grad_h_k_ab, 0, 2)  

  
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

  def get_num_wann(self):
    """
    Extract number of Wannier functions
    """
    with open(self.win_f) as f:
      for line in f:
        if "num_wann" in line:
          # Split at '=' and strip spaces
          parts = line.split("=")
          if len(parts) == 2:
            return int(parts[1].strip())
#print("---OOP---")
#WAN2BSE = WAN2BSE("WSe2.win", "WSe2_u.mat", "WSe2_hr.dat", "WF.wout")
#WAN2BSE.get_grad_Hk()

