try:
  from pyelpa import DistributedMatrix
  HAS_ELPA = True
except ImportError:
  HAS_ELPA = False

import os
import shutil
import sys
import time
from functools import partial
from itertools import combinations

import numpy as np
import h5py
from scipy.sparse import issparse
from scipy.spatial import KDTree

from wan90tobse import *
from read_yaml import *
from potential import *
from dft2bse import *
from addsoc import *
from cyfunc import *


from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0


class BSE(object):
  """
  Construct the BSE matrix, diagonalize, and post-process.
  Note on HDF5 I/O (04/07/2025): 
  In the previous PyMEX package, we used driver='mpio' and collective write. 
  But in many instances, the performance is the terrible. As a 
  result, switching to collecting to root node and writing a single 
  file. 
  """

  def __init__(self, inp_file):
    """
    Initializes the attributes
    """
    self.t0 = time.time()

    # --------------------
    # Get input parameters
    # --------------------
    self.inp_file = inp_file 
    self.config = print_yaml(self.inp_file)
    self.restart = self.config.get("restart", {})
    self.check_all()  
    self.estimate_memory_from_config()

    #-----------------------------------------------
    # Set up single particle bands and wavefunctions
    #-----------------------------------------------
    # When "dft": eigenvalues and eigenfunctions (cnmk's)
    #              are read from files. [Old/Not maintained.]
    # When "tb": eigenvalues and eigenfunctions are 
    #             generated using Wannier90 Hamiltonian.
    wannier = self.config["bse"]["wannier_io"]
    if self.config["bse"]["method"] == "dft":
      cnmk_inp = self.config["bse"]["dft"]["cnmk"]
    elif self.config["bse"]["method"] == "tb":
      cnmk_inp = self.config["bse"]["tb"]["cnmk"]

    self.wan2bse = WAN2BSE(wannier["win_file"], cnmk_inp["file"],
                           wannier["hr_file"], wannier["wsvec_file"],
                           wannier["wout_file"])

    if self.config["bse"]["method"] == "dft":
      self._setup_dft()
    elif self.config["bse"]["method"] == "tb":
      self._setup_tb()
    if rank == root:
      print_f("Received single-particle eigenvalues and Cnmk's")
      print_f(f"Time spent so far: {time.time() - self.t0:.4f} secs.\n")
     
    #---------------------------------------------------------------------
    # Set up electron-hole interactions (Screened Coulomb + Bare Coulomb)
    #---------------------------------------------------------------------
    # Electron-Hole interaction/potential
    # if dft: self.kvec is set through _setup_dft()
    # if tb: self.kvec is set though _setup_tb()
    self.A = self.wan2bse.get_lattice()
    self.map_wannier_functions()

    if self.config["bse"]["method"] == "tb":
      kgrid = self.config.get("bse", {}).get("tb", {}).get("kgrid", {})
    else:
      kgrid = self.wan2bse.get_kgrid()

    if self.config.get("material", None) is None:
      self.material = "2d"
      if rank == root:
        print_f(f"Material dimensionality (default): {self.material}")
    else:
      self.material = self.config.get("material", None)
      check_material(self.material)
      if rank == root:
        print_f(f"Material dimensionality: {self.material}")

    # Date: 22/07/2026
    # Added dimensionality check for the material.
    self.potential = POTENTIAL(self.wan2bse, self.material, kgrid=kgrid)
    self.Rvec = self.potential.get_Rvec()
    self.weight_Rvec = np.ones(self.Rvec.shape[0]) 
    if rank == root:
      print_f("Writing Supercell lattice vectors (Angstrom)")
      np.savetxt("Rvec.txt", self.Rvec, fmt="%.8f") 
      np.savetxt("Weight_Rvec.txt", self.weight_Rvec, fmt="%.8f")


    # Construct potential files from scratch
    if (not self.pot_files_exist() or
        self.restart == "from_scratch"):
      pot_in_r = self.config.get("eh_interaction", {}).get("space", {})
      if pot_in_r == "real":
        if rank == root:
          print_f("Using real-space potentials for el-hole interaction.")
        self.get_potential_in_realspace()
      else:
        if rank == root:
          print_f("Will FFT to real space for el-hole interaction.")
          print_f("Not implemented yet. Exiting...")
        comm.Abort(1)
    

    if rank == root:      
      print_f("El-Hole potential in real-space is done.")
      print_f(f"Time spent so far: {time.time() - self.t0:.3f} secs.\n")

    # ---------------- Set up Done -------------
    # -----x----x-----x-----x-----x-----x----x--


  def get_precomputed_potential(self):
    """
    Get the precomputed electron-hole potential
    """
    if getattr(self, '_potential_loaded', False):
      return
    
    add_direct = self.config.get("eh_interaction", {}).get("include", 
                                {}).get("direct", {})
    add_exchange = self.config.get("eh_interaction", {}).get("include", 
                          {}).get("exchange", {})
    screened_exchange = self.config.get("eh_interaction", {}).get("include",
                          {}).get("screen_exchange", {})

    # Load Direct term 
    if add_direct:
      self.V_r_keld = self.load_and_distribute_hdf5_array(
        "Screened_coulomb.hdf5", "W")
    else:
      self.V_r_keld = np.zeros((self.Rvec.shape[0],
                        self.wanfuncs.shape[0],
                        self.wanfuncs.shape[0]),
                        dtype=float)
    # Load Exchange term
    if add_exchange:
      if not screened_exchange:
        self.V_r_coul = self.load_and_distribute_hdf5_array(
          "Bare_coulomb.hdf5", "V")
      else:
        self.V_r_coul = self.V_r_keld
    else:
      self.V_r_coul = np.zeros((self.V_r_keld.shape))
    
    self._potential_loaded = True


  def get_potential_in_realspace(self):
    """
    Get the electron-hole potential in real-space
    """
    add_direct = self.config.get("eh_interaction", {}).get("include", 
                                {}).get("direct", {})
    add_exchange = self.config.get("eh_interaction", {}).get("include", 
                          {}).get("exchange", {})
    screened_exchange = self.config.get("eh_interaction", {}).get("include",
                          {}).get("screen_exchange", {})
    params = self.config.get("eh_interaction", {}).get("real", {})

    if add_direct:
      potname = self.setup_potname()
      if potname == "keldysh_monolayer_analytic":
        self.potential.keldysh_monolayer_analytic(self.Rvec,
                                self.wanfuncs, 
                                params["epsilon_t"], params["epsilon_b"], 
                                num_if_list(params["rho0"]), 
                                num_if_list(params["a0"]))
      elif potname == "keldysh_multilayer_numeric":
        self.potential.keldysh_multilayer_numeric(self.Rvec,
                                                  self.wanfuncs, 
                                params["epsilon_t"], params["epsilon_b"], 
                                list_if_num(params["rho0"]), 
                                list_if_num(params["a0"]),
                                list_if_num(params["zav"]),
                                params.get("thickness", 2.5))
      else:
        if rank == root:
          print_f(f"Potential '{potname}' not implemented yet.")
          print_f("Exiting...")
        comm.Abort(1)

    if add_exchange:
      if not screened_exchange:
        self.potential.coulomb_analytic(self.Rvec,
                                self.wanfuncs, 
                                params["epsilon_t"], params["epsilon_b"], 
                                list_if_num(params["a0"]),
                                list_if_num(params["zav"]),
                                params.get("thickness", 2.5))


  def setup_potname(self):
    """
    Set up potential name for the real-space electron-hole 
    screened potential.
    """
    cfg = self.config.get("eh_interaction", {}).get("real", {})
    potname_1 = cfg.get("potential", "")
    potname_2 = cfg.get("layer", "")
    potname_3 = cfg.get("type", "")
    return f"{potname_1}_{potname_2}_{potname_3}"


  def map_wannier_functions(self):
    """
    Wannier functions mapping (through Wannier90 io)
    """
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    
    # Defaults
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"    
    self.wanfuncs = self.wan2bse.map_WF(using_mlwf=using_mlwf,
                                        mode=wf_loc)
    if rank == root: 
      print_f()
      print_f(f"Mapped Wannier functions with {wf_loc} method.")
      if using_mlwf:
        print_f("Final positions of Wannier functions include MLWF centers.")        


  def print_self_vars_with_memory(self):
    """
    Take a look at what is "self"
    """
    for name, value in vars(self).items():
      if isinstance(value, np.ndarray):
        size_bytes = value.nbytes
      else:
        size_bytes = sys.getsizeof(value)
      size_mb = size_bytes / (1024**2)  # bytes to MB
      print_f(f"{name}: {type(value)} - approx. {size_mb:.2f} MB")


  def _setup_dft(self):
    """
    Set up eigenvalues and eigenvectors from DFT.
    """
    self.kvec = self.wan2bse.get_kpoints()
    self.E_ck = self.get_E_ck()
    self.E_vk = self.get_E_vk()

    self.C_nc_k = self.get_C_nc_k()
    self.C_nv_k = self.get_C_nv_k()


  def _setup_tb(self):
    """
    Compute Eigenvalues and eigenvectors for TB set up.
    Hamiltonian in real-space is constructed from i/o of Wannier90.
    Note on memory consumption (C_nm_k at each Q): 
    total elements = nQ * num_orb * num_orb * nk
    if nQ=1,num_orb=8000, nk=9, the memory required is:
      1 * 8000 * 8000 * 9 * 16 bytes ~ 9 GB.
    Overall, we need 2*C_nm_k and 2*E_vk memory for this i.e., in the above example,
    we would need approximately 18 GB. The memorey required to store eigenvalues are small.
    """
    if getattr(self, '_tb_setup_done', False):
      return
    
    # k and k+Q points
    self.kvec = self.set_kgrid_tb()
    # self.kvec = self.set_kgrid_tb_gamma_centered()
    
    hspoints = self.config.get("bse", {}).get("qpoints", {}).get("hspoints", {})
    numpoints_per_segment = self.config.get("bse", {}).get("qpoints", {}
                                           ).get("num_points", {})
    Qvec = self.get_interpolated_points(hspoints, numpoints_per_segment)
    if np.all(np.allclose(Qvec, 0, atol=1e-8)):
      if rank == root:
        print_f("Q=0 exciton momentum calculation!")
        print_f("Will compute tight binding eigenvalues and eigenvectors with k-grid")
        print_f("Please wait...")
      # To compute the eigenvalues or eigenvectors or load 
      if (not self.tb_files_at_kgrid_exist() or
          self.restart == "from_scratch"):
        self._get_tb_solution_at_kgrid()

      self.E_vk   = self.get_E_vk()
      self.C_nv_k = self.get_C_nv_k()

      tb_config = self.config.get("bse", {}).get("tb", {})
      shift     = next((v for k, v in tb_config.items()
                        if k.lower() == "gamma_k_shift"), None)
      wannier_io_config = self.config.get("bse", {}).get("wannier_io", {})
      wf_location       = wannier_io_config.get("wf_location", None)
      if not hasattr(self, 'wanfuncs'):
        self.map_wannier_functions()

      if shift:
        if wf_location == "atom_centered":
          if rank == root:
            print_f(f"Shifting d-character bands at folded Gamma up by {shift} eV.")
            print_f(f"Make sure you know what you are doing!")

          bands_shifted_file = "bands_shifted.npy"
          file_exists = comm.bcast(os.path.isfile(bands_shifted_file) if rank == root else None, root=root)

          if file_exists:
            if rank == root:
              print_f(f"Found '{bands_shifted_file}', loading pre-computed shifted bands.")
              E_vk_loaded = np.load(bands_shifted_file)
            else:
              E_vk_loaded = None
            self.E_vk = comm.bcast(E_vk_loaded, root=root)
            if rank == root:
              print_f(f"Done loading and broadcasting '{bands_shifted_file}'.")
          else:
            self.E_vk = self.selective_gamma_k_shift(self.C_nv_k, self.E_vk,
                                                    self.wanfuncs, shift)

          comm.Barrier()  # inside atom_centered block, covers both branches

        else:
          if rank == root:
            print_f(f"WARNING: gamma_K_shift is set to {shift} eV but "
                    f"wf_location = '{wf_location}' — shift requires "
                    f"wf_location: atom_centered. Skipping shift.")

      # Note that Q=0
      self.E_ckplusQ = np.expand_dims(self.get_E_ck(), axis=0)
      self.C_nc_kplusQ = np.expand_dims(self.get_C_nc_k(), axis=0)

    else:
      if rank == root:
        print_f("Calculations involve finite Q exciton.")
        print_f("Will compute tight binding eigenvalues and eigenvectors with k+Q grid")
        print_f("Please wait...")
      if (not self.tb_files_at_kplusQgrid_exist() or 
          self.restart == "from_scratch"):
        self._get_tb_solution_at_kplusQgrid(Qvec)
      self.E_vk = self.get_E_vk()
      self.C_nv_k = self.get_C_nv_k()
      # Note that Q contains non-zero vectors
      self.E_ckplusQ = self.get_E_ckplusQ()
      self.C_nc_kplusQ = self.get_C_nc_kplusQ()
    self._tb_setup_done = True


  def pot_files_exist(self):
    """
    Check if the potential files have already been computed.
    If computed and not corrupt then return True.
    """
    files = []
    add_direct = self.config.get("eh_interaction", {}).get("include",
                                {}).get("direct", {})
    add_exchange = self.config.get("eh_interaction", {}).get("include",
                          {}).get("exchange", {})
    screened_exchange = self.config.get("eh_interaction", {}).get("include",
                          {}).get("screen_exchange", {})
    if add_direct:
      files.append(("Screened_coulomb.hdf5", "W"))
    if add_exchange and not screened_exchange:
      files.append(("Bare_coulomb.hdf5", "V"))

    for fname, dset in files:
      if not os.path.exists(fname):
        return False
      try:
        with h5py.File(fname, "r") as f:
          if dset not in f:
            return False
      except OSError:
        return False
    return True


  def tb_files_at_kgrid_exist(self):
    """
    Check if the tb files have already been computed.
    If computed and not corrupt then return True.
    """
    for fname, dset in [("Eigval_k_tb.hdf5", "Eigenvalues"),
                        ("Eigvec_k_tb.hdf5", "Eigenvectors")]:
      if not os.path.exists(fname):
        return False
      try:
        with h5py.File(fname, "r") as f:
          if dset not in f:
            return False
      except OSError:
        return False
    return True


  def selective_gamma_k_shift(self, C_nv_k, E_vk, wanfuncs, shift):
    """
    Wannier90-specific version of the Gamma-K shift.
    Exploits fixed Wannier90 d-orbital ordering (dz2, dxz, dyz, dx2-y2, dxy)
    to directly compare dx2-y2+dxy vs dz2+dxz+dyz summed over all atoms
    per layer. At each k-point the top n_shift bands by two_dom-rest score
    are shifted, provided the score exceeds score_tol.
    If n_shift is None, all bands are scored and eligible for shifting.
    """
    n_shift    = self.config.get("bse", {}).get("tb", {}).get("num_shifts_per_k", None)
    score_tol  = self.config.get("bse", {}).get("tb", {}).get("shift_score_tol", 0.1)
    zav        = list_if_num(self.config.get("eh_interaction", {}).get("real", {}).get("zav", []))
    zav        = np.array(zav)
    z_tol      = 1.0
    n_layers   = len(zav)

    if rank == root and n_shift is None:
      print_f("num_shifts_per_k not set — scoring all bands per k-point.")

    shell_sizes   = (wanfuncs[:, 2] - wanfuncs[:, 1]).astype(int)
    d_shell_idx   = np.where(shell_sizes == 5)[0]
    d_atom_shells = [
      np.arange(wanfuncs[i, 1], wanfuncs[i, 2]).astype(int)
      for i in d_shell_idx
    ]
    d_atom_z = wanfuncs[d_shell_idx, 5]

    if len(d_atom_shells) == 0:
      if rank == root:
        print_f("ERROR: could not identify d orbital indices. Check wanfuncs format.")
      comm.Abort(1)

    layer_atoms = [[] for _ in range(n_layers)]
    for atom_i, z in enumerate(d_atom_z):
      for l, z_l in enumerate(zav):
        if np.abs(z - z_l) < z_tol:
          layer_atoms[l].append(atom_i)
          break

    # if rank == root:
    #   for l, atoms in enumerate(layer_atoms):
    #     print_f(f"Layer {l} (zav={zav[l]:.3f} Å): atoms {atoms}")

    n_val, n_kpts = E_vk.shape
    E_vk_shifted  = E_vk.copy()

    for k in range(n_kpts):
      scores = np.zeros(n_val)
      for v in range(n_val):
        best_score = -np.inf
        for l, atoms in enumerate(layer_atoms):
          layer_two_dom = 0.0
          layer_rest    = 0.0
          for atom_i in atoms:
            d_idx             = d_atom_shells[atom_i]
            d_orbital_weights = np.abs(C_nv_k[d_idx, v, k])**2
            layer_two_dom    += d_orbital_weights[3] + d_orbital_weights[4]
            layer_rest       += d_orbital_weights[0] + d_orbital_weights[1] + d_orbital_weights[2]
          best_score = max(best_score, layer_two_dom - layer_rest)
        scores[v] = best_score

      top_n = np.argsort(scores)[::-1][:n_shift]

      for v in top_n:
        if scores[v] > score_tol:
          E_vk_shifted[v, k] += shift
          status = "SHIFT"
        elif scores[v] > 0:
          status = "skip (weak)"
        else:
          status = "skip (neg)"
        # if rank == root:
        #   print_f(f"v={v:3d} k={k:3d} | score={scores[v]:.4f} → {status}")

    if rank == root:
      np.save("bands_unshifted.npy", E_vk)
      np.save("bands_shifted.npy",   E_vk_shifted)

    return E_vk_shifted


  def tb_files_at_kplusQgrid_exist(self):
    """
    Check if the tb files have already been computed.
    If computed and not corrupt then return True.
    """
    for fname, dset in [("Eigval_kplusQ_tb.hdf5", "Eigenvalues"),
                        ("Eigvec_kplusQ_tb.hdf5", "Eigenvectors")]:
      if not os.path.exists(fname):
        return False
      try:
        with h5py.File(fname, "r") as f:
          if dset not in f:
            return False
      except OSError:
        return False
    return True


  def _chunk_send_Cnmk(self, arr, dest, chunk_wann=100, chunk_axis=0):
    n = arr.shape[chunk_axis]
    for start in range(0, n, chunk_wann):
      end = min(start + chunk_wann, n)
      slc = [slice(None)] * arr.ndim
      slc[chunk_axis] = slice(start, end)
      comm.Send(np.ascontiguousarray(arr[tuple(slc)]), dest=dest, tag=0)


  def _chunk_recv_Cnmk(self, shape, dtype, source, chunk_wann=100, chunk_axis=0):
    arr = np.empty(shape, dtype=dtype)
    n = shape[chunk_axis]
    for start in range(0, n, chunk_wann):
      end = min(start + chunk_wann, n)
      chunk_shape = list(shape)
      chunk_shape[chunk_axis] = end - start
      buf = np.empty(chunk_shape, dtype=dtype)
      comm.Recv(buf, source=source, tag=0)
      slc = [slice(None)] * len(shape)
      slc[chunk_axis] = slice(start, end)
      arr[tuple(slc)] = buf
    return arr


  def _get_tb_solution_at_kgrid(self):
    """
    Compute tight-binding eigenvalues and eigenvectors at all k-points.

    Memory footprint (num_wann=10000, num_k=10):
      local arrays per rank  : num_wann^2 x my_count x 16 bytes ~ few GB
      global arrays on root  : eigvec/H_k ~ 16 GB each,
                              grad_H_k   ~ 48 GB
      => root requires ~81 GB; all other ranks hold only their local slice.

    MPI strategy — serialised chunked Send/Recv:
      Root issues a tag-99 green-light to each rank r before receiving from
      it. Ranks block on comm.recv(tag=99) and only then post their Sends.
      This ensures at most one rank has live OFI rendezvous state at a time,
      eliminating the "OFI rendezvous resource exhaustion" warning that
      occurred when all N-1 ranks posted simultaneously.

      Within each rank's transfer, large arrays are chunked along wann axis 0
      (CHUNK_WANN=100 rows) to keep individual messages at ~32 MB (eigvec/H_k)
      or ~96 MB (grad_H_k). eigval (~160 KB total) is sent unchunked.

    HDF5 output (root only):
      gzip level 4 reduces on-disk size by 2-5x. Downstream readers need
      no changes — HDF5 decompresses transparently on any slice access.
      chunks= is required by HDF5 to enable compression.
    """
    CHUNK_WANN = 100

    set_Ez = self.config.get("bse", {}).get("tb", {}).get("electric_field_z", None)
    if not hasattr(self, 'wanfuncs'):
      self.map_wannier_functions()
    if set_Ez is not None and rank == root:
      print_f(f"Vertical electric field applied: {set_Ez} eV/Å")

    num_kpt  = len(self.kvec)
    dim      = 3
    num_wann = self.wan2bse.get_num_wann()

    my_start, my_end = self.distribute_full(num_kpt, rank, size)
    my_count = max(0, my_end - my_start)

    if my_count > 0:
      local_eigval      = np.zeros((num_wann,           my_count),      dtype=np.float64)
      local_eigvec      = np.zeros((num_wann, num_wann, my_count),      dtype=np.complex128)
      local_h_ab_k      = np.zeros((num_wann, num_wann, my_count),      dtype=np.complex128)
      local_grad_h_ab_k = np.zeros((num_wann, num_wann, my_count, dim), dtype=np.complex128)

      for i, ikpt in enumerate(range(my_start, my_end)):
        kpt = self.kvec[ikpt]
        try:
          H_k, Grad_H_k = self.get_Hk_at_kpt(kpt, Ez=set_Ez, map_wf=self.wanfuncs)
        except Exception as e:
          print_f(f"[Rank {rank}] get_Hk_at_kpt failed at ikpt={ikpt}: {e}")
          comm.Abort(1)
        H_k      = H_k.reshape((num_wann, num_wann))
        Grad_H_k = Grad_H_k.reshape((num_wann, num_wann, dim))
        self.is_hermitian(H_k)
        eigval, eigvec = np.linalg.eigh(H_k)
        local_eigval[:,    i]         = eigval
        local_eigvec[:, :, i]         = eigvec
        local_h_ab_k[:, :, i]         = H_k
        local_grad_h_ab_k[:, :, i, :] = Grad_H_k
    else:
      local_eigval = local_eigvec = local_h_ab_k = local_grad_h_ab_k = None

    if rank == root:
      global_eigval      = np.zeros((num_wann,           num_kpt),      dtype=np.float64)
      global_eigvec      = np.zeros((num_wann, num_wann, num_kpt),      dtype=np.complex128)
      global_h_ab_k      = np.zeros((num_wann, num_wann, num_kpt),      dtype=np.complex128)
      global_grad_h_ab_k = np.zeros((num_wann, num_wann, num_kpt, dim), dtype=np.complex128)

      if my_count > 0:
        global_eigval[:,    my_start:my_end]          = local_eigval
        global_eigvec[:, :, my_start:my_end]          = local_eigvec
        global_h_ab_k[:, :, my_start:my_end]          = local_h_ab_k
        global_grad_h_ab_k[:, :, my_start:my_end, :] = local_grad_h_ab_k

      for r in range(size):
        if r == root:
          continue
        r_start, r_end = self.distribute_full(num_kpt, r, size)
        r_count = max(0, r_end - r_start)
        if r_count == 0:
          continue

        comm.send(None, dest=r, tag=99)

        recv_val = np.empty((num_wann, r_count), dtype=np.float64)
        comm.Recv(recv_val, source=r, tag=0)
        global_eigval[:, r_start:r_end] = recv_val

        global_eigvec[:, :, r_start:r_end] = self._chunk_recv_Cnmk(
          shape=(num_wann, num_wann, r_count),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=0,
        )
        global_h_ab_k[:, :, r_start:r_end] = self._chunk_recv_Cnmk(
          shape=(num_wann, num_wann, r_count),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=0,
        )
        global_grad_h_ab_k[:, :, r_start:r_end, :] = self._chunk_recv_Cnmk(
          shape=(num_wann, num_wann, r_count, dim),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=0,
        )

    else:
      if my_count > 0:
        comm.recv(source=root, tag=99)

        comm.Send(local_eigval, dest=root, tag=0)
        self._chunk_send_Cnmk(local_eigvec,      dest=root, chunk_wann=CHUNK_WANN, chunk_axis=0)
        self._chunk_send_Cnmk(local_h_ab_k,      dest=root, chunk_wann=CHUNK_WANN, chunk_axis=0)
        self._chunk_send_Cnmk(local_grad_h_ab_k, dest=root, chunk_wann=CHUNK_WANN, chunk_axis=0)

    if rank == root:
      try:
        c_w = min(64, num_wann)
        c_k = min(64, num_kpt)
        with h5py.File("Eigval_k_tb.hdf5",    "w") as f_val,  \
            h5py.File("Eigvec_k_tb.hdf5",    "w") as f_vec,  \
            h5py.File("H_k_tb.hdf5",         "w") as f_hmat, \
            h5py.File("Grad_H_k_tb.hdf5",    "w") as f_grad_hmat:
          f_val.create_dataset(
            "Eigenvalues",             data=global_eigval,
            chunks=(c_w, c_k),         compression="gzip", compression_opts=4)
          f_vec.create_dataset(
            "Eigenvectors",            data=global_eigvec,
            chunks=(c_w, c_w, c_k),    compression="gzip", compression_opts=4)
          f_hmat.create_dataset(
            "TB_Hamiltonian",          data=global_h_ab_k,
            chunks=(c_w, c_w, c_k),    compression="gzip", compression_opts=4)
          f_grad_hmat.create_dataset(
            "TB_Gradient_Hamiltonian", data=global_grad_h_ab_k,
            chunks=(c_w, c_w, c_k, dim), compression="gzip", compression_opts=4)
          print_f("TB solution written to HDF5 (gzip level 4)")
      except Exception as e:
        print_f(f"[Rank {rank}] HDF5 write failed: {e}")
        comm.Abort(1)


  def _get_tb_solution_at_kplusQgrid(self, Qvec):
    """
    Compute tight-binding eigenvalues and eigenvectors at k+Q points.

    Memory footprint (num_wann=10000, num_k=10, num_Q=num_Q):
      local arrays per rank  : num_Q x num_wann^2 x my_count x 16 bytes
      global arrays on root  : eigvec/H_k ~ 16*num_Q GB each,
                              grad_H_k   ~ 48*num_Q GB
      => root memory scales with num_Q; plan accordingly.

    MPI strategy — serialised chunked Send/Recv along wann axis 1:
      Arrays here have shape (num_Q, num_wann, num_wann, k, ...) so the
      first wann index is axis 1, not axis 0 as in _get_tb_solution_at_kgrid.
      chunk_axis=1 is passed explicitly to _chunk_send_Cnmk/_chunk_recv_Cnmk.
      Root green-lights each rank individually (tag=99) before receiving
      from it, so only one rank has live OFI rendezvous state at a time.
      eigval shape is (num_Q, num_wann, k) — chunked along axis 1.

    HDF5 output (root only):
      gzip level 4 reduces on-disk size by 2-5x. Downstream readers need
      no changes — HDF5 decompresses transparently on any slice access.
    """
    CHUNK_WANN = 100

    set_Ez = self.config.get("bse", {}).get("tb", {}).get("electric_field_z", None)
    if not hasattr(self, 'wanfuncs'):
      self.map_wannier_functions()
    if set_Ez is not None and rank == root:
      print_f(f"Vertical electric field applied: {set_Ez} eV/Å")

    num_kpt  = len(self.kvec)
    num_wann = self.wan2bse.get_num_wann()
    dim      = 3

    kvec_recip   = self.set_kgrid_tb_reciprocal()
    kplusQ_recip = (Qvec[:, None, :] + kvec_recip[None, :, :]) % 1.0
    B            = self.wan2bse.get_reciprocal()
    kplusQ       = kplusQ_recip @ B
    num_Q        = kplusQ.shape[0]

    my_start, my_end = self.distribute_full(num_kpt, rank, size)
    my_count = max(0, my_end - my_start)

    if my_count > 0:
      local_eigval           = np.zeros((num_Q, num_wann,           my_count),      dtype=np.float64)
      local_eigvec           = np.zeros((num_Q, num_wann, num_wann, my_count),      dtype=np.complex128)
      local_h_ab_kplusQ      = np.zeros((num_Q, num_wann, num_wann, my_count),      dtype=np.complex128)
      local_grad_h_ab_kplusQ = np.zeros((num_Q, num_wann, num_wann, my_count, dim), dtype=np.complex128)

      for iQ in range(num_Q):
        for i, ikpt in enumerate(range(my_start, my_end)):
          kpt = kplusQ[iQ, ikpt, :]
          try:
            H_kplusQ, Grad_H_kplusQ = self.get_Hk_at_kpt(kpt, Ez=set_Ez, map_wf=self.wanfuncs)
          except Exception as e:
            print_f(f"[Rank {rank}] get_Hk_at_kpt failed at iQ={iQ}, ikpt={ikpt}: {e}")
            comm.Abort(1)
          H_kplusQ      = H_kplusQ.reshape((num_wann, num_wann))
          Grad_H_kplusQ = Grad_H_kplusQ.reshape((num_wann, num_wann, dim))
          self.is_hermitian(H_kplusQ)
          eigval, eigvec = np.linalg.eigh(H_kplusQ)
          local_eigval[iQ, :, i]                  = eigval
          local_eigvec[iQ, :, :, i]               = eigvec
          local_h_ab_kplusQ[iQ, :, :, i]          = H_kplusQ
          local_grad_h_ab_kplusQ[iQ, :, :, i, :] = Grad_H_kplusQ
    else:
      local_eigval = local_eigvec = local_h_ab_kplusQ = local_grad_h_ab_kplusQ = None

    if rank == root:
      global_eigval           = np.zeros((num_Q, num_wann,           num_kpt),      dtype=np.float64)
      global_eigvec           = np.zeros((num_Q, num_wann, num_wann, num_kpt),      dtype=np.complex128)
      global_h_ab_kplusQ      = np.zeros((num_Q, num_wann, num_wann, num_kpt),      dtype=np.complex128)
      global_grad_h_ab_kplusQ = np.zeros((num_Q, num_wann, num_wann, num_kpt, dim), dtype=np.complex128)

      if my_count > 0:
        global_eigval[:, :, my_start:my_end]                  = local_eigval
        global_eigvec[:, :, :, my_start:my_end]               = local_eigvec
        global_h_ab_kplusQ[:, :, :, my_start:my_end]          = local_h_ab_kplusQ
        global_grad_h_ab_kplusQ[:, :, :, my_start:my_end, :] = local_grad_h_ab_kplusQ

      for r in range(size):
        if r == root:
          continue
        r_start, r_end = self.distribute_full(num_kpt, r, size)
        r_count = max(0, r_end - r_start)
        if r_count == 0:
          continue

        comm.send(None, dest=r, tag=99)

        global_eigval[:, :, r_start:r_end] = self._chunk_recv_Cnmk(
          shape=(num_Q, num_wann, r_count),
          dtype=np.float64, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=1,
        )
        global_eigvec[:, :, :, r_start:r_end] = self._chunk_recv_Cnmk(
          shape=(num_Q, num_wann, num_wann, r_count),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=1,
        )
        global_h_ab_kplusQ[:, :, :, r_start:r_end] = self._chunk_recv_Cnmk(
          shape=(num_Q, num_wann, num_wann, r_count),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=1,
        )
        global_grad_h_ab_kplusQ[:, :, :, r_start:r_end, :] = self._chunk_recv_Cnmk(
          shape=(num_Q, num_wann, num_wann, r_count, dim),
          dtype=np.complex128, source=r,
          chunk_wann=CHUNK_WANN, chunk_axis=1,
        )

    else:
      if my_count > 0:
        comm.recv(source=root, tag=99)

        self._chunk_send_Cnmk(local_eigval,           dest=root, chunk_wann=CHUNK_WANN, chunk_axis=1)
        self._chunk_send_Cnmk(local_eigvec,           dest=root, chunk_wann=CHUNK_WANN, chunk_axis=1)
        self._chunk_send_Cnmk(local_h_ab_kplusQ,      dest=root, chunk_wann=CHUNK_WANN, chunk_axis=1)
        self._chunk_send_Cnmk(local_grad_h_ab_kplusQ, dest=root, chunk_wann=CHUNK_WANN, chunk_axis=1)

    if rank == root:
      try:
        c_Q = min(8,  num_Q)
        c_w = min(64, num_wann)
        c_k = min(64, num_kpt)
        with h5py.File("Eigval_kplusQ_tb.hdf5",    "w") as f_val, \
            h5py.File("Eigvec_kplusQ_tb.hdf5",    "w") as f_vec, \
            h5py.File("H_kplusQ_tb.hdf5",         "w") as f_h,   \
            h5py.File("Grad_H_kplusQ_tb.hdf5",    "w") as f_grad_h:
          f_val.create_dataset(
            "Eigenvalues",             data=global_eigval,
            chunks=(c_Q, c_w, c_k),      compression="gzip", compression_opts=4)
          f_vec.create_dataset(
            "Eigenvectors",            data=global_eigvec,
            chunks=(c_Q, c_w, c_w, c_k), compression="gzip", compression_opts=4)
          f_h.create_dataset(
            "TB_Hamiltonian",          data=global_h_ab_kplusQ,
            chunks=(c_Q, c_w, c_w, c_k), compression="gzip", compression_opts=4)
          f_grad_h.create_dataset(
            "TB_Gradient_Hamiltonian", data=global_grad_h_ab_kplusQ,
            chunks=(c_Q, c_w, c_w, c_k, dim), compression="gzip", compression_opts=4)
          print_f("TB k+Q solution written to HDF5 (gzip level 4)")
      except Exception as e:
        print_f(f"[Rank {rank}] HDF5 write failed: {e}")
        comm.Abort(1)


  def set_kgrid_tb_reciprocal(self):
    """
    Set up Wannier90-style k-point grids
    """
    kgrid = self.config.get("bse", {}).get("tb", {}).get("kgrid", {})
    # Generate points along each axis: 0, 1/N, 2/N, ..., (N-1)/N
    kx = np.arange(kgrid[0]) / kgrid[0]
    ky = np.arange(kgrid[1]) / kgrid[1]
    kz = np.arange(kgrid[2]) / kgrid[2]

    # Create meshgrid and flatten to get all combinations
    kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing='ij')
    kpoints = np.vstack([kxv.ravel(), kyv.ravel(), kzv.ravel()]).T
    return kpoints


  def set_kgrid_tb_reciprocal_gamma_centered(self):
    """
    Set up Gamma-centered k-point grids in reciprocal space
    
    For odd N:  includes Gamma point exactly at origin
                e.g., N=5: [-2/5, -1/5, 0, 1/5, 2/5]
    
    For even N: Gamma point is between grid points
                e.g., N=4: [-3/8, -1/8, 1/8, 3/8]
    """
    kgrid = self.config.get("bse", {}).get("tb", {}).get("kgrid", {})
    
    def make_gamma_centered_grid(N):
      """Generate Gamma-centered grid for N points"""
      if N % 2 == 1:  # Odd: Gamma point on grid
        return (np.arange(N) - N // 2) / N
      else:  # Even: Gamma point between grid points
        return (np.arange(N) + 0.5 - N / 2) / N
    
    kx = make_gamma_centered_grid(kgrid[0])
    ky = make_gamma_centered_grid(kgrid[1])
    kz = make_gamma_centered_grid(kgrid[2])
    
    # Create meshgrid and flatten to get all combinations
    kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing='ij')
    kpoints = np.vstack([kxv.ravel(), kyv.ravel(), kzv.ravel()]).T
    
    return kpoints


  def set_kgrid_tb(self):
    """
    Set up Wannier90-style k-point grids
    """
    kgrid = self.config.get("bse", {}).get("tb", {}).get("kgrid", {})
    # Generate points along each axis: 0, 1/N, 2/N, ..., (N-1)/N
    kx = np.arange(kgrid[0]) / kgrid[0]
    ky = np.arange(kgrid[1]) / kgrid[1]
    kz = np.arange(kgrid[2]) / kgrid[2]

    # Create meshgrid and flatten to get all combinations
    kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing='ij')
    kpoints = np.vstack([kxv.ravel(), kyv.ravel(), kzv.ravel()]).T
    B = self.wan2bse.get_reciprocal()
    return kpoints @ B 


  def set_kgrid_tb_gamma_centered(self):
    """
    Set up Gamma-centered k-point grids
    
    For odd N:  includes Gamma point exactly at origin
                e.g., N=5: [-2/5, -1/5, 0, 1/5, 2/5]
    
    For even N: Gamma point is between grid points
                e.g., N=4: [-3/8, -1/8, 1/8, 3/8]
    """
    kgrid = self.config.get("bse", {}).get("tb", {}).get("kgrid", {})
    
    def make_gamma_centered_grid(N):
      """Generate Gamma-centered grid for N points"""
      if N % 2 == 1:  # Odd: Gamma point on grid
        return (np.arange(N) - N // 2) / N
      else:  # Even: Gamma point between grid points
        return (np.arange(N) + 0.5 - N / 2) / N
    
    kx = make_gamma_centered_grid(kgrid[0])
    ky = make_gamma_centered_grid(kgrid[1])
    kz = make_gamma_centered_grid(kgrid[2])
    
    # Create meshgrid and flatten to get all combinations
    kxv, kyv, kzv = np.meshgrid(kx, ky, kz, indexing='ij')
    kpoints = np.vstack([kxv.ravel(), kyv.ravel(), kzv.ravel()]).T
    
    B = self.wan2bse.get_reciprocal()
    return kpoints @ B


  def check_all(self):
    """
    Preparatory safety checks. Always perform them 
    so that compute time is not wasted. Most things
    are not intentionally default. So, the user 
    knows reasonably what is being done.
    """
    width = 70
    if rank == root:
      print_f(f"\n")
      print_f(width*"+")
      print_f("Start of Warning|Error messages".center(width))
      print_f()
      check_excitation(self.config["excitation"])
      check_material(self.config["material"])
      check_bse(self.config["bse"])
      check_wannier_io(self.config["bse"])
      check_eh_interaction(self.config["eh_interaction"])
      check_absorption(self.config["absorption"])
      check_system(self.config["system"])
      check_diagonalization(self.config["diagonalize"])
      print_f()
      print_f("End of Warning|Error messages".center(width))
      print_f(width*"+")

      print_f()
      print_f("~~~Passed the preliminary checks~~~".center(width))
      print_f()
      print_f("|"+width*"-"+"|")
      print_f(("|"+"Start of OUTPUT".center(width)+"|"))
      print_f("|"+width*"-"+"|")
      print_f()


  def gaussian(self, E, Eex, sigma):
    """
    Gaussian function as a representation of the delta function
    Args:
      E: Energy that's a variable
      Eex: Exciton discrete values
      sigma: Half-width at half-maximum
    Returns:
      Gaussian function value at E
    """
    return np.exp((-(E-Eex)**2.)/(2*sigma**2))* 1/(sigma*np.sqrt(2*np.pi))

  
  def select_Gamma(self):
    """
    Selects the Gamma point from the Q-points file.
    Reads 'Selected_Q.txt' file and finds the index of the Gamma point.
    The Gamma point is defined as the point where all components of Q are zero.
    """
    method = self.config.get("bse", {}).get("method", {})
    if method == "dft":
      Qpoints = np.loadtxt('Selected_Q.txt', comments='#')
      if Qpoints.ndim == 1:
        Qpoints = Qpoints.reshape(1, -1)
      Gamma_rows = np.all(Qpoints == 0, axis=1)
      Gamma_idx = np.where(Gamma_rows)[0]
      return Gamma_idx[0]
    elif method == "tb":
      hspoints = self.config.get("bse", {}).get("qpoints", {}).get("hspoints", {})
      numpoints_per_segment = self.config.get("bse", {}).get("qpoints", {}
                                           ).get("num_points", {})
      Qvec = self.get_interpolated_points(hspoints, numpoints_per_segment)
      Gamma_idx = np.where(np.all(np.abs(Qvec) < 10**-8, axis=1))[0]
      return Gamma_idx[0]


  def load_and_distribute_eigensol(self, filename, target_idx,
                                    chunk_size=10_000_000):
    """
    Load and distribute eigenvals/eigenvecs at the specific target index
    """
    if rank == root:
      print_f(f"Using {filename} to load BSE solutions.")
      with h5py.File(filename, 'r') as f:
        for idx, key in enumerate(f.keys()):
          if idx == target_idx:
            data = np.array(f[key])
            break
        else:
          data = None
    else:
      data = None
    shape = comm.bcast(data.shape if rank == root else None, root=root)
    dtype = comm.bcast(data.dtype if rank == root else None, root=root)
    if rank != root:
      data = np.empty(shape, dtype=dtype)

    flat = data.ravel()
    total = flat.size

    for i in range(0, total, chunk_size):
      end = min(i + chunk_size, total)
      comm.Bcast(flat[i:end], root=root)
    return data   


  def get_Hk_at_kpt(self, kpt, Ez=None, map_wf=None):
    """
    Compute H_ab_k with Wannier90 derived Hamiltonian
    at that specific kpt for that specific rank.
    Returns:
      H_ab_k: shape (n_a, n_b, n_k)
      grad_H_ab_k: shape (n_a, n_b, n_k, 3)
    """
    H_ab_k, grad_H_ab_k = self.wan2bse.get_Hk_and_grad_Hk(method="tb", kpt=kpt, Ez=Ez, map_wf=map_wf)
    return H_ab_k, grad_H_ab_k


  def get_Hk_and_grad_Hk_wannier(self):
    """
    Compute H_ab_k and grad_H_ab_k from Wannier functions
    Returns:
      H_ab_k: shape (n_a, n_b, n_k)
      grad_H_ab_k: shape (n_a, n_b, n_k, 3)
    """
    # Get data and metadata at root
    if rank == root:
      H_ab_k, grad_H_ab_k = self.wan2bse.get_Hk_and_grad_Hk(method="dft")
      shape_H = H_ab_k.shape
      dtype_H = H_ab_k.dtype
      shape_grad = grad_H_ab_k.shape
      dtype_grad = grad_H_ab_k.dtype
    else:
      shape_H, dtype_H, shape_grad, dtype_grad = None, None, None, None

    # Broadcast metadata
    shape_H = comm.bcast(shape_H, root=root)
    dtype_H = comm.bcast(dtype_H, root=root)
    shape_grad = comm.bcast(shape_grad, root=root)
    dtype_grad = comm.bcast(dtype_grad, root=root)

    # Allocate arrays on non-root ranks
    if rank != root:
      H_ab_k = np.empty(shape_H, dtype=dtype_H)
      grad_H_ab_k = np.empty(shape_grad, dtype=dtype_grad)

    # Broadcast H_ab_k in chunks
    flat_H = H_ab_k.ravel()
    total_elements_H = flat_H.size
    chunk_size = 10_000_000  # ~80MB for float64

    for i in range(0, total_elements_H, chunk_size):
      end = min(i + chunk_size, total_elements_H)
      comm.Bcast(flat_H[i:end], root=root)

    # Broadcast grad_H_ab_k in chunks
    flat_grad = grad_H_ab_k.ravel()
    total_elements_grad = flat_grad.size

    for i in range(0, total_elements_grad, chunk_size):
      end = min(i + chunk_size, total_elements_grad)
      comm.Bcast(flat_grad[i:end], root=root)

    return H_ab_k, grad_H_ab_k


  def load_Hk_and_grad_Hk(self, Hk_file="H_k_tb.hdf5",
                          gradHk_file="Grad_H_k_tb.hdf5",
                          chunk_size=10_000_000):
    """
    Load and broadcast H_ab_k and grad_H_ab_k mostly for TB.
    """
    if rank == root:
      with h5py.File(Hk_file, "r") as f:
        H_ab_k = f["TB_Hamiltonian"][:]
      with h5py.File(gradHk_file, "r") as f:
        grad_H_ab_k = f["TB_Gradient_Hamiltonian"][:]
      H_shape, H_dtype = H_ab_k.shape, H_ab_k.dtype
      G_shape, G_dtype = grad_H_ab_k.shape, grad_H_ab_k.dtype
    else:
      H_ab_k = grad_H_ab_k = None
      H_shape = G_shape = None
      H_dtype = G_dtype = None

    # Broadcast metadata
    H_shape = comm.bcast(H_shape, root=root)
    H_dtype = comm.bcast(H_dtype, root=root)
    G_shape = comm.bcast(G_shape, root=root)
    G_dtype = comm.bcast(G_dtype, root=root)

    # Allocate memory on non-root ranks
    if rank != root:
      H_ab_k = np.empty(H_shape, dtype=H_dtype)
      grad_H_ab_k = np.empty(G_shape, dtype=G_dtype)

    # Broadcast helper
    def bcast_array(array):
      flat = array.ravel()
      total = flat.size
      for i in range(0, total, chunk_size):
        comm.Bcast(flat[i:i+chunk_size], root=root)
      return array

    # Broadcast actual data
    H_ab_k = bcast_array(H_ab_k)
    grad_H_ab_k = bcast_array(grad_H_ab_k)
    return H_ab_k, grad_H_ab_k


  def _chunked_gather_and_write_H(self, local_data, dset, key, total_Q,
                                  target_memory_gb=0.5,
                                  completion_msg=None):
    import math

    try:
      items     = list(local_data.items()) if local_data else []
      num_items = len(items)

      # ── 1. Broadcast item_shape from first rank that has data ──
      has_data = np.array([1 if num_items > 0 else 0], dtype=np.int32)
      all_has  = np.zeros(size, dtype=np.int32)
      comm.Allgather(has_data, all_has)
      src_rank = int(np.argmax(all_has))

      shape_buf = np.zeros(3, dtype=np.int64)
      if rank == src_rank:
        shape_buf[:] = items[0][1].shape
      comm.Bcast(shape_buf, root=src_rank)
      item_shape = tuple(shape_buf.tolist())
      item_size  = int(np.prod(item_shape))

      # ── 2. Share per-rank counts ──
      local_count = np.array([num_items], dtype=np.int64)
      all_counts  = np.zeros(size, dtype=np.int64)
      comm.Allgather(local_count, all_counts)

      # ── 3. Compute chunk size ──
      bytes_per_item = item_size * np.dtype(np.complex128).itemsize
      chunk_size     = max(1, int(target_memory_gb * 1024**3 / (bytes_per_item * size)))
      local_chunks   = math.ceil(num_items / chunk_size) if num_items > 0 else 0
      max_chunks     = int(comm.allreduce(local_chunks, op=MPI.MAX))

    except Exception as e:
      import traceback
      print_f(f"Rank {rank} failed during setup:\n{traceback.format_exc()}")
      comm.Abort(1)

    if rank == root:
      print_f(f"chunk_size={chunk_size}  "
              f"({bytes_per_item * chunk_size * size / 1024**3:.3f} GB/gather) "
              f"| {max_chunks} chunk(s)")
      print_f(f"Writing data for Q-point {key+1}/{total_Q} to HDF5...")

    # ── 4. Chunked Gatherv loop ──
    for c in range(max_chunks):
      lo = c * chunk_size
      hi = min(num_items, lo + chunk_size)

      if lo < num_items:
        chunk_items = items[lo:hi]
        c_keys = np.ascontiguousarray(
          [[ck, cv, ck2] for (ck, cv, ck2), _ in chunk_items], dtype=np.int64)
        c_vals = np.ascontiguousarray(
          [v for _, v in chunk_items], dtype=np.complex128)
      else:
        c_keys = np.empty((0, 3),           dtype=np.int64)
        c_vals = np.empty((0, *item_shape), dtype=np.complex128)

      # All ranks share chunk sizes for this iteration
      local_n  = np.array([len(c_keys)], dtype=np.int64)
      chunk_ns = np.zeros(size, dtype=np.int64)
      comm.Allgather(local_n, chunk_ns)
      total_n = int(chunk_ns.sum())

      if total_n == 0:
        continue

      # ── Gatherv for keys (n, 3) ──
      if rank == root:
        try:
          recv_keys = np.empty((total_n, 3), dtype=np.int64)
        except MemoryError:
          import traceback
          print_f(f"Root failed to allocate recv_keys at chunk {c}:\n{traceback.format_exc()}")
          comm.Abort(1)
      else:
        recv_keys = None

      comm.Gatherv(
        c_keys.ravel(),
        (recv_keys, (chunk_ns * 3).tolist()) if rank == root else None,
        root=root,
      )

      # ── Gatherv for values (n, *item_shape) ──
      if rank == root:
        try:
          recv_vals = np.empty((total_n, *item_shape), dtype=np.complex128)
        except MemoryError:
          import traceback
          print_f(f"Root failed to allocate recv_vals at chunk {c}:\n{traceback.format_exc()}")
          comm.Abort(1)
      else:
        recv_vals = None

      comm.Gatherv(
        c_vals.ravel(),
        (recv_vals, (chunk_ns * item_size).tolist()) if rank == root else None,
        root=root,
      )

      # ── Write to HDF5 (outside any collective) ──
      if rank == root:
        try:
          recv_keys = recv_keys.reshape(total_n, 3)
          for i in range(total_n):
            ci, vi, ki = recv_keys[i]
            dset[key, ci, vi, ki, :, :, :] = recv_vals[i]
        except Exception as e:
          import traceback
          print_f(f"Root failed writing chunk {c} to HDF5:\n{traceback.format_exc()}")
          comm.Abort(1)
        del recv_keys, recv_vals

    comm.Barrier()
    if rank == root and completion_msg:
      print_f(f"Q-point {key + 1}/{total_Q}: {completion_msg}")
      print_f()


  def _chunked_gather_and_write_conductivity(self, local_indices, local_conductivity,
                                            dset, target_memory_gb=0.5,
                                            completion_msg=None):
    import math

    try:
      num_items = len(local_indices)

      # ── 1. Share per-rank counts via Allgather ──
      local_count = np.array([num_items], dtype=np.int64)
      all_counts  = np.zeros(size, dtype=np.int64)
      comm.Allgather(local_count, all_counts)
      total_items = int(all_counts.sum())

      # ── 2. Compute chunk size (bytes of conductivity values only) ──
      bytes_per_item = 3 * np.dtype(np.complex128).itemsize   # 3 × 16 B
      chunk_size     = max(1, int(target_memory_gb * 1024**3 / (bytes_per_item * size)))
      local_chunks   = math.ceil(num_items / chunk_size) if num_items > 0 else 0
      max_chunks     = int(comm.allreduce(local_chunks, op=MPI.MAX))

    except Exception as e:
      import traceback
      print_f(f"Rank {rank} failed during setup:\n{traceback.format_exc()}")
      comm.Abort(1)

    if rank == root:
      print_f(f"chunk_size={chunk_size}  "
              f"({bytes_per_item * chunk_size * size / 1024**3:.3f} GB/gather) "
              f"| {max_chunks} chunk(s) | {total_items} total BSE states")
      print_f("Writing conductivity to HDF5...")

    # ── 3. Chunked Gatherv loop ──
    for c in range(max_chunks):
      lo = c * chunk_size
      hi = min(num_items, lo + chunk_size)

      if lo < num_items:
        c_indices = np.ascontiguousarray(local_indices[lo:hi],      dtype=np.int64)
        c_values  = np.ascontiguousarray(local_conductivity[lo:hi], dtype=np.complex128)
      else:
        c_indices = np.empty(0,      dtype=np.int64)
        c_values  = np.empty((0, 3), dtype=np.complex128)

      # All ranks share their chunk sizes for this iteration
      local_n  = np.array([len(c_indices)], dtype=np.int64)
      chunk_ns = np.zeros(size, dtype=np.int64)
      comm.Allgather(local_n, chunk_ns)
      total_n = int(chunk_ns.sum())

      if total_n == 0:
        continue

      # ── Gatherv for indices ──
      if rank == root:
        try:
          recv_idx = np.empty(total_n, dtype=np.int64)
        except MemoryError:
          import traceback
          print_f(f"Root failed to allocate recv_idx at chunk {c}:\n{traceback.format_exc()}")
          comm.Abort(1)
      else:
        recv_idx = None

      comm.Gatherv(
        c_indices,
        (recv_idx, chunk_ns.tolist()) if rank == root else None,
        root=root,
      )

      # ── Gatherv for conductivity values (flattened to 1-D for MPI) ──
      if rank == root:
        try:
          recv_vals = np.empty(total_n * 3, dtype=np.complex128)
        except MemoryError:
          import traceback
          print_f(f"Root failed to allocate recv_vals at chunk {c}:\n{traceback.format_exc()}")
          comm.Abort(1)
      else:
        recv_vals = None

      comm.Gatherv(
        c_values.ravel(),
        (recv_vals, (chunk_ns * 3).tolist()) if rank == root else None,
        root=root,
      )

      # ── Write to HDF5 (outside any collective) ──
      if rank == root and total_n > 0:
        try:
          dset[recv_idx, :] = recv_vals.reshape(total_n, 3)
        except Exception as e:
          import traceback
          print_f(f"Root failed writing chunk {c} to HDF5:\n{traceback.format_exc()}")
          comm.Abort(1)
        del recv_idx, recv_vals

    comm.Barrier()
    if rank == root and completion_msg:
      print_f(completion_msg)
      print_f()


  def optical_conductivity(self, Gamma_only=True):
    """
    Optical conductivity
    """
    io_files = self.set_exciton_io_files()
    method = self.config.get("bse", {}).get("method", {})
    Conductivity_file = io_files["conductivity_file"]
    absorption_spin = self.config.get("absorption", {}).get("spin", {})
    photon_energy = self.config.get("absorption", {}).get("photon_energy", {})
    photon_min = photon_energy.get("min", {})
    photon_max = photon_energy.get("max", {})
    step = photon_energy.get("step", {})
    sigma =  photon_energy.get("sigma", {})

    if Gamma_only:
      # Load data at root and distribute
      Gamma_idx = self.select_Gamma()
      eigvals = self.load_and_distribute_eigensol(io_files["eigenval_file"], Gamma_idx)
      eigvecs = self.load_and_distribute_eigensol(io_files["eigenvec_file"], Gamma_idx)

      # Get how many BSE eigenvalues to work with
      Emin = np.min(eigvals)
      threshold = Emin + photon_max 
      S_to_include = np.sum(eigvals <= threshold)
      # A_Scvk in the right-format
      if method == "dft":
        A_Scvk = np.zeros((S_to_include, self.C_nc_k.shape[1],
                          self.C_nv_k.shape[1], self.C_nc_k.shape[2]), 
                          dtype = complex)
        for i in range(S_to_include):
          A_Scvk[i] = eigvecs[:,i].reshape((self.C_nc_k.shape[1],
                        self.C_nv_k.shape[1],
                        self.C_nc_k.shape[2]))
        H_ab_k, grad_H_ab_k = self.get_Hk_and_grad_Hk_wannier()

      elif method == "tb":
        A_Scvk = np.zeros((S_to_include,self.C_nc_kplusQ.shape[2],
                           self.C_nv_k.shape[1], self.C_nc_kplusQ.shape[3]),
                           dtype = complex)
        for i in range(S_to_include):
          A_Scvk[i] = eigvecs[:,i].reshape((self.C_nc_kplusQ.shape[2],
                        self.C_nv_k.shape[1],
                        self.C_nc_kplusQ.shape[3]))
        H_ab_k, grad_H_ab_k = self.load_Hk_and_grad_Hk()
        
      if rank == root:
        print_f()
        print_f("Computed Fourier transformed Hamiltonian and it's derivative.")
        print_f("Computing dipole operators.")
        print_f(f"Including {S_to_include} BSE eigenvalues based on photon energies.")
        
      # Get how many BSE eigenvalues to work with
      Emin = np.min(eigvals)
      threshold = Emin + photon_max 
      S_to_include = np.sum(eigvals <= threshold)
      Slist = self.distribute_full(S_to_include, rank, size)
      local_size = Slist[1] - Slist[0]
      wanfuncs_pos = self.wan2bse.get_WF_loc()
      num_wan_orb = wanfuncs_pos.shape[0]

      # IM: get_distvec returns t_ab = (t_b - t_a) 
      # As a result, we will do (\GradH_k_ab - t_ab*H_k_ab)
      # For more details, see cyfunc.pyx
      time_1 = time.time()
      tvec_ab = self.potential.build_tvec_ab(np.zeros(3), wanfuncs_pos, n_images=1)
      time_tmp = time.time() - time_1 
      if rank == root:
        print_f(f"Time taken to get t_ab = (t_b-t_a): {time_tmp:.3f} secs.")

      # # PyMEX test
      # from test_pymex import test_build_tvec_ab
      # test_build_tvec_ab(self.potential, wanfuncs_pos) 

      # Local computation of conductivity
      local_conductivity = np.zeros((local_size, 3), dtype=np.complex128)
      if method == "dft":
        for i_local, i_global in enumerate(range(Slist[0], Slist[1])):
          local_conductivity[i_local, :] = compute_conductivity(
              A_Scvk[i_global],
              self.C_nc_k,
              self.C_nv_k,
              grad_H_ab_k,
              H_ab_k,
              tvec_ab,
          )
      elif method == "tb":
        for i_local, i_global in enumerate(range(Slist[0], Slist[1])):
          local_conductivity[i_local, :] = compute_conductivity(
              A_Scvk[i_global],
              self.C_nc_kplusQ[Gamma_idx,:,:,:],
              self.C_nv_k,
              grad_H_ab_k,
              H_ab_k,
              tvec_ab,
          )

      # Prepare data to send to root
      local_indices = np.arange(Slist[0], Slist[1], dtype=np.int32)

      if rank == root:
        print_f(f"Writing dipole operators to {Conductivity_file}")
        try:
          f = h5py.File(Conductivity_file, "w")
          dset = f.create_dataset(
            "optical_conductivity",
            shape=(S_to_include, 3),
            dtype=np.complex128,
          )
        except Exception as e:
          print_f(f"Error creating HDF5 file {Conductivity_file}: {e}")
          comm.Abort(1)
      else:
        dset = None

      self._chunked_gather_and_write_conductivity(
        local_indices, local_conductivity, dset,
        completion_msg=f"Conductivity written to {Conductivity_file}.")
      comm.Barrier()
      if rank == root:
        f.close()

      # Save absorption spectra
      if absorption_spin == "unpolarized" or\
        absorption_spin == "polarized":
        if rank == root:
          print_f(f"Reading {Conductivity_file} to extract the ansorption spectra.")
          with h5py.File(Conductivity_file, "r") as f:
            dset = f["optical_conductivity"][:]  
            x_component = dset[:, 0]
            y_component = dset[:, 1]
            z_component = dset[:, 2]
            f_S_x = np.abs(x_component)**2 
            f_S_y = np.abs(y_component)**2 
            f_S_z = np.abs(z_component)**2 
          omega_photon = np.arange(Emin + photon_min, Emin + photon_max + step, step)
          smeared_x = self.set_gaussian_smearing(omega_photon, eigvals[:S_to_include], 
                                                f_S_x, sigma)
          smeared_y = self.set_gaussian_smearing(omega_photon, eigvals[:S_to_include], 
                                                f_S_y, sigma)
          smeared_z = self.set_gaussian_smearing(omega_photon, eigvals[:S_to_include], 
                                                f_S_z, sigma)
          data_to_save = np.column_stack((omega_photon, smeared_x, smeared_y, smeared_z))
          header = "photon_energy_eV smeared_x smeared_y smeared_z"
          filename = (f"sigma_{absorption_spin}_tb.txt"
                        if method == "tb"
                        else f"sigma_{absorption_spin}.txt")
          np.savetxt(filename, data_to_save, header=header)
          print_f(f"{filename} written for plotting absoprtion spectra.")
      else:
        print_f("Not implemeted error! SOC")
        comm.Abort(1)

    else:
      if rank == root:
        print_f("Not Implemented error! Use Gamma_only (Q=0)")
      comm.Abort(1)


  def set_gaussian_smearing(self, omega_photon, omega_S, f_S, sigma):
    """
    Gaussian smeared values. 
    """
    prefactor = 1 / (np.sqrt(2 * np.pi) * sigma)
    omega_diff = omega_photon[:, None] - omega_S[None, :]
    gaussians = prefactor * np.exp(-0.5 * (omega_diff / sigma) ** 2)
    smeared = np.dot(gaussians, f_S)
    return smeared      


  def load_and_distribute_hdf5_array(self, filename, dataset_key,
                                    chunk_size=10_000_000):
    """
    Generic function to load the data from an HDF5 file on root node
    and distribute across all in chunks.
    """
    if rank == root:
      with h5py.File(filename, 'r') as f:
        data = np.array(f[dataset_key])
    else:
      data = None

    shape = comm.bcast(data.shape if rank == root else None, root=root)
    dtype = comm.bcast(data.dtype if rank == root else None, root=root)

    if rank != root:
      data = np.empty(shape, dtype=dtype)

    flat = data.ravel()
    total = flat.size

    for i in range(0, total, chunk_size):
      end = min(i + chunk_size, total)
      comm.Bcast(flat[i:end], root=root)
    return data

  #  if rank == root:
  #    print_f("Parallelizing over photon enrgies")
  #  # Get local E_ph points for the ranks
  #  Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
  #  print_f("%d-th rank handles %d photon enrgies"%\
  #         (rank,(Elist[1]-Elist[0])))

  #  # Synchronize
  #  comm.Barrier()

  #  t2 = time.time()
  #  # Calculation runs without modifying exciton
  #  # eigenvalues; See the perturbative inclusion 
  #  # of spin-orbit-coupling below. 

  #  if self.absorp[1].casefold() == "full":
  #    # HDF5 in parallel
  #    f = h5py.File("SIGMA_full.hdf5", "w", libver='latest',\
  #                  driver='mpio',comm=comm)
  #    dset = f.create_dataset("sigma_xx",\
  #           ((E_ph.shape[0],)),\
  #           dtype='float')
  #    # Computation of \Sigma_xx
  #    # Cythonized version
  #    # Every rank computes conductivity
  #    # based on the photon energies it handles
  #    tmp_dim = Elist[1]-Elist[0]
  #    tmp = np.zeros((tmp_dim), dtype=float)
  #    t3 = time.time()
  #    for i in range(Elist[0], Elist[1]):

  #      # Modified - 13/11/2023.
  #      if self.parallel[1].casefold() == "thread":
  #        #print_f("positions x coord", pos[:,0])
  #        #print_f("sigma", sigma)
  #        #print_f("E_ph[i]", E_ph[i])
  #        tmp[i-Elist[0]] = sigma_xx_full_E_thread(eigval,eigvnew,\
  #                          self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
  #                          self.Hk, pos[:,0],\
  #                          E_ph[i],sigma).real
  #      else:
  #        tmp[i-Elist[0]] = sigma_xx_full_E(eigval,eigvnew,\
  #                          self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
  #                          self.Hk, pos[:,0],\
  #                          E_ph[i],sigma).real

  #    t4 = time.time()
  #    if rank == root:
  #      print_f("Calculations done. Will collect")
  #      print_f("Time for sigma calculations:%.4f secs."%(t4-t3))

  #    with dset.collective:
  #      # Collects from every rank
  #      dset[Elist[0]:Elist[1]] = tmp
  #    f.close()
  #    if rank == root:
  #      print_f("Conductivity data written to HDF5")
  #      print_f("Time for sigma-parallelization:%.4f secs"\
  #               %(time.time()-t2))
  #    # Synchronize
  #    comm.Barrier()
#
#    # Calculation runs with the modification of BSE 
#    # eigenvalues in order to capture the SOC.
#    # Based on: Phys. Rev. Lett. 111, 216805 (2013).
#    elif self.absorp[1].casefold() == "perturbation":
#      # Spin-Orbit-Coupling (SOC) class
#      # -- SOC --
#      # Look at the ordering
#      if rank == root:
#        print_f()
#        print_f("----------------------------")
#        print_f("Adding SOC as a perturbation")
#        print_f("----------------------------")
#      self.soc = SOC(self.dft[0,0], self.absorp[5],\
#                 self.absorp[6],self.absorp[2],self.absorp[3],\
#                 self.absorp[4])
#
#      # Unfold the cvk for moiré to unit-cell c_uv_uk_u
#      if self.absorp[2] == "True":
#        if rank == root:
#          print_f()
#          print_f("Ideally requires unfolding")
#          print_f("However, not implemented yet")
#        comm.Abort(1)
#
#      # Unit-cell calculations
#      elif self.absorp[2]== "False":
#        if rank == root:
#          print_f("Don't require unfolding")
#          print_f("Unit-cell calculations")
#        # Get Delta_nsk for all the bands 
#        self.soc.get_Delta_nsk_unit()
#        # PLOT
#        #import matplotlib.pyplot as plt
#        #for i in range(Delta_cvsk.shape[2]):
#        #  plt.plot(Delta_cvsk[1,1,i,:])
#        #plt.show()
#        # Compute \Delta_{vcsk} (see function)
#        Delta_cvsk = self.get_Delta_cvsk_unit()
#        eigval_s = np.zeros((eigval.shape[0],2))
#        eigval_ = Eigval_perturb(eigval,eigvnew,\
#                   Delta_cvsk, eigval_s)
#        eigval_s[:,:] = eigval_
#        # Minimum and maximum window for conductivity calculations
#        # All input parameters are in eV.
#        xmin = np.min(eigval_s)
#        E_ph = np.arange(xmin+self.ephparam[0], xmin+self.ephparam[1],\
#                         self.ephparam[2])
#        sigma = self.ephparam[3]
#        # Photon enrgies to a file
#        if rank == root:
#          g = h5py.File("E_ph_per.hdf5", "w", libver='latest')
#          dset = g.create_dataset("photon",\
#                 ((E_ph.shape[0])),\
#                   dtype='float')
#          dset[:] = E_ph
#          g.close()
#
#        if rank == root:
#          print_f("Parallelizing over photon enrgies")
#        # Get local E_ph points for the ranks
#        Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
#        print_f("%d-th rank handles %d photon enrgies"%\
#               (rank,(Elist[1]-Elist[0])))
#
#        # Synchronize
#        comm.Barrier()
#
#        # HDF5 in parallel
#        f = h5py.File("SIGMA_per_unit.hdf5", "w", libver='latest',\
#                      driver='mpio',comm=comm)
#        dset = f.create_dataset("sigma_xx",\
#               ((E_ph.shape[0],)),\
#               dtype='float')
#        # IM : Modified - 13/11/2023
#        # Computation of \Sigma_xx
#        # Cythonized version
#        # Every rank computes conductivity
#        # based on the photon energies it handles
#        tmp_dim = Elist[1]-Elist[0]
#        tmp = np.zeros((tmp_dim), dtype=float)
#        for i in range(Elist[0], Elist[1]):
#          if self.parallel[1].casefold() == "thread":
#            tmp[i-Elist[0]] = sigma_xx_per_E_thread(eigval_s,eigvnew,\
#                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
#                            self.Hk, pos[:,0],\
#                            E_ph[i],sigma).real
#          else:
#            tmp[i-Elist[0]] = sigma_xx_per_E(eigval_s,eigvnew,\
#                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
#                            self.Hk, pos[:,0],\
#                            E_ph[i],sigma).real
#        if rank == root:
#          print_f("Calculations done. Will collect")
#        # Delete un-necessary things
#        del eigval_s; del eigvnew
#
#        with dset.collective:
#          # Collects from every rank
#          dset[Elist[0]:Elist[1]] = tmp
#        f.close()
#        if rank == root:
#          print_f("Conductivity data written to HDF5")
#          print_f("Time for sigma-parallelization:%.4f secs"\
#                   %(time.time()-t2))
#        # Synchronize
#        comm.Barrier()     
#
#      # Based on unit-cell calculations
#      # PMU approach
#      elif self.absorp[2].casefold() == "pmu":
#        if rank == root:
#          print_f()
#          print_f("WARNING: Moiré requires unfolding, ideally")
#          print_f("Will be using a simplified version :D")
#          print_f("Poor Man's Unfolding (PMU)")
#          print_f()
#        # 
#        # Compute \Delta_{vcsk} (see function)
#        Delta_cvsk = self.get_Delta_cvsk_pmu()
#        eigval_s = np.zeros((eigval.shape[0],2))
#        eigval_ = Eigval_perturb(eigval,eigvnew,\
#                   Delta_cvsk, eigval_s)
#        eigval_s[:,:] = eigval_
#        # Minimum and maximum window for conductivity calculations
#        # All input parameters are in eV.
#        xmin = np.min(eigval_s)
#        E_ph = np.arange(xmin+self.ephparam[0], xmin+self.ephparam[1],\
#                         self.ephparam[2])
#        sigma = self.ephparam[3]
#
#        # Photon enrgies to a file
#        if rank == root:
#          g = h5py.File("E_ph_per_pmu.hdf5", "w", libver='latest')
#          dset = g.create_dataset("photon",\
#                 ((E_ph.shape[0])),\
#                   dtype='float')
#          dset[:] = E_ph
#          g.close()
#
#        if rank == root:
#          print_f("Parallelizing over photon enrgies")
#        # Get local E_ph points for the ranks
#        Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
#        print_f("%d-th rank handles %d photon enrgies"%\
#               (rank,(Elist[1]-Elist[0])))
#
#        # Synchronize
#        comm.Barrier()
#
#        # HDF5 in parallel
#        f = h5py.File("SIGMA_per_pmu.hdf5", "w", libver='latest',\
#                      driver='mpio',comm=comm)
#        dset = f.create_dataset("sigma_xx",\
#               ((E_ph.shape[0],)),\
#               dtype='float')
#        # Computation of \Sigma_xx
#        # Cythonized version
#        # Every rank computes conductivity
#        # based on the photon energies it handles
#        tmp_dim = Elist[1]-Elist[0]
#        tmp = np.zeros((tmp_dim), dtype=float)
#        for i in range(Elist[0], Elist[1]):
#          if self.parallel[1].casefold() == "thread":
#            tmp[i-Elist[0]] = sigma_xx_per_E_thread(eigval_s,eigvnew,\
#                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
#                            self.Hk, pos[:,0],\
#                            E_ph[i],sigma).real
#          else:
#            tmp[i-Elist[0]] = sigma_xx_per_E(eigval_s,eigvnew,\
#                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
#                            self.Hk, pos[:,0],\
#                            E_ph[i],sigma).real
#        if rank == root:
#          print_f("Calculations done. Will collect")
#        # Delete un-necessary things
#        del eigval_s; del eigvnew
#
#        with dset.collective:
#          # Collects from every rank
#          dset[Elist[0]:Elist[1]] = tmp
#        f.close()
#        if rank == root:
#          print_f("Conductivity data written to HDF5")
#          print_f("Time for sigma-parallelization:%.4f secs"\
#                   %(time.time()-t2))
#        # Synchronize
#        comm.Barrier()     
#      else:
#        if rank == root:
#          print_f("Unknwon keyword found in Absorption type")
#          print_f("Exiting...")
#        comm.Abort(1)  
#
#
#  def get_Delta_cvsk_unit(self):
#    """
#    \Delta_{cvsk} = (E_ck - E_vk) +  \\ at the DFT and non-polar\
#                    (\delta_csk - delta_vsk) + \\ SOC contrib.\
#                    (rigid shift for GW at K-point)
#                    s = 2
#    NOTE: Rigid-shift is likely going to fail in general cases
#          But for now, we will *not* use it.
#          The c,v,k are the same as for DFT single-particle input.
#          \delta_csk/\delta_vsk read from "DELTA.HDF5" file.
#          Also, E_ck -E_vk is not included in this part.
#    """
#    #-----
#    spin = 2
#    shift = 0.0 # eV
#    #-----
#    Delta_cvsk = np.zeros((self.E_ck.shape[0], self.E_vk.shape[0],\
#                           spin, self.E_ck.shape[1]), dtype=float)
#
#    for c in range(self.E_ck.shape[0]):
#      for v in range(self.E_vk.shape[0]):
#        for s in range(spin):
#          for k in range(self.E_ck.shape[1]):
#            Delta_cvsk[c,v,s,k] = \
#          (self.get_Delta_csk_unit()[c,s,k]-\
#           self.get_Delta_vsk_unit()[v,s,k])+\
#           shift 
#          #+\
#          #(self.E_ck[c,k] - self.E_vk[v,k])
#    return Delta_cvsk
#
#
#  def get_Delta_cvsk_pmu(self):
#    """
#    \Delta_{cvsk} = (E_ck - E_vk) +  \\ at the DFT and non-polar\
#                    (\delta_csk - delta_vsk) + \\ SOC contrib.\
#                    (rigid shift for GW at K-point)
#                    s = 2
#    NOTE: Rigid-shift is likely going to fail in general cases
#          But for now, we will *not* use it.
#          The c,v,k are the same as for DFT single-particle input.
#          \delta_csk/\delta_vsk read from "DELTA.HDF5" file.
#          Also, E_ck -E_vk is not included in this part.
#    """
#    #-----
#    spin = 2
#    shift = 0.0 # eV
#    #-----
#    Delta_cvsk = np.zeros((self.E_ck.shape[0], self.E_vk.shape[0],\
#                           spin, self.E_ck.shape[1]), dtype=float)
#    for c in range(self.E_ck.shape[0]):
#      for v in range(self.E_vk.shape[0]):
#        for s in range(spin):
#          for k in range(self.E_ck.shape[1]):
#            Delta_cvsk[c,v,s,k] = \
#          (self.get_Delta_csk_pmu()[c,s,k]-\
#           self.get_Delta_vsk_pmu()[v,s,k])+\
#           shift 
#          #+\
#          #(self.E_ck[c,k] - self.E_vk[v,k])
#    return Delta_cvsk
#
#
#
#  def get_eigval_perturb_unit(self):
#    """
#    Omega_Ms = Omega_M + D_vcsk;
#               D_vcsk = \sum_{vck} |A^M_{vcsk}|^{2} \Delta_{vcks}
#               and Omega_M are the BSE eigenvalues with non-polar.
#               DFT calculations. 
#    NOTE: The valence and conduction band assignments are the same
#          as with DFT single-particle eigenvalues. We don't use the
#          WANNIER90 eigenvalues at all throughout. 
#    Ref: Qiu et al., Phys. Rev. Lett. 111, 216805 (2013).
#    """
#     
#
#  def factor(self):
#    """
#    """
#    return 1.0
#

  def get_interpolated_points(self, hspoints, numpoints_per_segment):
    """
    Generate interpolated points between consecutive high-symmetry points.
    Returns original points if interpolation is not needed.
    Args:
      hspoints (list/np.ndarray): High-symmetry points (e.g., [[0.5,0,0], [0,0,0]]).
      numpoints_per_segment (int): Points per segment. Returns original if ==1.
    Returns:
      np.ndarray: Flattened interpolated points, shape (N, 3).
    """
    hspoints = np.asarray(hspoints)
    if len(hspoints) == 1 or numpoints_per_segment == 1:
      return hspoints
    interpolated_segments = []
    for i in range(len(hspoints) - 1):
      p1 = hspoints[i]; p2 = hspoints[i + 1]
      t = np.linspace(0, 1, numpoints_per_segment)
      segment = p1 + t[:, np.newaxis] * (p2 - p1)
      interpolated_segments.append(segment)
    return np.vstack(interpolated_segments)


  def compute_all_Q(self, k_points):
    """
    Compute all Q = k" - k for an array of k-points.
    Q is in fractional coordinates. 
    Args:
      k_points (np.ndarray): Array of shape (N, 3) representing N k-points in 
                            fractional coordinates.
    Returns:
      np.ndarray: Array of shape (N^2, 3) containing all Q vectors.
    """
    N = len(k_points); Q_list = []
    # Generate all Q = k_i - k_j
    for i in range(N):
      for j in range(N):
        Q = (k_points[i] - k_points[j]) % 1
        Q_list.append(Q)
    Q_all = np.array(Q_list)  
    # Remove duplicate Q vectors (keep unique ones)
    Q_unique = np.unique(Q_all, axis=0)
    return Q_unique


  def find_nearby_points(self, Qpoints, Qpoints_large, 
                        tolerance=1e-3, decimals=6):
    """
    Faster for large datasets using KDTree.
    """
    Qpoints_rounded = np.round(Qpoints, decimals=decimals)
    Qpoints_large_rounded = np.round(Qpoints_large, decimals=decimals)

    tree = KDTree(Qpoints_large_rounded)
    matching_indices = tree.query_ball_point(Qpoints_rounded, r=tolerance)
    matching_indices = np.unique(np.concatenate(matching_indices)).astype(np.int64)
    return np.unique(Qpoints_large_rounded[matching_indices], axis=0)


  def write_Qpoints_to_file(self, Qpoints, filename='Selected_Q.txt'):
    """
    Writes Nx3 Qpoints array to text file.
    Args:
      Qpoints (np.ndarray): Nx3 array of Q-points
      filename (str): Output filename
    """
    header = f"Selected Q-points (total {len(Qpoints)})\n" \
              "Qx\t\tQy\t\tQz"
  
    # Format numbers to scientific notation with 8 decimals
    np.savetxt(filename, 
               Qpoints,
               header=header,
               fmt='%.6f',
               delimiter='\t')
    return None

    
  def find_reshuffle_indices(self, k_points, Qpoints, tol=1e-3):
    """
    Finds indices that reshuffle k to match k" = k + Q.
    Args:
      k_points (np.ndarray): Nx3 original k-points
      Qpoints (np.ndarray): Mx3 Q-points (must produce exact reshuffling)
      tol (float): Numerical tolerance
    Returns:
      dict: {Q_index: permutation_array} for each Q that produces valid reshuffling
    """
    k_tree = KDTree(k_points)
    results = {}
    # Write selected Q-points to file
    self.write_Qpoints_to_file(Qpoints, filename='Selected_Q.txt')
    for Q_idx, Q in enumerate(Qpoints):
      k_dprime = np.mod(k_points + Q, 1)  # Compute all k"
      # Find which original k-point matches each k"
      _, orig_indices = k_tree.query(k_dprime, distance_upper_bound=tol)
      # Verify perfect reshuffling
      if np.setdiff1d(orig_indices, np.arange(len(k_points))).size == 0:
        results[Q_idx] = orig_indices  # Store permutation indices
    return results 

  def get_kplusQ_shuffle(self):
    """
    Get reshuffled k" = k + Q
    Args:
      k_points (np.ndarray): Nx3 original k-points
      Q_points (np.ndarray): Mx3 Q-points
    Returns:
      np.ndarray: Reshuffled k" points, shape (M, 3)
    """
    Q_points = self.config.get("bse", {}).get("qpoints", {}).get("excitonq", {})
    hspoints = self.config.get("bse", {}).get("qpoints", {}).get("hspoints", {})
    numpoints_per_segment = self.config.get("bse", {}).get("qpoints", {}
                                           ).get("num_points", {})
    if rank == root:
      print_f("Reshuffling k-points to obtain k + Q points.")
      print_f("Useful to reordering Cnmk to align with Cnmk+Q.")

      print_f(f"Q-path requested: {hspoints}")
      print_f(f"Points per segment along the Q-path: {numpoints_per_segment}")
      print_f("Searching for Q-points close to your input via reshuffling...")

      print_f("Step 1: Compute all possible Q-vectors from k-points.")
      print_f("Step 2: Identify Q-vectors close to the provided ones.")
      print_f("Step 3: Determine reshuffle indices to get k+Q from k.")

    # Step 1: Compute all Q vectors from k-points
    kpoints_reciprocal = self.wan2bse.get_kpoints_reciprocal()
    all_Q = self.compute_all_Q(kpoints_reciprocal)

    # Step 2: Find Q vectors close to provided ones (increase numpoints
    # to scan more points); Reduce 1000 to small number if you want to reduce.
    if len(hspoints) == 1 or numpoints_per_segment == 1:
      Q_provided = self.get_interpolated_points(hspoints, numpoints_per_segment)
      Q_selected = self.find_nearby_points(Q_provided, all_Q, tolerance=1e-3)
    else:
      Q_provided = self.get_interpolated_points(hspoints, 1000)
      Q_selected = self.find_nearby_points(Q_provided, all_Q, tolerance=1e-3)
    # Step 3: Find reshuffle indices for k + Q
    shuflling_idx = self.find_reshuffle_indices(kpoints_reciprocal, Q_selected)
    if rank == root:
      print_f(f"Selected Q: {Q_selected}")
    return shuflling_idx, Q_selected


  def write_exciton_H(self):
    """
    Save the electron-hole Hamiltonian to an HDF5 file using MPI I/O.

    Parameters:
      savefile (str): Output filename (default: "H_eh.hdf5").
    """
    self.get_precomputed_potential()

    Ham_file = self.config.get("io", {}).get("exciton_hamiltonian_file", {})
    if Ham_file is None:
      Ham_file = "H_" + self.config.get("excitation", {}) + ".hdf5"

    system = self.config.get("system", {})
    method = self.config.get("bse", {}).get("method", {})
    if rank == root:
      print_f()
      print_f(f"Constructing the BSE Hamiltonian with {system} ({method})")
      print_f("Please wait...")

    if method == "dft":
      shuffling_idx, Q = self.get_kplusQ_shuffle()
      num_c = self.C_nc_k.shape[1]
      num_v = self.C_nv_k.shape[1]; num_k = self.C_nv_k.shape[2]

      if rank == root:
        f = h5py.File(Ham_file, "w")
        dset = f.create_dataset(
          "exciton_H",
          shape=(len(Q), num_c, num_v, num_k, num_c, num_v, num_k),
          dtype=np.complex128,
          chunks=(1, 1, 1, 1, num_c, num_v, num_k),              # one row per chunk
          compression="gzip", compression_opts=4
        )
      else:
        dset = None

      comm.Barrier()

      # Process each Q-point
      for key, value in shuffling_idx.items():
        counter = 0
        C_nc_kplusQ = self.C_nc_k[:, :, value]
        E_ckplusQ = self.E_ck[:, value]
        myQ = crys2ang(self.wan2bse.get_reciprocal(), Q[key])
        plist = self.distribute_full(num_c * num_v * num_k, rank, size)
        c_, v_, k_ = np.mgrid[0:num_c, 0:num_v, 0:num_k]
        myelem = np.column_stack((c_.ravel(), v_.ravel(), k_.ravel()))
        assigned = myelem[plist[0]:plist[1]]
        mywanfuncs = self.wanfuncs.astype(int)

        local_data = {}
        for c, v, k in assigned:
          local_data[(c, v, k)] = H_optfull_thread(
            C_nc_kplusQ, self.C_nv_k,
            E_ckplusQ, self.E_vk,
            c, v, k,
            mywanfuncs, self.Rvec, self.weight_Rvec, self.kvec, myQ,
            self.V_r_keld, self.V_r_coul
          )
          if rank == root:
            counter = counter + 1
            print_f(f"Root Computed [{counter}/{plist[1]-plist[0]}]")

        # Chunked gather to avoid memory exhaustion issues
        self._chunked_gather_and_write_H(local_data, dset, key, len(Q),
                                         completion_msg=f"BSE Hamiltonian is written to {Ham_file}.")
        comm.Barrier()

      if rank == root:
        f.close()

    elif method == "tb":

      hspoints = self.config.get("bse", {}).get("qpoints", {}).get("hspoints", {})
      numpoints_per_segment = self.config.get("bse", {}).get("qpoints", {}
                                           ).get("num_points", {})
      Qvec = self.get_interpolated_points(hspoints, numpoints_per_segment)

      num_Q = Qvec.shape[0]; num_c = self.C_nc_kplusQ.shape[2]
      num_v = self.C_nv_k.shape[1]; num_k = self.C_nv_k.shape[2]

      if rank == root:
        f = h5py.File(Ham_file, "w")
        dset = f.create_dataset(
          "exciton_H",
          shape=(num_Q, num_c, num_v, num_k, num_c, num_v, num_k),
          dtype=np.complex128,
          chunks=(1, 1, 1, 1, num_c, num_v, num_k),              # one row per chunk
          compression="gzip", compression_opts=4
        )
      else:
        dset = None

      comm.Barrier()

      for iQ in range(num_Q):
        counter = 0
        myQ = Qvec[iQ]
        plist = self.distribute_full(num_c * num_v * num_k, rank, size)
        c_, v_, k_ = np.mgrid[0:num_c, 0:num_v, 0:num_k]
        myelem = np.column_stack((c_.ravel(), v_.ravel(), k_.ravel()))
        assigned = myelem[plist[0]:plist[1]]
        mywanfuncs = self.wanfuncs.astype(int)

        local_data = {}
        for c, v, k in assigned:
          local_data[(c, v, k)] = H_optfull_thread(
            self.C_nc_kplusQ[iQ], self.C_nv_k,
            self.E_ckplusQ[iQ], self.E_vk,
            c, v, k,
            mywanfuncs, self.Rvec, self.weight_Rvec, self.kvec, myQ,
            self.V_r_keld, self.V_r_coul
          )
          if rank == root:
            counter = counter + 1
            print_f(f"Root Computed [{counter}/{plist[1]-plist[0]}]")

        # Chunked gather to avoid memory exhaustion issues
        self._chunked_gather_and_write_H(local_data, dset, iQ, num_Q,
                                         completion_msg=f"BSE Hamiltonian is written to {Ham_file}.")
        comm.Barrier()

      if rank == root:
        f.close()

    # Free potentials — not needed after this point
    del self.V_r_keld; self.V_r_keld = None
    del self.V_r_coul; self.V_r_coul = None
    self._potential_loaded = False
    if rank == root:
      print_f("Memory freed after write_exciton_H.")
  

  def diagonalize_bse_lapack(self, io_files, t_diag, t0, num_eigs=None):
    """
    Single-node BSE Hamiltonian diagonalization using LAPACK.
    Runs only on root -- use for small matrices or testing.
    Parameters
    ----------
    io_files  : dict [Ham_file, eigenval_file, eigenvec_file]
    t_diag    : float [diagonalization start time]
    t0        : float [global start time]
    num_eigs  : int or None -- None=full spectrum, int=lowest N eigenpairs
    """
    if rank == root:
      print_f(f"Diagonalizing BSE Hamiltonian, read from {io_files['Ham_file']}.")
      with h5py.File(io_files["Ham_file"], "r") as f, \
          h5py.File(io_files["eigenval_file"], "w") as eigval_file, \
          h5py.File(io_files["eigenvec_file"], "w") as eigvec_file:
        dset = f["exciton_H"]
        nQ   = dset.shape[0]
        for iQ in range(nQ):
          H_eh = dset[iQ]                              # decompress one chunk at a time
          m1, n1, p1, m2, n2, p2 = H_eh.shape
          re   = m1 * n1 * p1
          H_eh = H_eh.reshape((re, re))
          self.is_hermitian(H_eh)
          # Full or partial spectrum
          full_spectrum = (num_eigs is None) or (num_eigs >= re)
          if full_spectrum:
            solver = "numpy.linalg.eigh (full spectrum)"
            eigval, eigvec = np.linalg.eigh(H_eh)
          else:
            solver = f"scipy.linalg.eigh (lowest {num_eigs} eigenpairs)"
            from scipy.linalg import eigh
            eigval, eigvec = eigh(H_eh, subset_by_index=[0, num_eigs - 1])
          if iQ == 0:
            print_f(f"Using solver: {solver} | matrix size: {re}x{re}")
          eigval_file.create_dataset(f"Q{iQ}", data=eigval)
          eigvec_file.create_dataset(f"Q{iQ}", data=eigvec)
          print_f(f"  Q{iQ}: {eigval[:min(24, len(eigval))]}")
      print_f(f"Eigensolutions written to: {io_files['eigenval_file']}, {io_files['eigenvec_file']}")
      print_f(f"Time spent during diagonalization: {time.time() - t_diag:.3f} secs.")
      print_f(f"Time spent so far: {time.time() - t0:.3f} secs.")


  def diagonalize_bse_elpa(self, io_files, t_diag, t0, num_eigs=None):
    """
    Diagonalise BSE Hamiltonian using ELPA complex Hermitian solver.
    """
    # File names and setup
    ham_file      = io_files["Ham_file"]
    reshaped_file = ham_file.replace(".hdf5", "_reshaped.hdf5")
    eigval_file   = io_files["eigenval_file"]
    eigvec_file   = io_files["eigenvec_file"]
    job_id        = os.environ.get("SLURM_JOB_ID", "local")
    tmp_dir       = os.path.join(os.getcwd(), f"tmp_eigvec_{job_id}")

    self._elpa_preprocess_ham(ham_file, reshaped_file)
    self._elpa_setup_tmp(tmp_dir)
    self._elpa_setup_outputs(eigval_file, eigvec_file)

    # Prepare for diagonalization with ELPA
    if rank == root:
      with h5py.File(reshaped_file, "r", locking=False) as f:
        nQ = len(f.keys())
        re = f["Q0"].shape[0]
    else:
      nQ = re = None
    nQ = comm.bcast(nQ, root=root)
    re = comm.bcast(re, root=root)

    nev_solve = re if num_eigs is None else min(num_eigs, re)
    nblk      = min(64, max(1, re // size))

    if rank == root:
      print_f(f"Using ELPA (complex) | size: {re}x{re} | "
            f"nev_solve: {nev_solve} | nblk: {nblk} | ranks: {size}")

    for iQ in range(nQ):

      # Create distributed matrix for ELPA
      a = DistributedMatrix.from_comm_world(re, nev_solve, nblk,
                                            dtype=np.complex128)
      rows, cols    = a.get_global_index(np.arange(a.na_rows),
                                        np.arange(a.na_cols))
      h_rows_needed = np.unique(rows[rows < re])
      H_local       = self._elpa_read_H_rows(reshaped_file, iQ, h_rows_needed)
      self._elpa_fill_matrix(a, rows, cols, H_local, h_rows_needed)

      # Diagonalise at iQ
      if rank == root:
        print_f(f"  Q{iQ}: diagonalizing...")
        t1 = time.time()

      diag_data  = a.compute_eigenvectors()
      eigval_raw = diag_data["eigenvalues"]
      eigvec_dm  = diag_data["eigenvectors"]

      if rank == root:
        print_f(f"  Q{iQ}: diagonalization done in {time.time()-t1:.3f}s")
        t2 = time.time()

      # Write tmp eigenvectors
      evec_rows, evec_cols = eigvec_dm.get_global_index(
        np.arange(eigvec_dm.na_rows), np.arange(eigvec_dm.na_cols))
      self._elpa_write_tmp(eigvec_dm, evec_rows, evec_cols,
                          nev_solve, re, iQ, tmp_dir)
      comm.Barrier()

      # Assemble eigenvectors on root and write final outputs
      if rank == root:
        print_f(f"  Q{iQ}: all ranks wrote tmp in {time.time()-t2:.3f}s")
        # self._elpa_assemble(re, nev_solve, eigval_raw, iQ,
        #                     tmp_dir, eigval_file, eigvec_file, reshaped_file)
        self._elpa_assemble(re, nev_solve, eigval_raw, iQ,
                            tmp_dir, eigval_file, eigvec_file)
        print_f(f"  Q{iQ}: total post-diag {time.time()-t2:.3f}s")
      comm.Barrier()

    # Cleanup temporary files
    self._elpa_cleanup_tmp(tmp_dir, nQ)

    if rank == root:
      print_f(f"Eigensolutions written to: {eigval_file}, {eigvec_file}")
      print_f(f"Diagonalization time: {time.time()-t_diag:.3f}s")
      print_f(f"Total time:           {time.time()-t0:.3f}s")

  # ELPA diagonalization private helper functions
  def _elpa_preprocess_ham(self, ham_file, reshaped_file):
    """
    Reshape(iQ,c,v,k,c,v,k) → (iQ, re, re) where re = num_c*num_v*num_k
     - Done only on root, then reshaped file is read by all ranks in parallel.
     - Uses chunked read/write to avoid memory issues for large matrices.
     - If reshaped file already exists, skip preprocessing (useful for testing).
    """
    if rank == root:
      if os.path.exists(reshaped_file):
        print_f("Reshaped file exists, skipping preprocessing.")
      else:
        print_f(f"Preprocessing {ham_file} → {reshaped_file}")
        with h5py.File(ham_file, "r", locking=False) as f_in, \
            h5py.File(reshaped_file, "w", locking=False) as f_out:
          for iQ in range(f_in["exciton_H"].shape[0]):
            H_raw             = f_in["exciton_H"][iQ]
            m1, n1, p1,_,_,_ = H_raw.shape
            re                = m1 * n1 * p1
            H_eh              = H_raw.reshape(re, re)
            del H_raw
            self.is_hermitian(H_eh)

            # Chunked write to avoid memory issues; 
            ds = f_out.create_dataset(f"Q{iQ}", shape=(re, re),
                                      dtype=np.complex128, chunks=(256, re))
            for start in range(0, re, 256):
              end = min(start + 256, re)
              ds[start:end, :] = H_eh[start:end, :]
              f_out.flush()
            del H_eh
            print_f(f"  Q{iQ}: reshaped to ({re}, {re})")
    comm.Barrier()


  def _elpa_setup_tmp(self, tmp_dir):
    """
    Create temporary directory for storing partial 
    eigenvector results from each rank.
    """
    if rank == root:
      if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
      os.makedirs(tmp_dir)
      print_f(f"Created {tmp_dir}/")
    comm.Barrier()


  def _elpa_setup_outputs(self, eigval_file, eigvec_file):
    """
    Remove existing output files on root to avoid appending to old data.
    """
    if rank == root:
      for f in [eigval_file, eigvec_file]:
        if os.path.exists(f):
          os.remove(f)
    comm.Barrier()


  def _elpa_read_H_rows(self, reshaped_file, iQ, h_rows_needed):
    """
    Read only the needed rows of H for the current rank.
    """
    with h5py.File(reshaped_file, "r", locking=False) as f:
      return f[f"Q{iQ}"][h_rows_needed, :]
    

  def _elpa_fill_matrix(self, a, rows, cols, H_local, h_rows_needed):
    t0 = time.time()

    re = H_local.shape[1]

    valid_rows = rows < re
    valid_cols = cols < re

    rows_f = rows[valid_rows]
    cols_f = cols[valid_cols]

    pos = np.searchsorted(h_rows_needed, rows_f)

    assert np.all(pos < len(h_rows_needed))
    assert np.all(h_rows_needed[pos] == rows_f)

    H_rows = H_local[pos, :]

    row_idx = np.where(valid_rows)[0]
    col_idx = np.where(valid_cols)[0]

    a.data[np.ix_(row_idx, col_idx)] = H_rows[:, cols_f]

    if rank == root:
        print_f(f"  fill_elpa_matrix: done in {time.time()-t0:.3f}s")


  def _elpa_write_tmp(self, eigvec_dm, evec_rows, evec_cols,
                      nev_solve, re, iQ, tmp_dir):
    valid_col_mask = evec_cols < nev_solve
    valid_row_mask = evec_rows < re

    valid_col_idx = evec_cols[valid_col_mask]   # global col indices in [0, nev_solve)
    valid_row_idx = evec_rows[valid_row_mask]   # global row indices in [0, re)

    # Extract the valid 2D block — no column expansion, no zero padding
    local_block = eigvec_dm.data[valid_row_mask, :][:, valid_col_mask]
    # shape: (n_valid_rows, n_valid_cols)

    np.save(os.path.join(tmp_dir, f"rows_rank{rank}_Q{iQ}.npy"), valid_row_idx)
    np.save(os.path.join(tmp_dir, f"cols_rank{rank}_Q{iQ}.npy"), valid_col_idx)
    np.save(os.path.join(tmp_dir, f"eigvec_rank{rank}_Q{iQ}.npy"), local_block)

  # def _elpa_assemble(self, re, nev_solve, eigval_raw, iQ,
  #                    tmp_dir, eigval_file, eigvec_file, reshaped_file):
  def _elpa_assemble(self, re, nev_solve, eigval_raw, iQ,
                     tmp_dir, eigval_file, eigvec_file):
    t0          = time.time()
    eigvec_full = np.zeros((re, nev_solve), dtype=np.complex128)

    all_row_idx = []
    all_col_idx = []
    for r in range(size):
      row_idx = np.load(os.path.join(tmp_dir, f"rows_rank{r}_Q{iQ}.npy"))
      col_idx = np.load(os.path.join(tmp_dir, f"cols_rank{r}_Q{iQ}.npy"))
      block   = np.load(os.path.join(tmp_dir, f"eigvec_rank{r}_Q{iQ}.npy"))
      eigvec_full[np.ix_(row_idx, col_idx)] = block
      all_row_idx.append(row_idx)
      all_col_idx.append(col_idx)

    # Sanity check — union of rows/cols must cover full range
    unique_rows = np.unique(np.concatenate(all_row_idx))
    unique_cols = np.unique(np.concatenate(all_col_idx))
    if len(unique_rows) != re:
      raise RuntimeError(
        f"Q{iQ}: row coverage error — got {len(unique_rows)} unique rows, expected {re}"
      )
    if len(unique_cols) != nev_solve:
      raise RuntimeError(
        f"Q{iQ}: col coverage error — got {len(unique_cols)} unique cols, expected {nev_solve}"
      )

    print_f(f"  assemble: read {size} tmp files in {time.time()-t0:.3f}s")

    eigenval          = eigval_raw[:nev_solve]
    # norms             = np.linalg.norm(eigvec_full, axis=0)
    # norms[norms == 0] = 1.0
    # eigvec            = eigvec_full / norms
    # print_f(f"  assemble: normalise done in {time.time()-t1:.3f}s")

    t2 = time.time()

    with h5py.File(eigval_file, "a") as ef, \
        h5py.File(eigvec_file, "a") as vf:
      ef.create_dataset(f"Q{iQ}", data=eigenval)
      vf.create_dataset(f"Q{iQ}", data=eigvec_full,
                        chunks=(min(256, re), nev_solve),
                        dtype=np.complex128)

    print_f(f"  assemble: write done in {time.time()-t2:.3f}s")
    print_f(f"  assemble: total {time.time()-t0:.3f}s")
    print_f(f"  Q{iQ}: eigenvalues[:24] = {eigenval[:min(24, nev_solve)]}")

    # # ── One-time residual check on Q0 — commenting out after confirming correctness ──
    # if iQ == 0:
    #   n_check  = min(10, nev_solve)
    #   V        = eigvec_full[:, :n_check]         # (re, n_check)
    #   lam      = eigenval[:n_check]
    #   with h5py.File(reshaped_file, "r", locking=False) as f:
    #     H      = f["Q0"][:]                  # (re, re) — remove if re is large
    #   residual = np.linalg.norm(H @ V - V * lam, axis=0)
    #   print_f(f"  residual check Q0 (first {n_check} vecs): "
    #           f"max={residual.max():.2e}, mean={residual.mean():.2e}")
    #   del H


  def _elpa_cleanup_tmp(self, tmp_dir, nQ):
    comm.Barrier()
    for iQ in range(nQ):
      for fname in [f"rows_rank{rank}_Q{iQ}.npy",
                    f"cols_rank{rank}_Q{iQ}.npy",
                    f"eigvec_rank{rank}_Q{iQ}.npy"]:
        path = os.path.join(tmp_dir, fname)
        if os.path.exists(path):
          os.remove(path)
    comm.Barrier()
    if rank == root and os.path.exists(tmp_dir):
      shutil.rmtree(tmp_dir)
      print_f(f"Cleaned up {tmp_dir}/")



  def diagon_BSE(self):
    """
    Loads and diagonalizes the BSE Hamiltonian for each Q point.
    At the moment, we support LAPACK (through numpy), ELPA 
    (Eigensolver Library for Parallel Applications), and 
    SLEPc (not publicly released yet). 
    """
    # Synchornize and time-track for diagonalization
    comm.Barrier()
    t_diag = time.time()

    io_files = self.set_exciton_io_files()
    diagon_method = self.config.get("diagonalize", {}).get("library", {})
    num_eigs = self.config.get("diagonalize", {}).get("num_eigs", None)

    # Diagonalization through "elpa" (scalable)
    if diagon_method == "elpa":
      if not HAS_ELPA:
        if rank == root:
          print_f("ERROR: pyelpa not found. Install ELPA or use 'lapack'.")
        comm.Abort(1)
      self.diagonalize_bse_elpa(io_files, t_diag, self.t0, num_eigs=num_eigs)

    # standard "lapack" routine (Basic diagonalization in the root node)
    elif diagon_method == "lapack":  
      self.diagonalize_bse_lapack(io_files, t_diag, self.t0, num_eigs=num_eigs)

    else:
      if rank == root:
        print_f(f"Unknown diagonalization library: {diagon_method}")
        print_f("Supported options are: 'lapack', 'elpa', 'slepc'")
        print_f("Exiting...")
      comm.Abort(1)


  def set_exciton_io_files(self):
    """
    Set up I/O files for exciton calculations.
    """
    prefix = self.config.get("excitation", "")
    files = {
      "Ham_file": self.config.get("io", {}).get("exciton_hamiltonian_file", f"H_{prefix}.hdf5"),
      "eigenval_file": self.config.get("io", {}).get("exciton_eigenval_file", f"Eigval_{prefix}.hdf5"),
      "eigenvec_file": self.config.get("io", {}).get("exciton_eigenvec_file", f"Eigvec_{prefix}.hdf5"),
      "conductivity_file": self.config.get("io", {}).get("exciton_conductivity_file", f"Conductivity_{prefix}.hdf5"),
      "wfn_rh_file":self.config.get("io", {}).get("exciton_wfn_rh_file", f"Wfn_rh_{prefix}.hdf5")
    }
    return files


  def distribute_full(self, num_p, rank, size):
    """
    Distributes num_p points across size processes in a balanced way.
    Args: 
      num_p: Total points to distribute (int)
      rank: Current MPI rank (int)
      size: Total number of processes (int)
    Returns:
      np.array: [start_idx, end_idx] for current rank
    """
    if num_p < size:
      if rank == 0:
        print_f("WARNING: Inefficient parallelisation (some processes will be idle).")
        print_f(f"Maxmimum (optimum) mpi processes for this is {num_p}")

    base = num_p // size
    rem = num_p % size

    # Distribute remainder among the first 'rem' ranks
    if rank < rem:
      start = rank * (base + 1)
      end = start + base + 1
    else:
      start = rem * (base + 1) + (rank - rem) * base
      end = start + base
    return np.array([start, end])


  def get_C_nc_k(self):
    """
    Returns the coefficients for conduction bands
    C_nc_k = C_nm_k[n][c][k]
    where n is the wannier index, c is the conduction band index
    and k is the k-point index.
    """
    method = self.config.get("bse", {}).get("method", {})
    cnmk_inp = self.config["bse"][method]["cnmk"]
    cb_start = cnmk_inp["cb_idx"][0] + cnmk_inp["cb_skip"]
    cb_end = cnmk_inp["cb_idx"][1] + cnmk_inp["cb_skip"] 
    # C_nm_k's read at root
    if method == "dft":
      if rank == root:
        C_nm_k = self.wan2bse.get_C()
        shape = C_nm_k.shape
        dtype = C_nm_k.dtype
      else:
        shape, dtype = None, None
    elif method == "tb":
      if rank == root:
        C_nm_k = self.get_C_tb("Eigvec_k_tb.hdf5")
        shape = C_nm_k.shape
        dtype = C_nm_k.dtype
      else:
        shape, dtype = None, None

    # Broadcast metadata
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)

    # Allocate array on all non-root ranks
    if rank != root:
      C_nm_k = np.empty(shape, dtype=dtype)

    # Broadcast in smaller chunks (adjust based on system memory)
    flat_array = C_nm_k.ravel()
    chunk_size = 10_000_000  # ~80MB for float64
    total_elements = flat_array.size

    for i in range(0, total_elements, chunk_size):
      end = min(i + chunk_size, total_elements)
      comm.Bcast(flat_array[i:end], root=root) 
    return self.c_nc_k(C_nm_k, cb_start, cb_end)


  def get_C_nc_kplusQ(self):
    """
    Returns the coefficients for conduction bands at k+Q.
    C_nc_kplusQ = C_nm_k[Q, n, c, kplusQ]
    where n is the wannier index, c is the conduction band index,
    kplusQ is the k-point index, and Q is the exciton momentum index.
    """
    method   = self.config.get("bse", {}).get("method", {})
    cnmk_inp = self.config["bse"][method]["cnmk"]
    cb_start = cnmk_inp["cb_idx"][0] + cnmk_inp["cb_skip"]
    cb_end   = cnmk_inp["cb_idx"][1] + cnmk_inp["cb_skip"]

    if method != "tb":
      if rank == root:
        print_f("You must set method to 'tb' to do this!")
        print_f("Exiting...")
      comm.Abort(1)

    if rank == root:
      C_nm_k = self.get_C_tb("Eigvec_kplusQ_tb.hdf5")

      # slice on root first — only num_Q x num_wann x num_c x num_k gets broadcast
      C_nc_kplusQ = self.c_nc_kplusQ(C_nm_k, cb_start, cb_end)
      del C_nm_k  # free 16 GB immediately, before broadcast starts
      shape = C_nc_kplusQ.shape
      dtype = C_nc_kplusQ.dtype
    else:
      shape, dtype = None, None

    # broadcast metadata so non-root ranks can allocate correctly
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)

    if rank != root:
      C_nc_kplusQ = np.empty(shape, dtype=dtype)

    # guarantee C-contiguous so ravel() returns a view, not a copy
    C_nc_kplusQ = np.ascontiguousarray(C_nc_kplusQ)

    # chunked Bcast — each chunk ~80 MB
    flat = C_nc_kplusQ.ravel()
    chunk_size = 10_000_000
    for i in range(0, flat.size, chunk_size):
      end = min(i + chunk_size, flat.size)
      comm.Bcast(flat[i:end], root=root)

    return C_nc_kplusQ


  def get_C_nv_k(self):
    """
    Returns the coefficients for valence bands only.
    C_nv_k = C_nm_k[n][v][k]
    where n is the wannier index, v is the valence band index,
    and k is the k-point index.
    """
    method    = self.config.get("bse", {}).get("method", {})
    cnmk_inp  = self.config["bse"][method]["cnmk"]
    vb_start  = cnmk_inp["vb_idx"][0] + cnmk_inp["vb_skip"]
    vb_end    = cnmk_inp["vb_idx"][1] + cnmk_inp["vb_skip"]

    if rank == root:
      if method == "dft":
        C_nm_k = self.wan2bse.get_C()
      elif method == "tb":
        C_nm_k = self.get_C_tb("Eigvec_k_tb.hdf5")

      # slice on root first — only num_wann x num_v x num_k gets broadcast
      C_nv_k = self.c_nv_k(C_nm_k, vb_start, vb_end)
      del C_nm_k  # free memory immediately, before broadcast starts
      shape = C_nv_k.shape
      dtype = C_nv_k.dtype
    else:
      shape, dtype = None, None

    # broadcast metadata so non-root ranks can allocate correctly
    shape = comm.bcast(shape, root=root)
    dtype = comm.bcast(dtype, root=root)

    if rank != root:
      C_nv_k = np.empty(shape, dtype=dtype)

    # guarantee C-contiguous so ravel() returns a view, not a copy
    C_nv_k = np.ascontiguousarray(C_nv_k)

    # chunked Bcast — each chunk ~80 MB
    flat = C_nv_k.ravel()
    chunk_size = 10_000_000
    for i in range(0, flat.size, chunk_size):
      end = min(i + chunk_size, flat.size)
      comm.Bcast(flat[i:end], root=root)

    return C_nv_k
 

  def get_E_ck(self):
    """
    Returns the Electronic single particle bands
    for conduction bands
    Mind:
      Python starts from 0 and so do we
    """
    method = self.config.get("bse", {}).get("method", {})
    bands_data = self.config.get("bse", {}).get(method, {}).get("bands", {})
    engine = bands_data["engine"]
    cb_start = bands_data["cb_idx"][0] + bands_data["cb_skip"]
    cb_end = bands_data["cb_idx"][1] + bands_data["cb_skip"]   
    if method == "tb":
      return self.get_E_tb("Eigval_k_tb.hdf5")[cb_start:cb_end]
    elif method == "dft":   
      if engine == "siesta":
        return E_siesta(bands_data["file"])[cb_start:cb_end]
      elif engine == "quantum_espresso":
         return E_qe(bands_data["file"])[cb_start:cb_end]
      else:
        if rank == root:
          print_f("Unknown DFT engine found in BSE config file")
        comm.Abort(1)
    else:
      if rank == root:
        print_f("Unknown method!")
      comm.Abort(1)


  def get_E_ckplusQ(self):
    """
    Omly relevant for k+Q for the TB method.
    Method DFT reshuffles to extract E_ckplusQ. 
    Mind:
      Python starts from 0 and so do we
    """
    method = self.config.get("bse", {}).get("method", {})
    bands_data = self.config.get("bse", {}).get(method, {}).get("bands", {})
    engine = bands_data["engine"]
    cb_start = bands_data["cb_idx"][0] + bands_data["cb_skip"]
    cb_end = bands_data["cb_idx"][1] + bands_data["cb_skip"]   
    if method == "tb":
      return self.get_E_tb("Eigval_kplusQ_tb.hdf5")[:,cb_start:cb_end,:]
    else:
      if rank == root:
        print_f("You must set method to 'tb' to do this!")
        print_f("Exiting...")
      comm.Abort(1)


  def get_E_vk(self):
    """
    Returns the Electronic single particle bands
    for valence bands
    Mind:
      Python starts from 0 and so do we
    """
    method = self.config.get("bse", {}).get("method", {})
    bands_data = self.config.get("bse", {}).get(method, {}).get("bands", {})
    engine = bands_data["engine"]
    vb_start = bands_data["vb_idx"][0] + bands_data["vb_skip"]
    vb_end = bands_data["vb_idx"][1] + bands_data["vb_skip"]   
    if method == "tb":
      data = self.get_E_tb("Eigval_k_tb.hdf5")[vb_start:vb_end]
      return self.get_E_tb("Eigval_k_tb.hdf5")[vb_start:vb_end]
    elif method == "dft":   
      if engine == "siesta":
        return E_siesta(bands_data["file"])[vb_start:vb_end]
      elif engine == "quantum_espresso":
         return E_qe(bands_data["file"])[vb_start:vb_end]
      else:
        if rank == root:
          print_f("Unknown DFT engine found in BSE config file")
        comm.Abort(1)
    else:
      if rank == root:
        print_f("Unknown method!")
      comm.Abort(1)


  def get_E_tb(self, filename):
    """
    Load the E_nk file (num_band x num_kpt dimension)
    Data is loaded on the root and distributed to all participating
    ranks. 
    """
    return self.load_and_distribute_hdf5_array(filename, dataset_key="Eigenvalues")


  def get_C_tb(self, filename):
    """
    Load the C_nm_k file (num_band x num_band x num_kpt);
    Note that this is called at root. No need to broadcast it.
    """
    with h5py.File(filename, "r") as eigvec_file:
      eigvec = eigvec_file["Eigenvectors"][:]
    return eigvec
    
    


#  def get_Delta_csk_unit(self):
#    """
#    Returns the SOC effects with respect to non-polar.
#    calculations for the unit cell. 
#    """
#    # Load the appropriate file
#    f1 = h5py.File("DELTA.hdf5", 'r')
#    Delta_nsk = np.array(f1['D_nsk_unit'])
#    f1.close() 
#
#    for i in range(self.dft.shape[0]):
#      for j in range(self.dft.shape[1]):
#        if isinstance(self.dft[i,j],str):
#          if "cb" in self.dft[i,j].casefold():
#            # SIESTA only
#            #if "siesta" in self.dft[i,0].casefold():
#            #  return \
#            #  Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
#            #  int(self.dft[i,3])+int(self.dft[i,4]),:,:]
#            return \
#              Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
#              int(self.dft[i,3])+int(self.dft[i,4]),:,:]
#
#
#  def get_Delta_vsk_unit(self):
#    """
#    Returns the SOC effects with respect to non-polar.
#    calculations for the unit cell. 
#    """ 
#    # Load the appropriate file
#    f1 = h5py.File("DELTA.hdf5", 'r')
#    Delta_nsk = np.array(f1['D_nsk_unit'])
#    f1.close() 
#
#    for i in range(self.dft.shape[0]):
#      for j in range(self.dft.shape[1]):
#        if isinstance(self.dft[i,j],str):
#          if "vb" in self.dft[i,j].casefold():
#            return \
#              Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
#              int(self.dft[i,3])+int(self.dft[i,4]),:,:]
#
#
#  def get_Delta_vsk_pmu(self):
#    """
#    Returns the SOC effects with respect to non-polar.
#    calculations. This is a poor man's unfolding version.
#    ONLY use if you know very well what you are doing!!!
#    ** Under approximation that the valence bands contributing
#       to the excitons come from K point and G-point
#       At the K-point, the splitting is about 445 meV.
#       At the G-point, the splitting is zero.
#       One needs to look at the band-structure and analyze\
#       the wave-functions further to confirm this hypotheis.
#       ****Twist-angle dependent****
#    # Ideally this could be done within the source code by calling 
#    # the following lines, however, as these numbers are twist-angle
#    # dependent and needs care to use, I will use a separate file
#    # for convencience. 
#    """
#    Delta_vsk = np.load(self.absorp[3])[0]
#    return Delta_vsk 
#
#
#  def get_Delta_csk_pmu(self):
#    """
#    Returns the SOC effects with respect to non-polar.
#    calculations. This is a poor man's unfolding version.
#    ONLY use if you know very well what you are doing!!!
#    ** Under approximation that the conduction bands contributing
#       to the excitons come from K point
#       At the conduction bands, the splitting is about 40 meV.
#    # Ideally this could be done within the source code by calling 
#    # the following lines, however, as these numbers are twist-angle
#    # dependent and needs care to use, I will use a separate file
#    # for convencience. 
#    """
#    Delta_csk = np.load(self.absorp[3])[1]
#    return Delta_csk 
#
#
  def phasefix(self, C_nm_k):
    """
    Fixes the phase of single-particle wave-functions
    by choosing the sum of basis-set coeffs of wfns 
    to be a real number. 
    Ref: Rohlfing and Louie, PRB 62 (2000).
    Args:
      C_nm_k: Linear Combination of Atomic or Atomic-like
              Orbital coefficients
    Returns:
      C_nm_k: Phase fixed coeeficients
    """
    if rank == 0:
      print_f("Fixing the phases of single-particle eigenvectors (Cnmk's)")
    if C_nm_k.ndim == 3:
      # for each Bloch wave-functions (m)
      for i in range(C_nm_k.shape[1]):
        # for each electronic k-point
        for j in range(C_nm_k.shape[2]):
          # Sum over Basis-set coefficients
          # Basis sets could come from TB/DFT
          s = np.sum(C_nm_k[:,i,j])
          phase = np.exp(-1j*np.angle(s))
          C_nm_k[:,i,j] = phase*C_nm_k[:,i,j]

          # Check if phase fixing worked
          if np.imag(np.sum(C_nm_k[:,i,j])) > 10**-8:
            if rank == root:
              print_f("Phase fixing didn't work properly")
              print_f("Exiting...")
            comm.Abort(1)
    elif C_nm_k.ndim == 4:
      for iQ in range(C_nm_k.shape[0]):
        # for each Bloch wave-functions (m)
        for i in range(C_nm_k.shape[2]):
          # for each electronic k-point
          for j in range(C_nm_k.shape[3]):
            # Sum over Basis-set coefficients
            # Basis sets could come from TB/DFT
            s = np.sum(C_nm_k[iQ,:,i,j])
            phase = np.exp(-1j*np.angle(s))
            C_nm_k[iQ,:,i,j] = phase*C_nm_k[iQ,:,i,j]

            # Check if phase fixing worked
            if np.imag(np.sum(C_nm_k[iQ,:,i,j])) > 10**-8:
              if rank == root:
                print_f("Phase fixing didn't work properly")
                print_f("Exiting...")
              comm.Abort(1)
    return C_nm_k



  def c_nc_k(self,C_nm_k,cb_min,cb_max):
    """
    Separate Conduction band manifold
    (num_wann x num_bloch x num_kpt)
    """
    return self.phasefix(C_nm_k)[:, cb_min:cb_max, :]  

  def c_nc_kplusQ(self,C_nm_k,cb_min,cb_max):
    """
    Separate Conduction band manifold
    (num_Q x num_wann x num_bloch x num_kpt)
    """
    return self.phasefix(C_nm_k)[:, :, cb_min:cb_max, :]  


  def c_nv_k(self,C_nm_k,vb_min,vb_max):
    """
    Separate Valence band manifold
    """
    return self.phasefix(C_nm_k)[:, vb_min:vb_max, :]


  def is_hermitian(self, M, tol=1e-8, n_samples=500):
    re = M.shape[0]
    if re > 10000:
      rng   = np.random.default_rng(seed=42)
      idx_i = rng.choice(re, n_samples, replace=False)
      idx_j = rng.choice(re, n_samples, replace=False)
      diff  = np.abs(M[idx_i, idx_j] - M[idx_j, idx_i].conj())
      scale = np.abs(M[idx_i, idx_j]) + 1.0
      rel   = diff / scale
      if not np.all(rel < tol):
        worst = np.argmax(rel)
        if rank == root:
          print_f(f"ERROR: not Hermitian (sampled {n_samples} pairs) — "
                  f"max relative: {rel.max():.2e}, max absolute: {diff.max():.2e}")
          print_f(f"  Worst pair: i={idx_i[worst]}, j={idx_j[worst]}, "
                  f"M[i,j]={M[idx_i[worst], idx_j[worst]]:.6e}, "
                  f"M[j,i]*={M[idx_j[worst], idx_i[worst]].conj():.6e}")
        comm.Abort(1)
    else:
      diff  = np.abs(M - M.conj().T)
      scale = np.abs(M) + 1.0
      rel   = diff / scale
      if not np.all(rel < tol):
        worst = np.unravel_index(np.argmax(rel), rel.shape)
        if rank == root:
          print_f(f"ERROR: not Hermitian (full {re}x{re}) — "
                  f"max relative: {rel.max():.2e}, max absolute: {diff.max():.2e}")
          print_f(f"  Worst element: i={worst[0]}, j={worst[1]}, "
                  f"M[i,j]={M[worst[0], worst[1]]:.6e}, "
                  f"M[j,i]*={M[worst[1], worst[0]].conj():.6e}")
        comm.Abort(1)


  def get_electron_density(self):
    """
    Extracts the Wannier90-derived electron densities for all conduction
    bands (as specified by cb_idx).

    Output:
      Saves (k, x, y, d) density data to HDF5 for each band.
      - k: k-point index
      - x, y: orbital position
      - d: charge density
    """
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    # Defaults
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"
    if rank == root:
      map_wf = self.wan2bse.map_WF(using_mlwf=using_mlwf,
                                   mode=wf_loc)
      pos_wf = map_wf[:, 3:6].astype(float)
      ind_wf = map_wf[:, 0:3].astype(int)

      num_kpts = self.C_nc_kplusQ.shape[3]
      num_orbs = ind_wf.shape[0]
      num_bands = self.C_nc_kplusQ.shape[2]

      with h5py.File("Electron_densities.hdf5", 'w') as f:
        for i_cb in range(num_bands):
          loc_r = np.zeros((num_kpts, num_orbs, 4), dtype=float)
          for k in range(num_kpts):
            for i in range(num_orbs):
              j_start, j_end = ind_wf[i, 1], ind_wf[i, 2]
              density = np.sum(np.abs(self.C_nc_kplusQ[0,j_start:j_end, i_cb, k])**2)
              loc_r[k, i] = [*pos_wf[i], density]

        f.create_dataset(f"cb_{i_cb}", data=loc_r)
        print_f(f"Stored conduction band {i_cb} in HDF5")
    return None


  def get_hole_density(self):
    """
    Extracts the Wannier90-derived hole densities for all valence
    bands (from the data in C_nv_k).

    Output:
      Saves (k, x, y, d) density data to HDF5 for each band.
      - k: k-point index
      - x, y: orbital position
      - d: charge density
    """
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    # Defaults
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"
      
    if rank == root:
      map_wf = self.wan2bse.map_WF(using_mlwf=using_mlwf,
                                   mode=wf_loc)
      pos_wf = map_wf[:, 3:6].astype(float)
      ind_wf = map_wf[:, 0:3].astype(int)

      num_kpts = self.C_nv_k.shape[2]
      num_orbs = ind_wf.shape[0]
      num_bands = self.C_nv_k.shape[1]

      with h5py.File("Hole_densities.hdf5", 'w') as f:
        for i_vb in range(num_bands):
          loc_r = np.zeros((num_kpts, num_orbs, 4), dtype=float)
          for k in range(num_kpts):
            for i in range(num_orbs):
              j_start, j_end = ind_wf[i, 1], ind_wf[i, 2]
              density = np.sum(np.abs(self.C_nv_k[j_start:j_end, i_vb, k])**2)
              loc_r[k, i] = [*pos_wf[i], density]

          f.create_dataset(f"vb_{i_vb}", data=loc_r)
          print_f(f"Stored valence band {i_vb} in HDF5")
    return None


  def get_exciton_wfn_at_rh(self, rh, num_S=None, S_indices=None):
    """
    Computes the exciton wave-function in real-space
    for a fixed position of the hole, rh
    Parameters:
      rh: (x,y,z)- a vector for position in Angstrom
      num_S: number of exciton indices to use, states 0..num_S-1
             (ignored if S_indices is given)
      S_indices: explicit list/array of exciton state indices to use,
                 e.g. [3, 47, 112]. Overrides num_S if provided.
    """
    if S_indices is None:
      if num_S is None:
        if rank == root:
          print_f("Must specify either num_S or S_indices")
        comm.Abort(1)
      S_indices = list(range(num_S))
    else:
      S_indices = list(S_indices)
    num_S = len(S_indices)

    method = self.config.get("bse", {}).get("method", {})
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"
    io_files = self.set_exciton_io_files()
    Exciton_wfn_rh_file = io_files["wfn_rh_file"]
    n3 = self.get_hole_loc(rh)
    if n3 is None:
      print_f("Use a hole position (rh) that falls within 0.2 Ang")
      comm.Abort(1)
    else:
      if rank == root:
        print_f()
        print_f(f"Computing {num_S} exciton wave function by fixing hole @{rh}")
        print_f(f"Using exciton indices: {S_indices}")
        print_f(f"Wannier Functions from {n3[1]} to {n3[2]} will be used")
        print_f(f"By default, I will use Gamma point for visualization!")

    map_wf = self.wan2bse.map_WF(using_mlwf=using_mlwf, mode=wf_loc)
    pos_wf = map_wf[:,3:6].astype(float)
    ind_wf = map_wf[:,0:3].astype(int)

    Gamma_idx = self.select_Gamma()
    eigvals = self.load_and_distribute_eigensol(io_files["eigenval_file"], Gamma_idx)
    eigvecs = self.load_and_distribute_eigensol(io_files["eigenvec_file"], Gamma_idx)

    # Slist now indexes POSITIONS within S_indices, not raw exciton indices
    Slist = self.distribute_full(num_S, rank, size)
    local_size = Slist[1] - Slist[0]
    local_S_indices = S_indices[Slist[0]:Slist[1]]
    local_exciton_wfn = np.zeros((local_size, ind_wf.shape[0], self.Rvec.shape[0]),
                                  dtype=np.complex128)

    if method == "dft":
      A_Scvk = np.zeros((num_S, self.C_nc_k.shape[1],
                         self.C_nv_k.shape[1], self.C_nc_k.shape[2]),
                         dtype = np.complex128)
      for i, s_idx in enumerate(S_indices):
        A_Scvk[i] = eigvecs[:, s_idx].reshape((self.C_nc_k.shape[1],
                        self.C_nv_k.shape[1],
                        self.C_nc_k.shape[2]))
      for i_local, s_idx in enumerate(local_S_indices):
        i_global = Slist[0] + i_local
        local_exciton_wfn[i_local, :, :] = opt_exciton_r(A_Scvk[i_global],
                                      self.Rvec, self.kvec, self.C_nc_k,
                                      self.C_nv_k, ind_wf, pos_wf,
                                      n3[0:3].astype(int), rh)

    elif method == "tb":
      A_Scvk = np.zeros((num_S, self.C_nc_kplusQ.shape[2],
                         self.C_nv_k.shape[1], self.C_nc_kplusQ.shape[3]),
                         dtype = np.complex128)
      for i, s_idx in enumerate(S_indices):
        A_Scvk[i] = eigvecs[:, s_idx].reshape((self.C_nc_kplusQ.shape[2],
                        self.C_nv_k.shape[1],
                        self.C_nc_kplusQ.shape[3]))
      for i_local, s_idx in enumerate(local_S_indices):
        i_global = Slist[0] + i_local
        local_exciton_wfn[i_local, :, :] = opt_exciton_r(A_Scvk[i_global],
                                      self.Rvec, self.kvec,
                                      self.C_nc_kplusQ[Gamma_idx,:,:,:],
                                      self.C_nv_k, ind_wf, pos_wf,
                                      n3[0:3].astype(int), rh)

    local_indices = np.arange(Slist[0], Slist[1], dtype=np.int32)
    send_data = (local_indices, local_exciton_wfn)
    all_data = comm.gather(send_data, root=root)
    if rank == root:
      try:
        print_f(f"Writing {Exciton_wfn_rh_file} for exciton wfns.")
        with h5py.File(Exciton_wfn_rh_file, "w") as f:
          dset = f.create_dataset("exciton_wfn",
                                    shape=(num_S, ind_wf.shape[0], self.Rvec.shape[0]),
                                    dtype=np.complex128)
          for indices, values in all_data:
            dset[indices, :, :] = values
      except Exception as e:
        print_f(f"Error writing HDF5 file '{Exciton_wfn_rh_file}': {e}")
        comm.Abort(1)
        
      # Save Rvec, pos_wf, and S_indices for reference
      np.savetxt("Rvec.txt", self.Rvec, fmt="%.8f")
      np.savetxt("pos_wf.txt", map_wf, fmt="%.8f")
      np.savetxt("S_indices.txt", np.array(S_indices), fmt="%d")
      

  def get_hole_loc(self, rh):
    """
    For a given hole-location, find the Wannier functions
    that corresponds to n3. See Eqn.7 of our paper.
    https://www.nature.com/articles/s41699-025-00538-4
    """
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    # Defaults
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"
    map_wf = self.wan2bse.map_WF(using_mlwf=using_mlwf,
                                 mode=wf_loc)
    rh = np.asarray(rh)
    if rh.shape != (3,):
      raise ValueError("Input 'rh' must be a 3-element vector.")

    for i in range(map_wf.shape[0]):
      pos = map_wf[i, 3:6]
      if np.linalg.norm(pos - rh) < 0.5:
        return map_wf[i, :]
    return None  


  def get_electron_for_exciton(self, S):
    """
    Computes the electron wave-function for a given exciton index S.
    Elaborate computation of exciton wavefunctions.
    """
    wf_loc = self.config.get("bse", {}).get("wannier_io", {}
                                            ).get("wf_location")
    using_mlwf = self.config.get("bse", {}).get("wannier_io", {}
                                             ).get("using_mlwf", False)
    # Defaults
    if not isinstance(wf_loc, str) or wf_loc.strip() == "":
      wf_loc = "atom_centered"
    if rank ==  root:
      print_f(f"Computing electron density for exciton index S = {S}")
      print_f(f"(Summing over all hole contributions)")
    method = self.config.get("bse", {}).get("method", {})
    io_files = self.set_exciton_io_files()

    # This is going to be used for n1 and n3
    map_wf = self.wan2bse.map_WF(using_mlwf=using_mlwf,
                                 mode=wf_loc)
    pos_wf = map_wf[:,3:6].astype(float)
    ind_wf = map_wf[:,0:3].astype(int)

    # Load data at root and distribute
    Gamma_idx = self.select_Gamma()
    eigvals = self.load_and_distribute_eigensol(io_files["eigenval_file"], Gamma_idx)
    eigvecs = self.load_and_distribute_eigensol(io_files["eigenvec_file"], Gamma_idx)

    num_orb = ind_wf.shape[0]
    nlist = self.distribute_full(num_orb, rank, size)
    local_size = nlist[1] - nlist[0]
    local_electron_wfn = np.zeros((local_size, self.Rvec.shape[0]), 
                                  dtype=np.complex128)
    if method == "tb":
      A_Scvk = eigvecs[:,S].reshape((self.C_nc_kplusQ.shape[2],
                        self.C_nv_k.shape[1],
                        self.C_nc_kplusQ.shape[3]))
      counter = 0 
      for i_local, i_global in enumerate(range(nlist[0], nlist[1])):
        local_exciton_wfn[i_local, :] = opt_electron_for_exciton(A_Scvk,
                              self.Rvec, self.kvec, 
                              self.C_nc_kplusQ[Gamma_idx,:,:,:],
                              self.C_nv_k, ind_wf, pos_wf, i_global)
        if rank == root:
          counter += 1
          print_f(f"Root computed {counter} out of {local_size}")

      # Prepare data to send to root
      local_indices = np.arange(nlist[0], nlist[1], dtype=np.int32)
      send_data = (local_indices, local_exciton_wfn)
      all_data = comm.gather(send_data, root=root)

      # Root writes to HDF5
      electron_for_exciton = f"Electron_for_S_{S}.hdf5"
      if rank == root:
        try:
          print_f(f"Writing {electron_for_exciton}.")
          with h5py.File(electron_for_exciton, "w") as f:
            dset = f.create_dataset("electron_density",
                                      shape=(ind_wf.shape[0], self.Rvec.shape[0]),
                                      dtype=np.complex128)
            for indices, values in all_data:
              dset[indices, :] = values
        except Exception as e:
          print_f(f"Error writing HDF5 file '{electron_for_exciton}': {e}")
          comm.Abort(1)

    elif method == "dft":
      print_f("Exiting...")
      comm.Abort(1)


  def estimate_memory_from_config(self):
    """
    Estimate peak memory per rank for each main function.
    """
    method        = self.config.get("bse", {}).get("method", {})
    cnmk_inp      = self.config["bse"][method]["cnmk"]
    nc            = cnmk_inp["cb_idx"][1] - cnmk_inp["cb_idx"][0]
    nv            = cnmk_inp["vb_idx"][1] - cnmk_inp["vb_idx"][0]
    kgrid         = self.config["bse"][method]["kgrid"]
    nk            = kgrid[0] * kgrid[1] * kgrid[2]
    nR            = nk
    re            = nc * nv * nk
    diagon_method = self.config.get("diagonalize", {}).get("library", "lapack")

    wannier  = self.config["bse"]["wannier_io"]
    nW       = self._read_nwann_from_win(wannier["win_file"])

    add_direct        = self.config.get("eh_interaction", {}).get("include", {}).get("direct", False)
    add_exchange      = self.config.get("eh_interaction", {}).get("include", {}).get("exchange", False)
    screened_exchange = self.config.get("eh_interaction", {}).get("include", {}).get("screen_exchange", False)
    mem_limit_gb      = self.config.get("memory", {}).get("limit_gb", 256.0)

    ranks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", size))
    n_threads      = int(os.environ.get("OMP_NUM_THREADS", 1))

    c128 = 16; f64 = 8
    gb   = lambda b: b / 1024**3
    fmt  = lambda b: f"{gb(b):>8.2f} GB" if b >= 1024**3 else f"{b/1024**2:>8.1f} MB"

    funcs = {
        "_setup_tb": {
          "H_k (tmp)"      : nW * nW * nk * c128,
          "grad_H_k (tmp)" : nW * nW * nk * 3 * c128,
          "V_r_keld (tmp)" : nR * nW * nW * f64 if add_direct else 0,
          "V_r_coul (tmp)" : nR * nW * nW * f64 if (add_exchange and not screened_exchange) else 0,
        },
        "write_exciton_H": {
          "C_nv_k"      : nW * nv * nk * c128,
          "C_nc_kplusQ" : nW * nc * nk * c128,
          "V_r_keld"    : nR * nW * nW * f64 if add_direct else 0,
          "V_r_coul"    : nR * nW * nW * f64 if (add_exchange and not screened_exchange) else 0,
        },
        "diagon_BSE": {
          "H_eh"  : re * re * c128,
          "eigvec": re * re * c128,
        },
        "optical_conductivity": {
          "C_nv_k"      : nW * nv * nk * c128,
          "C_nc_kplusQ" : nW * nc * nk * c128,
          "H_k"         : nW * nW * nk * c128,
          "grad_H_k"    : nW * nW * nk * 3 * c128,
        },
        "get_exciton_wfn_at_rh": {
          "C_nv_k"      : nW * nv * nk * c128,
          "C_nc_kplusQ" : nW * nc * nk * c128,
        },
      }

    notes = {
      "_setup_tb"            : "root only — temporary during TB setup",
      "write_exciton_H"      : "all ranks",
      "diagon_BSE"           : "root only" if diagon_method == "lapack" else "all ranks",
      "optical_conductivity" : "all ranks",
      "get_exciton_wfn_at_rh": "all ranks",
    }

    if rank == root:
      print_f(f"  Peak memory per rank")
      print_f(f"  nW={nW}, nk={nk}, nv={nv}, nc={nc}, re={re}")
      print_f(f"  ranks={size} ({ranks_per_node}/node) | threads={n_threads} | diagon={diagon_method}")
      print_f()
      for fname, mem in funcs.items():
        total = sum(mem.values())
        flag  = " ← exceeds limit" if gb(total) > mem_limit_gb else ""
        print_f(f"  {fname} ({notes[fname]})")
        for name, nb in mem.items():
          if nb > 0:
            print_f(f"    {name:<18} {fmt(nb)}")
        print_f(f"    Peak per rank    {fmt(total)}{flag}")
        print_f()
      print_f(f"  Limit: {mem_limit_gb:.0f} GB/rank | node total = peak × {ranks_per_node} ranks")
      print_f()


  def _read_nwann_from_win(self, win_file):
    """
    Read num_wann from Wannier90 win file — lightweight text read only.
    """
    with open(win_file, "r") as f:
      for line in f:
        l = line.strip().lower()
        if l.startswith("num_wann"):
          return int(l.split("=")[-1].strip())
    if rank == root:
      print_f("WARNING: num_wann not found in win file — skipping memory check.")
    return 0

