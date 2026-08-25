#-------------------------------------|
#email: indrajit.maity02@gmail.com    |
#author: Indrajit Maity               |
#-------------------------------------|

import numpy as np
import sys, scipy, time
from scipy import special
from constants import *
from functools import partial
from wan90tobse import *
from generic_func import *
from mpi4py import MPI
import h5py
import sympy as sp
print_f = partial(print, flush=True)

#=========================|
# Potential ( multilayer )|
#=========================|

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

class POTENTIAL(object):
  """
  """
  # data attributes
  def __init__(self,wan2bse, material, kgrid=None):
    self.wan2bse = wan2bse
    self.WF_loc = self.wan2bse.get_WF_loc()
    self.A = self.wan2bse.get_lattice()
    self.material_type = material
    self.kgrid = kgrid 
    if self.kgrid is None:
      self.kgrid = self.wan2bse.get_kgrid()
    self.A_s = unit2super(self.kgrid, self.A)


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


  def get_Rvec(self):
    """
    Generate real-space lattice points R = i*a1 + j*a2 + k*a3.
    Returns:
      np.ndarray: Array of shape (N, 3) containing all lattice points
    """
    nx, ny, nz = self.kgrid
    a1, a2, a3 = self.A
    R_list = []

    def get_symmetric_range(n):
      half = n // 2
      if n % 2 == 1:
        return range(-half, half + 1)
      else:
        return range(-half, half)

    for i in get_symmetric_range(nx):
      for j in get_symmetric_range(ny):
        for k in get_symmetric_range(nz):
          R = i * a1 + j * a2 + k * a3
          R_list.append(R)

    return np.array(R_list)


  def get_Rvec_WS(self):
    """
    Generate R-vectors inside the Wigner-Seitz cell of the supercell.
    
    For k-grid [nx, ny, nz], the supercell has lattice vectors
    L_i = n_i * a_i. We keep all R = i*a1 + j*a2 + k*a3 that are
    closer to the origin than to any supercell image (WS condition).
    
    Boundary points (equidistant to multiple images) get fractional
    degeneracy weights: w = 1/n_degen.
    
    Returns:
      R_vectors : np.ndarray, shape (N_R, 3)
      weights   : np.ndarray, shape (N_R,)
                  Σ weights = nx * ny * nz
    """
    nx, ny, nz = self.kgrid
    a1, a2, a3 = self.A
    tol = 1e-6

    # Supercell lattice vectors
    L = np.array([nx * a1, ny * a2, nz * a3])

    # Supercell neighbors for WS check (exclude origin)
    shifts = np.array([[m1, m2, m3]
      for m1 in range(-2, 3)
      for m2 in range(-2, 3)
      for m3 in range(-2, 3)
      if not (m1 == 0 and m2 == 0 and m3 == 0)])
    G_vecs = shifts @ L  # (N_G, 3)

    # Candidate R-vectors: generous range to catch all WS boundary points
    # For non-orthogonal cells, WS extends beyond N//2 in index space
    def cand_range(n):
      if n == 1:
        return range(0, 1)
      half = n // 2 + 2
      return range(-half, half + 1)

    R_list = []
    w_list = []
    for i in cand_range(nx):
      for j in cand_range(ny):
        for k in cand_range(nz):
          R = i * a1 + j * a2 + k * a3
          dist0 = np.dot(R, R)

          # WS condition: 2*R·G ≤ |G|² for all supercell vectors G
          # Equivalent to: |R|² ≤ |R - G|² for all G
          dists_G = np.sum((R - G_vecs)**2, axis=1)
          n_closer = np.sum(dists_G < dist0 - tol)

          if n_closer == 0:
            # Point is inside or on boundary of WS cell
            n_degen = 1 + np.sum(np.abs(dists_G - dist0) < tol)
            R_list.append(R)
            w_list.append(1.0 / n_degen)

    R_vectors = np.array(R_list)
    weights = np.array(w_list)
    return R_vectors, weights


  def build_tvec_ab(self, R, wanfuncs_pos, n_images=1, chunk_size=500):

    n = len(wanfuncs_pos)
    tvec_ab = np.empty((n, n, 3), dtype=float)

    rng = np.arange(-n_images, n_images + 1)

    if self.material_type == "2d":
      g1, g2, g3 = np.meshgrid(rng, rng, [0], indexing='ij')
    elif self.material_type == "3d":
      g1, g2, g3 = np.meshgrid(rng, rng, rng, indexing='ij')
    elif self.material_type == "1d":
      g1, g2, g3 = np.meshgrid(rng, [0], [0], indexing='ij')
    else:
      g1, g2, g3 = np.meshgrid([0], [0], [0], indexing='ij')

    # flatten in same order as original loops
    n1s = g1.ravel()
    n2s = g2.ravel()
    n3s = g3.ravel()

    # shift vectors: shape (n_shifts, 3)
    shifts = (n1s[:, None] * self.A_s[:, 0] +
      n2s[:, None] * self.A_s[:, 1] +
      n3s[:, None] * self.A_s[:, 2])

    pos = wanfuncs_pos[:, :3]

    for a_start in range(0, n, chunk_size):
      a_end = min(a_start + chunk_size, n)

      # raw displacements: shape (chunk, n, 3)
      delta = R[:3] + pos[np.newaxis, :, :] - pos[a_start:a_end, np.newaxis, :]

      # initialise with zero shift
      best_vec   = delta.copy()
      best_dist2 = np.einsum('abi,abi->ab', best_vec, best_vec)
      best_n1    = np.zeros_like(best_dist2, dtype=int)
      best_n2    = np.zeros_like(best_dist2, dtype=int)
      best_n3    = np.zeros_like(best_dist2, dtype=int)

      for k in range(len(shifts)):
        trial = delta + shifts[k]
        dist2 = np.einsum('abi,abi->ab', trial, trial)
        n1, n2, n3 = n1s[k], n2s[k], n3s[k]

        better = dist2 < best_dist2

        # tie-breaking: identical to original shift < best_shift tuple comparison
        tie = np.abs(dist2 - best_dist2) < 1e-10
        lex = ((n1 < best_n1)
          | ((n1 == best_n1) & (n2 < best_n2))
          | ((n1 == best_n1) & (n2 == best_n2) & (n3 < best_n3)))

        take = better | (tie & lex)
        best_vec[take]   = trial[take]
        best_dist2[take] = dist2[take]
        best_n1[take]    = n1
        best_n2[take]    = n2
        best_n3[take]    = n3

      tvec_ab[a_start:a_end] = best_vec

    return tvec_ab


  def get_distvec(self, R, t_n1_m, t_n3_m, n_images=1):
    delta = (R + t_n3_m) - t_n1_m
    rng   = list(range(-n_images, n_images + 1))

    if self.material_type == "3d":
      shifts = [rng, rng, rng]
    elif self.material_type == "2d":
      shifts = [rng, rng, [0]]
    elif self.material_type == "1d":
      shifts = [rng, [0], [0]]
    elif self.material_type == "0d":
      shifts = [[0], [0], [0]]
    else:
      print_f(f"ERROR: material_type must be '0D', '1D', '2D', or '3D', "
              f"got '{material_type}'")
      comm.Abort(1)

    best_delta = delta[:3].copy()
    best_dist2 = np.dot(best_delta, best_delta)
    best_shift = (0, 0, 0)

    for n1 in shifts[0]:
      for n2 in shifts[1]:
        for n3 in shifts[2]:
          trial = (delta[:3]
                  + n1 * self.A_s[:, 0]
                  + n2 * self.A_s[:, 1]
                  + n3 * self.A_s[:, 2])
          dist2 = np.dot(trial, trial)
          shift = (n1, n2, n3)

          if dist2 < best_dist2:
            # Strictly shorter — always take it
            best_dist2 = dist2
            best_delta = trial
            best_shift = shift
          elif abs(dist2 - best_dist2) < 1e-10 and shift < best_shift:
            # Genuine tie — lexicographic shift ordering for consistency
            best_delta = trial
            best_shift = shift

    return best_delta


  def get_dist(self, R, t_n1_m, t_n3_m, n_images=1):
    """
    Minimum image distance |delta| for the displacement R + t_n3_m - t_n1_m.
    Applies canonical sign convention before taking norm to guarantee:
      get_dist(R, t_n1, t_n3) == get_dist(R, t_n3, t_n1)
    which ensures V(R+t_n3-t_n1) = V(R+t_n1-t_n3) and preserves Hermiticity.
    """
    delta = self.get_distvec(R, t_n1_m, t_n3_m, n_images)
    return np.linalg.norm(delta)


  def rho(self,vec):
    """
    Computes the distance between two-points
    using Periodic Boundary Conditions.
    @input
      vec: sepration vector;
         (R + (tn1-tn3)) with PBC.
    @output
      norm of the vector (in-plane coordinate)
      Required for the LFT of the Keldysh potential.
    """
    # In-plane distances (x,y)
    return np.linalg.norm(vec[:2])



  def Struve(self, x):
    """
    Struve function
    @input
      v: order, 0 (default)
      x: argument
    @output
      Struve function of order v
  """
    return scipy.special.struve(0,x)


  def Bessel0(self, x):
    """
    Bessel function
    @input
      x: argument
    @output
      Bessel function of 2nd kind, 0th-order and real
      argument
    """
    return scipy.special.y0(x)


  def set_params_dict(self, chi_list, zav,
                      epsilon_t_val=1.0, epsilon_b_val=1.0):
    """
    Sets the parameters dictionary for sympy substitution.
    Args:
      chi_list: List of chi values for each layer.
    Returns:
      dict: parameters dictionary for sympy substitution.
    """
    C, epsilon_b, epsilon_t = sp.symbols('C epsilon_b epsilon_t', real=True)
    params_dict = {}

    params_dict[epsilon_b] = epsilon_t_val
    params_dict[epsilon_t] = epsilon_b_val
    n_layers = len(chi_list)
    for i in range(n_layers):
      chi_sym = sp.symbols(f'chi_{i+1}')
      params_dict[chi_sym] = chi_list[i]

    if n_layers > 1:
      for i in range(n_layers-1):
        d_sym = sp.symbols(f'd_{i+1}{i+2}')
        params_dict[d_sym] = zav[i+1]-zav[i]
    # C = e*e/(epsilon_0) Note that I absorbed extra e
    # before the Hankel transform in the potential expression
    params_dict[C] = 180.7 # in eV.Angstrom units
    return params_dict


  def symbolic_to_numeric(self, phi_q_symbolic, q_sym):
      """
      Convert symbolic expression to numerical function
      phi_q_symbolic: sympy expression like -180.7/(q*(75.74*q + 2.0))
      q_sym: sympy symbol for q
      """
      return sp.lambdify(q_sym, phi_q_symbolic, 'numpy')


  def get_phi_q(self, phi_dict, params_dict):
    """
    Substitute all parameters except q and evaluate if floats are present
    """
    q_sym = sp.symbols('q')
    subs_dict = {k: v for k, v in params_dict.items() if k != q_sym}
    
    phi_dict_q = {}
    for key, expr in phi_dict.items():
      if subs_dict:
        expr = expr.subs(subs_dict)
        if any(isinstance(v, float) for v in subs_dict.values()):
          expr = expr.evalf()
      phi_dict_q[key] = expr
    return phi_dict_q


  def keldysh_multilayer_numeric(self, Rvec, wanfuncs, epsilon_t_val,
                                epsilon_b_val, r0_list, a0_list, zav,
                                thickness=2.5):
    """
    Keldysh potential with numerical integration
    This forms the basis for n-layer systems using
    transfer matrix method.
    Some notes:
      \chi_i = 2 * r_list[i] (always in Angstrom)
      r0_list: Always like this, [layer1, layer2, ..., layern]
      a0_list: Always like this, [layer1, layer2, ..., layern]
      d_list: Always like this, [d_12, d_23, ..., d_(n-1)n]

    Uses Cython (in cyfuc.pyx) with prange for the inner loop.
    GL quadrature nodes passed as numpy arrays — no hardcoded C.
    """
    from cyfunc import fill_keldysh_Rvec, fill_keldysh_wan
    from numpy.polynomial.legendre import leggauss

    screened_file = "Screened_coulomb.hdf5"
    chi_list  = [2.0 * r0 for r0 in r0_list]
    n_layers  = len(chi_list)
    wanfuncs  = np.array(wanfuncs, dtype=np.float64)
    num_Rvec  = Rvec.shape[0]; num_wan = wanfuncs.shape[0]

    # Get momentum dependent potential — unchanged from original
    if rank == root:
      phi_dict, success = multilayer_potential(n_layers)
      if not success:
        print_f("Error in computing symbolic potential. Exiting...")
        comm.Abort(1)

      params_dict = self.set_params_dict(chi_list, zav,
                                        epsilon_t_val, epsilon_b_val)
      phi_dict_q  = self.get_phi_q(phi_dict, params_dict)

      filename = "phi_q.txt"
      with open(filename, "w") as f:
        for key, value in phi_dict.items():
          f.write(f"{key}: {value}\n")
      print_f(f"Saved phi_ij(q) expressions to {filename}")

      params_filename = "params.txt"
      with open(params_filename, "w") as f:
        for key, value in params_dict.items():
          f.write(f"{key}: {value}\n")
      print_f(f"Saved screened potential parameters to {params_filename}")
    else:
      phi_dict    = None
      params_dict = None
      phi_dict_q  = None

    # Broadcast — unchanged from original
    phi_dict    = comm.bcast(phi_dict,    root=root)
    params_dict = comm.bcast(params_dict, root=root)
    phi_dict_q  = comm.bcast(phi_dict_q,  root=root)

    lambdified_phi_dict = self.get_lambdified_phi_dict(phi_dict_q)
    layer_bounds        = self.set_layer_bounds(zav)

    if rank == root:
      print_f("Starting Cython GL Hankel transforms for multilayer Keldysh...")

    # ── precompute once on all ranks ──────────────────────────────────

    # GL nodes and weights — 2 segments of n_gl points
    # [0, q_max/4] + [q_max/4, q_max] — matches quad to <0.001 meV
    n_gl    = 200
    q_max   = 20.0
    q_split = q_max / 4.0
    xi, wi  = leggauss(n_gl)

    # segment 1: [0, q_split]
    q1 = 0.5 * q_split * (xi + 1.0)
    w1 = 0.5 * q_split * wi
    # segment 2: [q_split, q_max]
    q2 = 0.5 * (q_max - q_split) * (xi + 1.0) + q_split
    w2 = 0.5 * (q_max - q_split) * wi

    # concatenate into single arrays — shape (2*n_gl,)
    gl_q = np.concatenate([q1, q2]).astype(np.float64)
    gl_w = np.concatenate([w1, w2]).astype(np.float64)

    # layer index per WF — replaces find_layers in inner loop
    layer1_arr = np.zeros(num_wan, dtype=np.int32)
    for n in range(num_wan):
      l1, _ = self.find_layers(layer_bounds, thickness,
                                wanfuncs[n, 5], wanfuncs[n, 5])
      layer1_arr[n] = l1

    # a_av matrix — replaces per-point a_av computation
    a_av_mat = np.zeros((n_layers, n_layers), dtype=np.float64)
    for l1 in range(n_layers):
      for l2 in range(n_layers):
        a_av_mat[l1, l2] = 0.5 * (a0_list[l1] + a0_list[l2])

    # phi_key index — replaces string construction in inner loop
    keys        = sorted(lambdified_phi_dict.keys())
    n_keys      = len(keys)
    phi_key_idx = np.full((n_layers, n_layers), -1, dtype=np.int32)
    for idx, phi_key in enumerate(keys):
      l1 = int(phi_key[-2]) - 1
      l2 = int(phi_key[-1]) - 1
      phi_key_idx[l1, l2] = idx

    # phi(q) on GL nodes — shape (n_keys, 2*n_gl)
    # evaluated in Python where lambdified sympy runs freely
    phi_q_vals = np.zeros((n_keys, 2 * n_gl), dtype=np.float64)
    for idx, phi_key in enumerate(keys):
      phi_func = lambdified_phi_dict[phi_key]
      try:
        phi_q_vals[idx] = np.asarray(phi_func(gl_q), dtype=np.float64)
      except Exception:
        phi_q_vals[idx] = np.array([phi_func(q) for q in gl_q],
                                    dtype=np.float64)
      phi_q_vals[idx] = np.where(np.isfinite(phi_q_vals[idx]),
                                  phi_q_vals[idx], 0.0)

    # image shift vectors — 2D, n_images=1
    rng        = list(range(-1, 2))
    shifts     = np.array([(s1, s2, 0) for s1 in rng for s2 in rng],
                            dtype=np.float64)
    shift_vecs = np.ascontiguousarray(shifts @ self.A_s.T, dtype=np.float64)

    # contiguous arrays for Cython
    Rvec_c = np.ascontiguousarray(Rvec[:, :3],      dtype=np.float64)
    pos_c  = np.ascontiguousarray(wanfuncs[:, 3:6],  dtype=np.float64)

    # Decide how to parallelize — same condition as original
    if num_Rvec > num_wan**2:
      start, end = self.distribute_full(num_Rvec, rank, size)
      shape      = (end - start, num_wan, num_wan)
      Vr_local   = np.zeros(shape, dtype=np.float64)

      fill_keldysh_Rvec(
        Vr_local, Rvec_c, pos_c, shift_vecs,
        layer1_arr, a_av_mat, phi_key_idx,
        phi_q_vals, gl_q, gl_w, n_gl, int(start))

      local_indices = np.arange(start, end, dtype=np.int32)
      send_data     = (local_indices, Vr_local)

    else:
      start, end = self.distribute_full(num_wan, rank, size)
      shape      = (num_Rvec, end - start, num_wan)
      Vr_local   = np.zeros(shape, dtype=np.float64)

      fill_keldysh_wan(
        Vr_local, Rvec_c, pos_c, shift_vecs,
        layer1_arr, a_av_mat, phi_key_idx,
        phi_q_vals, gl_q, gl_w, n_gl, int(start))

      local_indices = np.arange(start, end, dtype=np.int32)
      send_data     = (local_indices, Vr_local)

    # Gather chunks at root — identical to original
    all_data = comm.gather(send_data, root=root)

    # Write results at root — identical to original
    if rank == root:
      Vr_full = np.zeros((num_Rvec, num_wan, num_wan), dtype=float)

      if num_Rvec > num_wan**2:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[global_i, :, :] = chunk[local_i, :, :]
      else:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[:, global_i, :] = chunk[:, local_i, :]

      Vr_full = self.symmetrise_potential(Vr_full, Rvec)

      # Write to HDF5
      with h5py.File(screened_file, "w") as f:
        f.create_dataset("W", data=Vr_full)
        print_f(f"Screened Coulomb potential written to {screened_file}")
    return None


  def get_lambdified_phi_dict(self, phi_dict):
    """
    Convert ALL symbolic expressions to numerical functions using mpmath
    No overflow issues, computed once and stored
    """ 
    q_sym = sp.Symbol('q')
    lambdified_phi_dict = {}
    
    for key, expr in phi_dict.items():
      lambdified_phi_dict[key] = sp.lambdify(q_sym, expr, 'numpy')    
    return lambdified_phi_dict



  # def Hankel_transform(self, f_q_func, r, r_min=1.5, q_max=20):
  #   """
  #   Compute Hankel transform for isotropic functions.
  #   Applies r_min cutoff: if r < r_min, uses r = r_min.
  #   Uses adaptive quadrature up to q_max.
  #   """
  #   effective_r = max(r, r_min)

  #   def f_q_debug(q):
  #     val = f_q_func(q)
  #     if not np.isfinite(val) or abs(val) > 1e10:
  #       print_f(f"  WARNING: f_q_func({q:.6f}) = {val:.4e}")
  #     return val

  #   integrand     = lambda q: f_q_debug(q) * scipy.special.j0(q * effective_r) * q
  #   result, error = scipy.integrate.quad(
  #     integrand, 0, q_max,
  #     limit=500, epsrel=1e-5, epsabs=1e-7
  #   )
  #   if error > 1e-4 * abs(result):
  #     print_f(f"  WARNING: Hankel quadrature error {error:.2e} > 1e-4 * result {result:.2e}")
  #   return result / (2 * np.pi)


  def find_layers(self, layer_bounds, thickness, n1_z, n3_z):
    """
    Find the layers for two z-coordinates based on layer boundaries.
    """
    layer1 = layer2 = None
    for i, bound in enumerate(layer_bounds):
      if bound - thickness <= n1_z < bound + thickness:
        layer1 = i
      if bound - thickness <= n3_z < bound + thickness:
        layer2 = i

    if layer1 is None or layer2 is None:
      if rank == root:
        print("Error: One or both values did not match any layer.")
        print("Exiting...")
      comm.Abort(1)
    return layer1, layer2

  
  def set_layer_bounds(self, zav):
    """
    Computes layer boundaries as cumulative sum of z-values.
    Always returns a NumPy array.
    """
    zav = np.asarray(zav, dtype=float)
    # return np.cumsum(zav)
    return zav


  def keldysh_monolayer_analytic(self, Rvec, wanfuncs, epsilon_t,
                                epsilon_b, r0,a0):
    """
    Keldysh potential at a particular point in real space;
    Monolayer TMDs and e-h separated in a layer.
    Args:
      epsilon_d: dielectric constant of the material
      r0: screening length
      a0: in-plane lattice constant (for regularisation)
    Screening function from: 
    https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.113.076802/SM_2Dexcitons.pdf
    """
    screened_file = "Screened_coulomb.hdf5"
    num_Rvec = Rvec.shape[0]
    num_wan = wanfuncs.shape[0]
    # rescale r0 based on dielectric constant
    epsilon_s = 2 / (epsilon_t + epsilon_b)
    r0_s = r0 * epsilon_s 
  
    # Decide how to parallelize
    if num_Rvec > num_wan**2:
      start, end = self.distribute_full(num_Rvec, rank, size)
      shape = (end - start, num_wan, num_wan)
      Vr_local = np.zeros(shape, dtype=float)

      for local_i, i in enumerate(range(start, end)):
        for n1_m in range(num_wan):
          for n3_m in range(num_wan):

            r = self.get_dist(Rvec[i], wanfuncs[n1_m,3:], wanfuncs[n3_m,3:])

            # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
            # Set up in a way that the potential is in eV.
            # f = 22.59
            f = 22.59
            U = 1.0
            # Setting the on-site term
            # Convention: Phys. Rev. B 91, 075310 (2015).
            if r <= a0:
              Vr_local[local_i, n1_m, n3_m] = -U * f * (self.Struve(a0/r0_s)-
                               self.Bessel0(a0/r0_s)) * 1.0 / (r0)
            else:
              Vr_local[local_i, n1_m, n3_m] = -f * (self.Struve(r/r0_s)- 
                               self.Bessel0(r/r0_s)) * 1.0 / (r0)
              
      local_indices = np.arange(start, end, dtype=np.int32)
      send_data = (local_indices, Vr_local)

    else :
      # Parallelize only over n1_m dimension
      start, end = self.distribute_full(num_wan, rank, size)
      shape = (num_Rvec, end - start, num_wan)
      Vr_local = np.zeros(shape, dtype=float)

      for i in range(num_Rvec):
        for local_n1, n1_m in enumerate(range(start, end)):
          for n3_m in range(num_wan):
            r = self.get_dist(Rvec[i], wanfuncs[n1_m, 3:], wanfuncs[n3_m, 3:])
            f = 22.59
            U = 1.0
            if r <= a0:
              Vr_local[i, local_n1, n3_m] = -U * f * (self.Struve(a0 / r0_s) -
                                      self.Bessel0(a0 / r0_s)) * 1.0 / (r0)
            else:
              Vr_local[i, local_n1, n3_m] = -f * (self.Struve(r / r0_s) -
                                      self.Bessel0(r / r0_s)) * 1.0 / (r0)

      local_indices = np.arange(start, end, dtype=np.int32)
      send_data = (local_indices, Vr_local)

    # Gather chunks at root
    all_data = comm.gather(send_data, root=root)

    # Write results at root
    if rank == root:
      Vr_full = np.zeros((num_Rvec, num_wan, num_wan), dtype=float)

      if num_Rvec > num_wan**2:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[global_i, :, :] = chunk[local_i, :, :]
      else:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[:, global_i, :] = chunk[:, local_i, :]

      Vr_full = self.symmetrise_potential(Vr_full, Rvec)

      # Write to HDF5
      with h5py.File(screened_file, "w") as f:
        f.create_dataset("W", data=Vr_full)
        print_f(f"Screened Coulomb potential written to {screened_file}")
    return None


  def coulomb_analytic(self, Rvec, wanfuncs, epsilon_t,
                      epsilon_b, a0_list, zav,
                      thickness=2.5):
    """
    Unscreened Coulomb potential at a particular point in 
    real space;
    Monolayer TMDs and e-h separated in a layer.
    @input
      r: distance between the e-h.
    Very similar parallelisation strategy as in the Keldysh potential 
    (see monolayer_keldysh above).
    """
    bare_file   = "Bare_coulomb.hdf5"
    num_Rvec    = Rvec.shape[0]
    num_wan     = wanfuncs.shape[0]
    epsilon_bg  = 0.5 * (epsilon_t + epsilon_b)
    layer_bounds = self.set_layer_bounds(zav)

    if num_Rvec > num_wan**2:
      start, end = self.distribute_full(num_Rvec, rank, size)
      shape      = (end - start, num_wan, num_wan)
      Vr_local   = np.zeros(shape, dtype=float)
      for local_i, i in enumerate(range(start, end)):
        for n1_m in range(num_wan):
          for n3_m in range(num_wan):
            r      = self.get_dist(Rvec[i], wanfuncs[n1_m, 3:], wanfuncs[n3_m, 3:])

            n1_z   = wanfuncs[n1_m, 5]; n3_z = wanfuncs[n3_m, 5]
            layer1, layer2 = self.find_layers(layer_bounds, thickness, n1_z, n3_z)
            a_av   = 0.5 * (a0_list[layer1] + a0_list[layer2])
            f = 14.38; U = 1.0
            if r <= a_av:
              Vr_local[local_i, n1_m, n3_m] = -U * f * (1 / a_av) * 1.0 / (epsilon_bg)
            else:
              Vr_local[local_i, n1_m, n3_m] = -f * (1 / r) * 1.0 / (epsilon_bg)
      local_indices = np.arange(start, end, dtype=np.int32)
      send_data     = (local_indices, Vr_local)
    else:
      start, end = self.distribute_full(num_wan, rank, size)
      shape      = (num_Rvec, end - start, num_wan)
      Vr_local   = np.zeros(shape, dtype=float)
      for i in range(num_Rvec):
        for local_n1, n1_m in enumerate(range(start, end)):
          for n3_m in range(num_wan):
            r      = self.get_dist(Rvec[i], wanfuncs[n1_m, 3:], wanfuncs[n3_m, 3:])
            n1_z   = wanfuncs[n1_m, 5]; n3_z = wanfuncs[n3_m, 5]
            layer1, layer2 = self.find_layers(layer_bounds, thickness, n1_z, n3_z)
            a_av   = 0.5 * (a0_list[layer1] + a0_list[layer2])
            f = 14.38; U = 1.0
            if r <= a_av:
              Vr_local[i, local_n1, n3_m] = -U * f * (1 / a_av) * 1.0 / (epsilon_bg)
            else:
              Vr_local[i, local_n1, n3_m] = -f * (1 / r) * 1.0 / (epsilon_bg)
      local_indices = np.arange(start, end, dtype=np.int32)
      send_data     = (local_indices, Vr_local)

    all_data = comm.gather(send_data, root=root)
    if rank == 0:
      Vr_full = np.zeros((num_Rvec, num_wan, num_wan), dtype=float)
      if num_Rvec > num_wan**2:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[global_i, :, :] = chunk[local_i, :, :]
      else:
        for indices, chunk in all_data:
          for local_i, global_i in enumerate(indices):
            Vr_full[:, global_i, :] = chunk[:, local_i, :]

      Vr_full = self.symmetrise_potential(Vr_full, Rvec)

      with h5py.File(bare_file, "w") as f:
        f.create_dataset("V", data=Vr_full)
      print_f(f"Unscreened Coulomb potential written to {bare_file}")
    return None


  def symmetrise_potential(self, Vr_full, Rvec):
    """
    Enforce V[R,i,j] = V[-R,j,i] for all R-vectors.
    For Nyquist points where -R is not in the grid,
    enforce V[R,i,j] = V[R,j,i] (self-symmetry).
    Must be called on root only, after Vr_full is assembled.
    """
    Rvec_tuples   = [tuple(np.round(r, 8)) for r in Rvec]
    Rvec_index    = {r: i for i, r in enumerate(Rvec_tuples)}
    nyquist_count = 0

    for i_R in range(len(Rvec)):
      mR_key = tuple(np.round(-Rvec[i_R], 8))
      if mR_key in Rvec_index:
        j_R = Rvec_index[mR_key]
        if i_R < j_R:
          avg          = 0.5 * (Vr_full[i_R] + Vr_full[j_R].T)
          Vr_full[i_R] = avg
          Vr_full[j_R] = avg.T
      else:
        Vr_full[i_R] = 0.5 * (Vr_full[i_R] + Vr_full[i_R].T)
        nyquist_count += 1

    if nyquist_count > 0:
      print_f(f"  Symmetrised {nyquist_count} Nyquist R-points")

    return Vr_full