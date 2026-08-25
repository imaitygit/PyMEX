# Option 1: module-level directives (recommended)
# cython: boundscheck=False, wraparound=False
import numpy as np
cimport function
import cython
from cython.parallel import prange, parallel
cdef extern from "math.h" nogil:
  double sqrt(double)
  double log(double)
  double cos(double)
  double sin(double)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex compute_hole_sum(
  double complex[:, :, :] C_nv_k,
  double[:, :] Rvec, double[:, :] kvec,
  long[:, :] ind_wf,
  long v, long vp, long k, long kp
  ):
  """
  Compute the hole contributions for the electron
  density for a specific exciton.
  """

  cdef long nind = ind_wf.shape[0]
  cdef long N = Rvec.shape[0]
  cdef long i, n3, r, n3min, n3max
  cdef double complex[:] tmp_array = np.zeros(nind, dtype=np.complex128)


  cdef double dkx = kvec[k, 0] - kvec[kp, 0]
  cdef double dky = kvec[k, 1] - kvec[kp, 1]
  cdef double dkz = kvec[k, 2] - kvec[kp, 2]

  cdef double dot
  cdef double complex phase, contrib
  cdef double complex hole_sum = 0.0

  # Parallel computation - fully nogil
  with nogil:
    for i in prange(nind, schedule='dynamic'):
      n3min = ind_wf[i, 1]
      n3max = ind_wf[i, 2]
      contrib = 0.0 + 0.0j
      for n3 in range(n3min, n3max):
        for r in range(N):
          dot = (dkx * Rvec[r, 0] + 
                 dky * Rvec[r, 1] + 
                 dkz * Rvec[r, 2])
          phase = function.cexp(-1j * dot)
          contrib = contrib +  (phase * C_nv_k[n3, v, k] * 
                                function.conj(C_nv_k[n3, vp, kp]))
      tmp_array[i] = contrib

  # Serial reduction
  for i in range(nind):
    hole_sum += tmp_array[i]
  return hole_sum


@cython.boundscheck(False)
@cython.wraparound(False)
def opt_electron_for_exciton(double complex[:,:,:] A_Scvk,
                         double[:, :] Rvec, double[:, :] kvec,
                         double complex[:, :, :] C_nc_k,
                         double complex[:, :, :] C_nv_k,
                         long[:, :] ind_wf, double[:, :] map_wf,
                         long i_global):
  """
  Compute electron distribution function for specific exciton, S
  after summing over hole contributions.
  """
  cdef long c, v, k, cp, vp, kp, r, n1
  cdef double complex tmp, hole_sum, phase, tmp2
  cdef double dkx, dky, dkz, dot

  cdef long N = Rvec.shape[0]
  cdef long cmax = C_nc_k.shape[1]
  cdef long vmax = C_nv_k.shape[1]
  cdef long kmax = C_nc_k.shape[2]

  cdef long n1min = ind_wf[i_global, 1]
  cdef long n1max = ind_wf[i_global, 2]
  cdef double complex[:] tmp_array = np.zeros(N, dtype=np.complex128)


  # [r dimensional array]
  for r in range(N):
    tmp = 0.0 + 0.0j
    for c in range(cmax):
      for v in range(vmax):
        for k in range(kmax):
          for cp in range(cmax):
            for vp in range(vmax):
              for kp in range(kmax):
                hole_sum = compute_hole_sum(C_nv_k, Rvec, kvec,
                                            ind_wf, v, vp, k, kp)
                dkx = kvec[k, 0] - kvec[kp, 0]
                dky = kvec[k, 1] - kvec[kp, 1]
                dkz = kvec[k, 2] - kvec[kp, 2]
                dot = (dkx * Rvec[r, 0] +
                       dky * Rvec[r, 1] +
                       dkz * Rvec[r, 2])
                phase = function.cexp(1j * dot)
                tmp2 = 0.0 + 0.0j
                for n1 in range(n1min, n1max):  
                  tmp2 = tmp2 +\
                    (function.conj(C_nc_k[n1,c,k]) * C_nc_k[n1,cp,kp]) 
                tmp = tmp + (function.conj(A_Scvk[c, v, k]) * A_Scvk[cp, vp, kp] * \
                       hole_sum * phase * tmp2)
    tmp_array[r] = tmp
  return tmp_array
  
  
  
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex opt_exciton_r_partial(double complex[:, :, :] AS_cvk,
                                          double complex[:, :, :] C_nc_k,
                                          double complex[:, :] tmp_h_vk,
                                          double complex[:] phase_k,
                                          long[:] ind_wf_loc) noexcept nogil:
  """
  Compute the exciton envelope at one Wannier function center.
  phase_k and tmp_h_vk are precomputed outside.
  """
  cdef long c, v, k, n1
  cdef long cmax  = AS_cvk.shape[0]
  cdef long vmax  = AS_cvk.shape[1]
  cdef long kmax  = AS_cvk.shape[2]
  cdef long n1min = ind_wf_loc[1]
  cdef long n1max = ind_wf_loc[2]

  cdef double complex tmp_e
  cdef double complex sum = 0.0
  cdef double complex compensation = 0.0
  cdef double complex y, t

  for k in range(kmax):
    for c in range(cmax):
      # electron sum — depends on ind_wf_loc, stays inside i loop
      tmp_e = 0.0
      for n1 in range(n1min, n1max):
        tmp_e += C_nc_k[n1, c, k] * phase_k[k]

      for v in range(vmax):
        y = AS_cvk[c, v, k] * tmp_h_vk[v, k] * tmp_e - compensation
        t = sum + y
        compensation = (t - sum) - y
        sum = t

  return sum


@cython.boundscheck(False)
@cython.wraparound(False)
def opt_exciton_r(double complex[:, :, :] AS_cvk,
                  double[:, :] Rvec, double[:, :] kvec,
                  double complex[:, :, :] C_nc_k,
                  double complex[:, :, :] C_nv_k,
                  long[:, :] ind_wf, double[:, :] map_wf,
                  long[:] n3list,
                  double[:] rh):
  """
  Compute the exciton envelope for all Wannier function centers.
  Returns:
    Xr[n_wf, n_R]: Exciton envelope in real space
  """
  cdef long i, r, v, k, n3
  cdef long N     = Rvec.shape[0]
  cdef long nind  = ind_wf.shape[0]
  cdef long vmax  = C_nv_k.shape[1]
  cdef long kmax  = C_nc_k.shape[2]
  cdef long n3min = n3list[1]
  cdef long n3max = n3list[2]

  cdef double complex[:, :] Xr = np.zeros((nind, N), dtype=np.complex128)

  # precompute hole sums per (v, k) — independent of i and r 
  cdef double complex[:, :] tmp_h_vk = np.zeros((vmax, kmax), dtype=np.complex128)
  for v in range(vmax):
    for k in range(kmax):
      for n3 in range(n3min, n3max):
        tmp_h_vk[v, k] += function.conj(C_nv_k[n3, v, k])

  # precompute phase per (k, r) — independent of i 
  cdef double complex[:, :] phase_kr = np.zeros((kmax, N), dtype=np.complex128)
  for r in prange(N, nogil=True, schedule='static'):
    for k in range(kmax):
      phase_kr[k, r] = function.cexp(1j * (
        kvec[k, 0] * Rvec[r, 0] +
        kvec[k, 1] * Rvec[r, 1] +
        kvec[k, 2] * Rvec[r, 2]
      ))

  # main loop — prange over r, serial over i
  for i in range(nind):
    for r in prange(N, nogil=True, schedule='static'):
      Xr[i, r] = opt_exciton_r_partial(
        AS_cvk,
        C_nc_k,
        tmp_h_vk,
        phase_kr[:, r],
        ind_wf[i, :]
      )

  return Xr


# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: nonecheck=False, initializedcheck=False, infer_types=True
cdef void compute_M_nogil(
  double complex[:, :, :, :] grad_H_ab_k,
  double complex[:, :, :] H_ab_k,
  double[:, :, :] tvec_ab,
  long y,
  long a_max, long b_max,
  double complex[:, :, :] M,
) noexcept nogil:
  cdef long a, b, d
  cdef double complex H_aby
  for a in prange(a_max, schedule='static'):
    for b in range(b_max):
      H_aby = H_ab_k[a, b, y]
      for d in range(3):
        M[a, b, d] = grad_H_ab_k[a, b, y, d] + function._Complex_I * tvec_ab[a, b, d] * H_aby

cdef void compute_P_partial(
  double complex[:, :, :] C_nc_k,
  double complex[:, :, :] M,
  long a, long j, long y,
  long b_max,
  double complex[:, :] P,
) noexcept nogil:
  cdef long b
  cdef double complex tmp0, tmp1, tmp2, c_val
  tmp0 = 0.0; tmp1 = 0.0; tmp2 = 0.0
  for b in range(b_max):
    c_val = C_nc_k[b, j, y]
    tmp0 = tmp0 + c_val * M[a, b, 0]
    tmp1 = tmp1 + c_val * M[a, b, 1]
    tmp2 = tmp2 + c_val * M[a, b, 2]
  P[a, 0] = tmp0
  P[a, 1] = tmp1
  P[a, 2] = tmp2

def compute_conductivity(
  double complex[:, :, :] eigvec,
  double complex[:, :, :] C_nc_k,
  double complex[:, :, :] C_nv_k,
  double complex[:, :, :, :] grad_H_ab_k,
  double complex[:, :, :] H_ab_k,
  double[:, :, :] tvec_ab,
):
  cdef:
    long a, j, x, y, d
    long a_max = C_nv_k.shape[0]
    long b_max = C_nc_k.shape[0]
    long num_c = C_nc_k.shape[1]
    long num_v = C_nv_k.shape[1]
    long num_k = C_nc_k.shape[2]
    double complex A_cvk, coeff
    double complex tmp[3]
    double complex[:, :, :] M = np.empty((a_max, b_max, 3), dtype=np.complex128)
    double complex[:, :]    P = np.empty((a_max, 3),        dtype=np.complex128)

  for d in range(3):
    tmp[d] = 0.0

  for y in range(num_k):
    compute_M_nogil(grad_H_ab_k, H_ab_k, tvec_ab, y, a_max, b_max, M)
    for j in range(num_c):
      for a in prange(a_max, nogil=True, schedule='static'):
        compute_P_partial(C_nc_k, M, a, j, y, b_max, P)
      for x in range(num_v):
        A_cvk = eigvec[j, x, y]
        for a in range(a_max):
          coeff = function.conj(C_nv_k[a, x, y])
          tmp[0] = tmp[0] + coeff * P[a, 0] * A_cvk
          tmp[1] = tmp[1] + coeff * P[a, 1] * A_cvk
          tmp[2] = tmp[2] + coeff * P[a, 2] * A_cvk

  return np.array([tmp[0], tmp[1], tmp[2]], dtype=np.complex128)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_ew_nogil(
  double[:, :] Rvec,
  double[:] weight_Rvec,
  double x0, double x1, double x2,
  long m2,
  double complex[:] ew,
) noexcept nogil:
  cdef long f
  cdef double dotR
  for f in range(m2):
    dotR = x0*Rvec[f,0] + x1*Rvec[f,1] + x2*Rvec[f,2]
    ew[f] = function.cexp(1j * dotR) * weight_Rvec[f]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_G_nogil(
  double[:, :, :] V_r,
  double complex[:] ew,
  long m1, long m2,
  double complex[:, :] G,
) noexcept nogil:
  cdef long i, j, f
  for i in range(m1):
    for j in range(m1):
      G[i, j] = 0.0
  for f in range(m2):
    for i in range(m1):
      for j in range(m1):
        G[i, j] = G[i, j] + V_r[f, i, j] * ew[f]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void compute_B_e_nogil(
  double complex[:, :, :] C_nv_k,
  long[:, :] wanfuncs,
  long v, long k, long vp, long kp,
  long m1,
  double complex[:] B_e_arr,
) noexcept nogil:
  cdef long j, e
  cdef double complex B_e
  for j in range(m1):
    B_e = 0.0
    for e in range(wanfuncs[j,1], wanfuncs[j,2]):
      B_e = B_e + C_nv_k[e, v, k] * function.conj(C_nv_k[e, vp, kp])
    B_e_arr[j] = B_e


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double complex H_optfull_partial(
  double complex[:, :, :] C_nc_kplusQ,
  double complex[:, :, :] C_nv_k,
  long c, long v, long k,
  long cp, long vp, long kp,
  long[:, :] wanfuncs,
  double complex[:, :] G_keld,
  double complex[:, :] G_coul,
  double complex[:] B_e_arr,
  long i
) noexcept nogil:
  cdef:
    long j, d, e
    long m1 = wanfuncs.shape[0]
    double complex A_d = 0.0
    double complex D_d = 0.0
    double complex E_e
    double complex partial1, partial2
    double complex val

  for d in range(wanfuncs[i,1], wanfuncs[i,2]):
    A_d = A_d + C_nc_kplusQ[d, cp, kp] * function.conj(C_nc_kplusQ[d, c, k])
    D_d = D_d + function.conj(C_nc_kplusQ[d, c, k]) * C_nv_k[d, v, k]

  partial1 = 0.0
  partial2 = 0.0
  for j in range(m1):
    E_e = 0.0
    for e in range(wanfuncs[j,1], wanfuncs[j,2]):
      E_e = E_e + function.conj(C_nv_k[e, vp, kp]) * C_nc_kplusQ[e, cp, kp]
    partial1 = partial1 + B_e_arr[j] * G_keld[i, j]
    partial2 = partial2 + E_e * G_coul[i, j]

  val = A_d * partial1 - D_d * partial2
  return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def H_optfull_thread(
  double complex[:, :, :] C_nc_kplusQ,
  double complex[:, :, :] C_nv_k,
  double[:, :] E_ckplusQ,
  double[:, :] E_vk,
  long c, long v, long k,
  long[:, :] wanfuncs,
  double[:, :] Rvec,
  double[:] weight_Rvec,
  double[:, :] kvec,
  double[:] Qvec,
  double[:, :, :] V_r_keld,
  double[:, :, :] V_r_coul,
):
  cdef:
    long cp, vp, kp, i
    long num_c = C_nc_kplusQ.shape[1]
    long num_v = C_nv_k.shape[1]
    long num_k = C_nc_kplusQ.shape[2]
    long m1    = wanfuncs.shape[0]
    long m2    = Rvec.shape[0]
    double E_cvkplusQ
    double complex tmp

    double complex[:] ew_k      = np.empty(m2, dtype=np.complex128)
    double complex[:] ew_Q      = np.empty(m2, dtype=np.complex128)
    double complex[:, :] G_keld = np.empty((m1, m1), dtype=np.complex128)
    double complex[:, :] G_coul = np.empty((m1, m1), dtype=np.complex128)
    double complex[:] tmp_array = np.empty(m1, dtype=np.complex128)
    double complex[:] B_e_arr   = np.empty(m1, dtype=np.complex128)
    double complex[:, :, :] arr = np.zeros((num_c, num_v, num_k),
                                            dtype=np.complex128, order='C')

  compute_ew_nogil(Rvec, weight_Rvec,
                   Qvec[0], Qvec[1], Qvec[2],
                   m2, ew_Q)

  compute_G_nogil(V_r_coul, ew_Q, m1, m2, G_coul)

  for kp in range(num_k):

    compute_ew_nogil(Rvec, weight_Rvec,
                     kvec[k,0]-kvec[kp,0],
                     kvec[k,1]-kvec[kp,1],
                     kvec[k,2]-kvec[kp,2],
                     m2, ew_k)

    compute_G_nogil(V_r_keld, ew_k, m1, m2, G_keld)

    for vp in range(num_v):

      compute_B_e_nogil(C_nv_k, wanfuncs, v, k, vp, kp, m1, B_e_arr)

      for cp in range(num_c):

        for i in prange(m1, nogil=True, schedule='static'):
          tmp_array[i] = H_optfull_partial(
            C_nc_kplusQ, C_nv_k,
            c, v, k, cp, vp, kp,
            wanfuncs, G_keld, G_coul, B_e_arr, i
          )

        tmp = 0.0
        for i in range(m1):
          tmp = tmp + tmp_array[i]

        E_cvkplusQ = (E_ckplusQ[c, k] - E_vk[v, k]) if (
          cp == c and vp == v and kp == k) else 0.0
        arr[cp, vp, kp] = tmp / num_k + E_cvkplusQ

  return arr.base if arr.base is not None else np.asarray(arr, order='C')


# Old sigma_xx refactored
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double complex _sigma_xx_nogil(
    double complex[:,:,:] myeigvnew,
    double complex[:,:,:] C_nc_k,
    double complex[:,:,:] C_nv_k,
    double complex[:,:,:] gradx_Hk,
    double complex[:,:,:] Hk,
    double[:,:] t_x) noexcept nogil:

  cdef long n1, n2, y, j, x
  cdef long n1_max = C_nc_k.shape[0]
  cdef long n2_max = C_nc_k.shape[0]
  cdef long cb = C_nc_k.shape[1]
  cdef long vb = C_nv_k.shape[1]
  cdef long k = C_nc_k.shape[2]

  cdef double complex s1, tmp1 = 0.0 + 0.0j, tmp = 0.0 + 0.0j

  for j in range(cb):
    for x in range(vb):
      for y in range(k):
        # s1 reset for each (j,x,y)
        s1 = 0.0 + 0.0j

        for n1 in range(n1_max):
          # Parallel sum over n2 into s1 with reduction
          for n2 in prange(n2_max, nogil=True, schedule='static'):
            #s1 += (
            #  function.conj(C_nv_k[n1, x, y]) *
            #  C_nc_k[n2, j, y] *
            #  (gradx_Hk[n1, n2, y] + 1j * t_x[n1, n2] * Hk[n1, n2, y])
            #)
            s1 += (
              function.conj(C_nv_k[n2, x, y]) *
              C_nc_k[n1, j, y] *
              (gradx_Hk[n1, n2, y] + 1j * t_x[n1, n2] * Hk[n1, n2, y])
            )
        tmp1 += s1 * myeigvnew[j, x, y]

  tmp += (function.cabs(tmp1)**2)
  return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
def sigma_xx(
    double complex[:, :, :] myeigvnew,
    double complex[:, :, :] C_nc_k,
    double complex[:, :, :] C_nv_k,
    double complex[:, :, :] gradx_Hk,
    double complex[:, :, :] Hk,
    double[:, :] t_x,
    ):
    # Calls the cdef function with nogil
    return _sigma_xx_nogil(myeigvnew, 
                           C_nc_k, C_nv_k, 
                           gradx_Hk, Hk, t_x)


# ── inline J0 — Abramowitz & Stegun 9.4.1/9.4.3, error < 5e-8 ────────
cdef inline double j0(double x) nogil:
  cdef double ax, z, p, q, xx
  ax = x if x >= 0.0 else -x
  if ax < 8.0:
    z = x * x
    p = (57568490574.0 + z * (-13362590354.0 + z * (651619640.7
          + z * (-11214424.18 + z * (77392.33017 + z * (-184.9052456))))))
    q = (57568490411.0 + z * (1029532985.0 + z * (9494680.718
          + z * (59272.64853 + z * (267.8532712 + z * 1.0)))))
    return p / q
  else:
    z  = 8.0 / ax
    xx = ax - 0.785398164
    p  = 1.0 + z*z*(-0.1098628627e-2 + z*z*(0.2734510407e-4
           + z*z*(-0.2073370639e-5 + z*z*0.2093887211e-6)))
    q  = (-0.1562499995e-1 + z*z*(0.1430488765e-3
           + z*z*(-0.6911147651e-5 + z*z*(0.7621095161e-6
           - z*z*0.934945152e-7))))
    return sqrt(0.636619772 / ax) * (cos(xx) * p - z * sin(xx) * q)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fill_keldysh_Rvec(
  double[:, :, :] Vr_local,    # (local_Rvec, num_wan, num_wan) — output
  double[:, :] Rvec,           # (num_Rvec, 3)
  double[:, :] pos,            # (num_wan, 3) — wanfuncs[:, 3:6]
  double[:, :] shift_vecs,     # (n_shifts, 3)
  int[:] layer1_arr,           # (num_wan,)
  double[:, :] a_av_mat,       # (n_layers, n_layers)
  int[:, :] phi_key_idx,       # (n_layers, n_layers) — -1 if missing
  double[:, :] phi_q_vals,     # (n_keys, 2*n_gl) — phi(q) on GL nodes
  double[:] gl_q,              # (2*n_gl,) — GL nodes for both segments
  double[:] gl_w,              # (2*n_gl,) — GL weights for both segments
  int n_gl,                    # points per segment
  int r_start,                 # MPI offset
):
  """
  Fill Vr_local for the num_Rvec > num_wan**2 branch.
  prange over local_i — true OpenMP parallelism.
  """
  cdef int local_i, i, n1, n3, s, g, key_idx
  cdef int num_wan    = pos.shape[0]
  cdef int local_Rvec = Vr_local.shape[0]
  cdef int n_shifts   = shift_vecs.shape[0]
  cdef int n_gl_total = 2 * n_gl
  cdef double dx, dy, dz, tx, ty, tz, dist2, best, r, a_av, V
  cdef double two_pi = 6.283185307179586

  for local_i in prange(local_Rvec, nogil=True, schedule='static'):
    i = local_i + r_start
    for n1 in range(num_wan):
      for n3 in range(num_wan):

        # minimum image distance — same logic as get_dist
        dx   = Rvec[i,0] + pos[n3,0] - pos[n1,0]
        dy   = Rvec[i,1] + pos[n3,1] - pos[n1,1]
        dz   = Rvec[i,2] + pos[n3,2] - pos[n1,2]
        best = dx*dx + dy*dy + dz*dz
        for s in range(n_shifts):
          tx    = dx + shift_vecs[s,0]
          ty    = dy + shift_vecs[s,1]
          tz    = dz + shift_vecs[s,2]
          dist2 = tx*tx + ty*ty + tz*tz
          if dist2 < best:
            best = dist2
        r = sqrt(best)

        # phi_key lookup — same logic as find_layers + string lookup
        key_idx = phi_key_idx[layer1_arr[n1], layer1_arr[n3]]
        if key_idx < 0:
          continue

        # r_min = a_av cutoff — same as Hankel_transform effective_r
        a_av = a_av_mat[layer1_arr[n1], layer1_arr[n3]]
        if r < a_av:
          r = a_av

        # 2-segment GL quadrature — replaces scipy.integrate.quad
        V = 0.0
        for g in range(n_gl_total):
          V = V + phi_q_vals[key_idx, g] * j0(gl_q[g] * r) * gl_q[g] * gl_w[g]
        Vr_local[local_i, n1, n3] = V / two_pi


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fill_keldysh_wan(
  double[:, :, :] Vr_local,    # (num_Rvec, local_wan, num_wan) — output
  double[:, :] Rvec,           # (num_Rvec, 3)
  double[:, :] pos,            # (num_wan, 3)
  double[:, :] shift_vecs,     # (n_shifts, 3)
  int[:] layer1_arr,           # (num_wan,)
  double[:, :] a_av_mat,       # (n_layers, n_layers)
  int[:, :] phi_key_idx,       # (n_layers, n_layers)
  double[:, :] phi_q_vals,     # (n_keys, 2*n_gl)
  double[:] gl_q,              # (2*n_gl,)
  double[:] gl_w,              # (2*n_gl,)
  int n_gl,
  int n1_start,                # MPI offset
):
  """
  Fill Vr_local for the num_wan parallelisation branch.
  prange over i — true OpenMP parallelism.
  """
  cdef int i, local_n1, n1, n3, s, g, key_idx
  cdef int num_Rvec   = Rvec.shape[0]
  cdef int local_wan  = Vr_local.shape[1]
  cdef int num_wan    = pos.shape[0]
  cdef int n_shifts   = shift_vecs.shape[0]
  cdef int n_gl_total = 2 * n_gl
  cdef double dx, dy, dz, tx, ty, tz, dist2, best, r, a_av, V
  cdef double two_pi = 6.283185307179586

  for i in prange(num_Rvec, nogil=True, schedule='static'):
    for local_n1 in range(local_wan):
      n1 = local_n1 + n1_start
      for n3 in range(num_wan):

        # minimum image distance
        dx   = Rvec[i,0] + pos[n3,0] - pos[n1,0]
        dy   = Rvec[i,1] + pos[n3,1] - pos[n1,1]
        dz   = Rvec[i,2] + pos[n3,2] - pos[n1,2]
        best = dx*dx + dy*dy + dz*dz
        for s in range(n_shifts):
          tx    = dx + shift_vecs[s,0]
          ty    = dy + shift_vecs[s,1]
          tz    = dz + shift_vecs[s,2]
          dist2 = tx*tx + ty*ty + tz*tz
          if dist2 < best:
            best = dist2
        r = sqrt(best)

        # phi_key lookup
        key_idx = phi_key_idx[layer1_arr[n1], layer1_arr[n3]]
        if key_idx < 0:
          continue

        # r_min = a_av cutoff
        a_av = a_av_mat[layer1_arr[n1], layer1_arr[n3]]
        if r < a_av:
          r = a_av

        # 2-segment GL quadrature
        V = 0.0
        for g in range(n_gl_total):
          V = V + phi_q_vals[key_idx, g] * j0(gl_q[g] * r) * gl_q[g] * gl_w[g]
        Vr_local[i, local_n1, n3] = V / two_pi


