
from __future__ import print_function
import numpy as np
cimport function
import cython
from cython.parallel import prange, parallel
@cython.boundscheck(False)
@cython.wraparound(False)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double complex opt_exciton_r_partial(double complex [:,:,:] AS_cvk,\
                  double [:] Rv, double [:,:] kvec,\
                  double complex [:,:,:] C_nc_k,\
                  double complex [:,:,:]C_nv_k,\
                  long [:] ind_wf_loc, double [:] map_wf_loc,\
                  long [:] n3list) nogil:
  """
  Exciton envelope calculations one of the  Wannier Function
  centers. This helps to speed up the calculations with Threads.
  """
  cdef int c, v, k, cp, vp, kp
  cdef int n1, n3, n1min, n1max, n3min, n3max
  cdef double complex phase, tmp, tmp2, tmp3
  cmax = C_nc_k.shape[1]; vmax = C_nv_k.shape[1]
  kmax = C_nc_k.shape[2]
  n3min = n3list[1]; n3max = n3list[2]
  n1min = ind_wf_loc[1]; n1max = ind_wf_loc[2]
  # ------------
  # Thread will be used later  
  # ------------
  # This for real-space unit cells

  tmp = 0.0
  for c in range(cmax):
    for v in range(vmax):
      for k in range(kmax):
        for cp in range(cmax):
          for vp in range(vmax):
            for kp in range(kmax):
              # Phase- e^{-(i(k-k').Rj1)}
              # Rv is only for a particular unit-cell
              phase = function.cexp(-1j*\
              ( ((kvec[k,0]-kvec[kp,0])*Rv[0])+\
                ((kvec[k,1]-kvec[kp,1])*Rv[1])+\
                ((kvec[k,2]-kvec[kp,2])*Rv[2]) )\
                )
              # Hole coefficenits contributions
              tmp2 = 0.0
              for n3 in range(n3min, n3max):
                tmp2 = tmp2 +\
                (C_nv_k[n3,v,k] * function.conj(C_nv_k[n3,vp,kp]))
              # Electron coefficients for n1
              tmp3 = 0.0
              for n1 in range(n1min, n1max):
                tmp3 = tmp3 +\
                (function.conj(C_nc_k[n1,c,k]) * C_nc_k[n1,cp,kp]) 
              # Terms to sum up for each Wannier centers
              tmp = tmp +\
              (function.conj(AS_cvk[c,v,k])* AS_cvk[cp,vp,kp] *\
               phase * tmp2 * tmp3)
  return tmp


@cython.boundscheck(False)
@cython.wraparound(False)
def opt_exciton_r_thread(double complex [:,:,:] AS_cvk,\
                  double [:,:] Rvec, double [:,:] kvec,\
                  double complex [:,:,:] C_nc_k,\
                  double complex [:,:,:]C_nv_k,\
                  long [:,:] ind_wf, double [:,:] map_wf,\
                  long [:] n3list):
  """
  Exciton envelope calculations for all the  Wannier Function
  centers. This helps to speed up the calculations with Threads.
  Xr[:,:] = (N_Wanniercenter, Numberofunitcells) dimensions
  """
  cdef int i, nind
  cdef double complex [:,:] Xr =\
       np.zeros((ind_wf.shape[0], Rvec.shape[0]), dtype=complex)
  cdef double complex [:] tmp_arr =\
       np.zeros((ind_wf.shape[0]), dtype=complex)
  cdef int N, r
  N = Rvec.shape[0]
  nind = ind_wf.shape[0]

  # 
  for r in range(N):
    # use all the available threads
    for i in prange(nind, nogil=True, schedule='dynamic'):
      tmp_arr[i] = opt_exciton_r_partial(AS_cvk, Rvec[r,:], kvec,\
                    C_nc_k, C_nv_k, ind_wf[i,:], map_wf[i,:],\
                    n3list)
    Xr[:,r] = tmp_arr
  return Xr 
           
      
@cython.boundscheck(False)
@cython.wraparound(False)
def opt_exciton_r(double complex [:,:,:] AS_cvk,\
                  double [:,:] Rvec, double [:,:] kvec,\
                  double complex [:,:,:] C_nc_k,\
                  double complex [:,:,:]C_nv_k,\
                  long [:,:] ind_wf, double [:,:] map_wf,\
                  long [:] n3list):
  """
  Exciton envelope calculations for all the  Wannier Function
  centers. This helps to speed up the calculations with Threads.
  Xr[:,:] = (N_Wanniercenter, Numberofunitcells) dimensions
  """
  cdef int i, nind
  cdef double complex [:,:] Xr =\
       np.zeros((ind_wf.shape[0], Rvec.shape[0]), dtype=complex)
  cdef int N, c, v, k, cp, vp, kp
  cdef int r, n1, n3, n1min, n1max, n3min, n3max
  cdef double complex phase, tmp, tmp2, tmp3
  cmax = C_nc_k.shape[1]; vmax = C_nv_k.shape[1]
  kmax = C_nc_k.shape[2]
  n3min = n3list[1]; n3max = n3list[2]
  N = Rvec.shape[0]
  # Simplest approach
  # ----------------
  # This for real-space unit cells
  nind = ind_wf.shape[0]
  for i in range(nind):
    n1min = ind_wf[i,1]; n1max = ind_wf[i,2]
    for r in range(N):
      tmp = 0.0
      for c in range(cmax):
        for v in range(vmax):
          for k in range(kmax):
            for cp in range(cmax):
              for vp in range(vmax):
                for kp in range(kmax):
                  # Phase- e^{-(i(k-k').Rj1)}
                  phase = function.cexp(-1j*\
                  ( ((kvec[k,0]-kvec[kp,0])*(Rvec[r,0]-map_wf[ind_wf[i,0],0]))+\
                    ((kvec[k,1]-kvec[kp,1])*(Rvec[r,1]-map_wf[ind_wf[i,0],1]))+\
                    ((kvec[k,2]-kvec[kp,2])*Rvec[r,2]))\
                    )
                  # Hole coefficenits contributions
                  tmp2 = 0.0
                  for n3 in range(n3min, n3max):
                    tmp2 = tmp2 +\
                    (C_nv_k[n3,v,k] * function.conj(C_nv_k[n3,vp,kp]))
                  # Electron coefficients for n1
                  tmp3 = 0.0
                  for n1 in range(n1min, n1max):
                    tmp3 = tmp3 +\
                    (function.conj(C_nc_k[n1,c,k]) * C_nc_k[n1,cp,kp]) 
                  # Terms to sum up for each Wannier centers
                  tmp = tmp +\
                  (function.conj(AS_cvk[c,v,k])* AS_cvk[cp,vp,kp] *\
                  phase * tmp2 * tmp3)
      Xr[i,r] = tmp
  return Xr


@cython.boundscheck(False)
def Eigval_perturb(double [:] eigval,\
               double complex [:,:,:,:] eigvnew,\
               double [:,:,:,:] Delta_cvsk,
               double [:,:] eigval_s):
  """
  Correct BSE eigenvalues perturbatively
  """
  cdef int i, c, v, s, k
  e_max = eigval.shape[0]
  c_max = eigvnew.shape[1]; v_max = eigvnew.shape[2]
  k_max = eigvnew.shape[3]; s_max = Delta_cvsk.shape[2]
  cdef double complex tmp

  with cython.nogil:
    for i in range(e_max):
      for s in range(s_max):
        tmp = 0.0
        for c in range(c_max):
          for v in range(v_max):
            for k in range(k_max):
              tmp = tmp + ((function.cabs(eigvnew[i,c,v,k])**2.)*\
                    Delta_cvsk[c,v,s,k])
        eigval_s[i,s] = eigval[i] + tmp.real
  return eigval_s


@cython.boundscheck(False)
def grad_Hk_opt( double [:,:] k,\
                 double complex [:,:,:] h_r_ab,\
                 double [:,:] r_ws,\
                 double [:,:] A, double [:,:] B,
                 double complex [:,:,:] grad_h_k_ab):
  """
  Convention II:
    H(k): \sum_{R} h_r_ab*e^{ik.R} (1j*R_x)
  Cythonized part of grad_Hk calculations
  Note that this implements grad_Hk calculations
  as you expect analytically.
   
  """
  cdef int i, m, n, j, kmax, a, b, r
  cdef double complex tmp
  cdef int dim

  kmax = k.shape[0]
  a = h_r_ab.shape[1]
  b = h_r_ab.shape[2]
  r = h_r_ab.shape[0]
  dim = 3 

  cdef double [:] r_j = np.zeros((dim), dtype=float)
  cdef double [:] k_i = np.zeros((dim), dtype=float)

  with cython.nogil:
    for i in range(kmax):
      for m in range(a):
        for n in range(b):
          tmp = 0.0
          for j in range(r):
            # Matrix-multiply by hand
            r_j[0] = r_ws[j][0]*A[0][0]+r_ws[j][1]*A[1][0]+\
                     r_ws[j][2]*A[2][0]
            r_j[1] = r_ws[j][0]*A[0][1]+r_ws[j][1]*A[1][1]+\
                     r_ws[j][2]*A[2][1]
            r_j[2] = r_ws[j][0]*A[0][2]+r_ws[j][1]*A[1][2]+\
                     r_ws[j][2]*A[2][2]
            k_i[0] = k[i][0]*B[0][0]+k[i][1]*B[1][0]+\
                     k[i][2]*B[2][0]
            k_i[1] = k[i][0]*B[0][1]+k[i][1]*B[1][1]+\
                     k[i][2]*B[2][1]
            k_i[2] = k[i][0]*B[0][2]+k[i][1]*B[1][2]+\
                     k[i][2]*B[2][2]
            # Only x-components for now
            # \sum_{R} h_r_ab*e^{ik.R} (1j*R_x)
            tmp = tmp + (h_r_ab[j][m][n]* function.cexp(1j*\
                        ((r_j[0]*k_i[0]) + (r_j[1]*k_i[1])+\
                         (r_j[2]*k_i[2])))* 1j*r_j[0])
          grad_h_k_ab[i][m][n] = tmp
  return grad_h_k_ab


@cython.boundscheck(False)
@cython.cdivision(True)
def sigma_xx_full_E(double [:] eigval,\
               double complex [:,:,:,:] eigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma):
  """
  Computes the \sigma_xx as in Phys. Rev. B 97, 205409(2018).
  """
  cdef int n1,n2,y, i,j,x
  cdef int n1_max, n2_max, m, cb,vb,k
  cdef double complex s1, tmp1, tmp
  # squareroot of 2pi
  cdef double sqrt2pi = 2.5066282746310002
 
  m = eigval.shape[0]
  n1_max = C_nc_k.shape[0]
  n2_max = C_nc_k.shape[0]
  cb = C_nc_k.shape[1]
  vb = C_nv_k.shape[1]
  k = C_nc_k.shape[2]
  tmp = 0.0

  # For every BSE eigenvalues
  with cython.nogil:
    for i in range(m):
      tmp1 = 0.0
      # Sum over c,v,k
      for j in range(cb):
        for x in range(vb):
          for y in range(k):
            s1 = 0.0
            # Sum over orbitals/wannier functions
            # This is the *badly scalable* and problematic part
            for n1 in range(n1_max):
              for n2 in range(n2_max):
                s1 = s1 + (function.conj(C_nv_k[n1,x,y])*\
                     C_nc_k[n2,j,y]*gradx_Hk[n1,n2,y])
            # Exciton eigenvectors go here
            tmp1 = tmp1 + (s1*eigvnew[i,j,x,y])
      # Delta function is represented by Gaussian
      tmp = tmp + ((function.cabs(tmp1)**2)*\
            function.cexp((-(E_ph-eigval[i])*(E_ph-eigval[i]))/(2*sigma*sigma))*\
                    1/(sigma*sqrt2pi))
  return tmp


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double complex sigma_partial_full(double myeigval,\
               double complex [:,:,:] myeigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma) nogil:
  cdef int n1,n2,y, j,x
  cdef int n1_max, n2_max, cb,vb,k
  cdef double complex s1, tmp1, tmp
  # squareroot of 2pi
  cdef double sqrt2pi = 2.5066282746310002

  n1_max = C_nc_k.shape[0]
  n2_max = C_nc_k.shape[0]
  cb = C_nc_k.shape[1]
  vb = C_nv_k.shape[1]
  k = C_nc_k.shape[2]
  tmp1 = 0.0; tmp = 0.0

  # Sum over c,v,k
  for j in range(cb):
    for x in range(vb):
      for y in range(k):
        s1 = 0.0
        # Sum over orbitals/wannier functions
        for n1 in range(n1_max):
          for n2 in range(n2_max):
            s1 = s1 + (function.conj(C_nv_k[n1,x,y])*\
                 C_nc_k[n2,j,y]*gradx_Hk[n1,n2,y])
        # Exciton eigenvectors go here
        tmp1 = tmp1 + (s1*myeigvnew[j,x,y])
  # Delta function is represented by Gaussian
  tmp = tmp + ((function.cabs(tmp1)**2)*\
        function.cexp((-(E_ph-myeigval)*(E_ph-myeigval))/(2*sigma*sigma))*\
                1/(sigma*sqrt2pi))
  return tmp


@cython.boundscheck(False)
@cython.cdivision(True)
def sigma_xx_full_E_thread(double [:] eigval,\
               double complex [:,:,:,:] eigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma):
  """
  Computes the \sigma_xx as in Phys. Rev. B 97, 205409(2018).
  """
  cdef int i
  cdef double complex tmp2

  m = eigval.shape[0]
  tmp2 = 0.0
  
  # Sum over BSE eigenvalues using threads
  # Uses default number of threads available
  for i in prange(m, nogil=True, schedule='dynamic'):
    tmp2 += sigma_partial_full(eigval[i],\
               eigvnew[i,:,:,:], C_nc_k, C_nv_k, gradx_Hk, E_ph,\
               sigma)
  return tmp2


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double complex sigma_partial_per(double[:] myeigval,\
               double complex [:,:,:] myeigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma) nogil:

  cdef int n1,n2,y, i,j,x
  cdef int n1_max, n2_max, m, cb,vb,k
  cdef double complex s1, tmp1, tmp
  # squareroot of 2pi
  cdef double sqrt2pi = 2.5066282746310002
 
  n1_max = C_nc_k.shape[0]; n2_max = C_nc_k.shape[0]
  cb = C_nc_k.shape[1]; vb = C_nv_k.shape[1]
  k = C_nc_k.shape[2]
  tmp1 = 0.0; tmp = 0.0

  # Sum over c,v,k
  for j in range(cb):
    for x in range(vb):
      for y in range(k):
        s1 = 0.0
        # Sum over orbitals/wannier functions
        # This is the *badly scalable* and problematic part
        for n1 in range(n1_max):
          for n2 in range(n2_max):
            s1 = s1 + (function.conj(C_nv_k[n1,x,y])*\
                 C_nc_k[n2,j,y]*gradx_Hk[n1,n2,y])
        # Exciton eigenvectors go here
        tmp1 = tmp1 + (s1*myeigvnew[j,x,y])
  # Delta function is represented by Gaussian
  tmp = tmp + ((function.cabs(tmp1)**2)*\
        function.cexp((-(E_ph-myeigval[0])*\
        (E_ph-myeigval[0]))/(2*sigma*sigma))*\
         1/(sigma*sqrt2pi)) +\
         ((function.cabs(tmp1)**2)*\
         function.cexp((-(E_ph-myeigval[1])*\
         (E_ph-myeigval[1]))/(2*sigma*sigma))*\
         1/(sigma*sqrt2pi))
  return tmp


@cython.boundscheck(False)
@cython.cdivision(True)
def sigma_xx_per_E_thread(double [:,:] eigval_s,\
               double complex [:,:,:,:] eigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma):
  """
  Computes the \sigma_xx as in Phys. Rev. B 97, 205409(2018).
  NOTE: Spin is 2
  This part utilizes threads (see below)
  NOTE: eigval for full vs eigval_s for perturbation.
  """
  cdef int i
  cdef double complex tmp2

  m = eigval_s.shape[0]
  tmp2 = 0.0

  # Sum over BSE eigenvalues using threads
  # Uses default number of threads available
  for i in prange(m, nogil=True, schedule='dynamic'):
    tmp2 += sigma_partial_per(eigval_s[i,:],\
               eigvnew[i,:,:,:], C_nc_k, C_nv_k, gradx_Hk, E_ph,\
               sigma)
  return tmp2


@cython.boundscheck(False)
@cython.cdivision(True)
def sigma_xx_per_E(double [:,:] eigval_s,\
               double complex [:,:,:,:] eigvnew,\
               double complex [:,:,:] C_nc_k,\
               double complex [:,:,:] C_nv_k,\
               double complex [:,:,:] gradx_Hk,\
               double E_ph, double sigma):
  """
  Computes the \sigma_xx as in Phys. Rev. B 97, 205409(2018).
  NOTE: Spin is 2 
  """
  cdef int n1,n2,y, i,j,x
  cdef int n1_max, n2_max, m, cb,vb,k
  cdef double complex s1, tmp1, tmp
  # squareroot of 2pi
  cdef double sqrt2pi = 2.5066282746310002
 
  m = eigval_s.shape[0]
  n1_max = C_nc_k.shape[0]
  n2_max = C_nc_k.shape[0]
  cb = C_nc_k.shape[1]
  vb = C_nv_k.shape[1]
  k = C_nc_k.shape[2]
  tmp = 0.0

  # For every BSE eigenvalues
  with cython.nogil:
    for i in range(m):
      tmp1 = 0.0
      # Sum over c,v,k
      for j in range(cb):
        for x in range(vb):
          for y in range(k):
            s1 = 0.0
            # Sum over orbitals/wannier functions
            # This is the *badly scalable* and problematic part
            for n1 in range(n1_max):
              for n2 in range(n2_max):
                s1 = s1 + (function.conj(C_nv_k[n1,x,y])*\
                     C_nc_k[n2,j,y]*gradx_Hk[n1,n2,y])
            # Exciton eigenvectors go here
            tmp1 = tmp1 + (s1*eigvnew[i,j,x,y])
      # Delta function is represented by Gaussian
      tmp = tmp + ((function.cabs(tmp1)**2)*\
            function.cexp((-(E_ph-eigval_s[i,0])*\
            (E_ph-eigval_s[i,0]))/(2*sigma*sigma))*\
             1/(sigma*sqrt2pi)) +\
             ((function.cabs(tmp1)**2)*\
             function.cexp((-(E_ph-eigval_s[i,1])*\
             (E_ph-eigval_s[i,1]))/(2*sigma*sigma))*\
             1/(sigma*sqrt2pi))
  return tmp


@cython.boundscheck(False)
@cython.cdivision(True)
def H_optk(double complex [:,:,:] C_nc_k,\
           double complex [:,:,:] C_nv_k,\
           double [:,:] E_ck, double [:,:] E_vk,\
           long c, long v, long [:] klist,\
           long [:,:] myatoms, double [:,:] Rvec,\
           double [:,:] kvec, double [:,:,:] V_r_keld,\
           double [:,:,:] V_r_coul):
  cdef long kl, num_c, num_v, num_k
  cdef long k, cp, vp, kp, kloc
  cdef double E_cvk
  cdef double complex tmp
  cdef long i,d,j,e,f,m1,m2

  cdef long kinit = klist[0]
  cdef long kend = klist[1]
  num_c = C_nc_k.shape[1]
  num_v = C_nv_k.shape[1]
  num_k = C_nc_k.shape[2]

  kl = kend-kinit
  m1 = myatoms.shape[0]; m2 = Rvec.shape[0]

  cdef double complex [:,:,:,:] arr = np.empty((kl,num_c,num_v,num_k), dtype=complex)
  with cython.nogil:
    for k in range(kinit,kend):
      for cp in range(num_c):
        for vp in range(num_v):
          for kp in range(num_k):
            # Single-particle electronic structure
            # DFT/TB/Wannier90/GW shouldn't matter
            if cp == c and vp == v and kp == k:
              E_cvk = (E_ck[c][k] -\
                       E_vk[v][k])
            else:
              E_cvk = 0.0
            tmp = 0.0 
            for i in range(m1):
              for d in range(myatoms[i,1],myatoms[i,2]):
                for j in range(m1):
                  for e in range(myatoms[j,1],myatoms[j,2]):
                    for f in range(m2):
                      tmp = tmp \
                    + (C_nc_k[d,cp,kp] * C_nv_k[e,v,k] * \
                    function.conj(C_nv_k[e,vp,kp]*C_nc_k[d,c,k])\
                    * V_r_keld[f,i,j] * function.cexp(1j*\
                    ((kvec[k][0]-kvec[kp][0])*Rvec[f][0]\
                    +(kvec[k][1]-kvec[kp][1])*Rvec[f][1]\
                    +(kvec[k][2]-kvec[kp][2])*Rvec[f][2])))\
                    -(function.conj(C_nc_k[d,c,k]*C_nv_k[e,vp,kp])\
                     * C_nv_k[d,v,k] * C_nc_k[e,cp,kp] \
                     * V_r_coul[f,i,j])
            arr[k-kinit][cp][vp][kp] = tmp/m2 + E_cvk
  return arr

@cython.boundscheck(False)
@cython.cdivision(True)
def H_optb(double complex [:,:,:] C_nc_k,\
           double complex [:,:,:] C_nv_k,\
           double [:,:] E_ck, double [:,:] E_vk,\
           long c, long v,\
           long [:,:] myatoms, double [:,:] Rvec,\
           double [:,:] kvec, double [:,:,:] V_r_keld,\
           double [:,:,:] V_r_coul):

  cdef long kl, num_c, num_v, num_k
  cdef long k, cp, vp, kp
  cdef double E_cvk
  cdef double complex tmp
  cdef long i,d,j,e,f,m1,m2
  num_c = C_nc_k.shape[1]
  num_v = C_nv_k.shape[1]
  num_k = C_nc_k.shape[2]
  m1 = myatoms.shape[0]; m2 = Rvec.shape[0]

  cdef double complex [:,:,:,:] arr =\
  np.zeros((num_k,num_c,num_v,num_k), dtype=complex)

  with cython.nogil:
    for k in range(num_k):
      for cp in range(num_c):
        for vp in range(num_v):
          for kp in range(num_k):
            # Single-particle electronic structure
            # DFT/TB/Wannier90/GW shouldn't matter
            if cp == c and vp == v and kp == k:
              E_cvk = (E_ck[c][k] -\
                       E_vk[v][k])
            else:
              E_cvk = 0.0
            tmp = 0.0 + 0j
            for i in range(m1):
              for d in range(myatoms[i,1],myatoms[i,2]):
                for j in range(m1):
                  for e in range(myatoms[j,1],myatoms[j,2]):
                    for f in range(m2):
                      tmp = tmp \
                    + (C_nc_k[d,cp,kp] * C_nv_k[e,v,k] * \
                    function.conj(C_nv_k[e,vp,kp]*C_nc_k[d,c,k])\
                    * V_r_keld[f,i,j] * function.cexp(1j*\
                    ((kvec[k][0]-kvec[kp][0])*Rvec[f][0]\
                    +(kvec[k][1]-kvec[kp][1])*Rvec[f][1]\
                    +(kvec[k][2]-kvec[kp][2])*Rvec[f][2])))\
                    -(function.conj(C_nc_k[d,c,k]*C_nv_k[e,vp,kp])\
                     * C_nv_k[d,v,k] * C_nc_k[e,cp,kp] \
                     * V_r_coul[f,i,j])
            arr[k][cp][vp][kp] = tmp/m2 + E_cvk
  return arr


@cython.boundscheck(False)
@cython.cdivision(True)
def H_optfull(double complex [:,:,:] C_nc_k,\
           double complex [:,:,:] C_nv_k,\
           double [:,:] E_ck, double [:,:] E_vk,\
           long c, long v, long k,\
           long [:,:] myatoms, double [:,:] Rvec,\
           double [:,:] kvec, double [:,:,:] V_r_keld,\
           double [:,:,:] V_r_coul):
  cdef long kl, num_c, num_v, num_k
  cdef long cp, vp, kp, kloc
  cdef double E_cvk
  cdef double complex tmp
  cdef long i,d,j,e,f,m1,m2

  num_c = C_nc_k.shape[1]
  num_v = C_nv_k.shape[1]
  num_k = C_nc_k.shape[2]

  m1 = myatoms.shape[0]; m2 = Rvec.shape[0]

  cdef double complex [:,:,:] arr = np.empty((num_c,num_v,num_k), dtype=complex)
  with cython.nogil:
    for cp in range(num_c):
      for vp in range(num_v):
        for kp in range(num_k):
          # Single-particle electronic structure
          # DFT/TB/Wannier90/GW shouldn't matter
          if cp == c and vp == v and kp == k:
            E_cvk = (E_ck[c][k] -\
                     E_vk[v][k])
          else:
            E_cvk = 0.0
          tmp = 0.0 
          for i in range(m1):
            for d in range(myatoms[i,1],myatoms[i,2]):
              for j in range(m1):
                for e in range(myatoms[j,1],myatoms[j,2]):
                  for f in range(m2):
                    tmp = tmp \
                  + (C_nc_k[d,cp,kp] * C_nv_k[e,v,k] * \
                  function.conj(C_nv_k[e,vp,kp]*C_nc_k[d,c,k])\
                  * V_r_keld[f,i,j] * function.cexp(1j*\
                  ((kvec[k][0]-kvec[kp][0])*Rvec[f][0]\
                  +(kvec[k][1]-kvec[kp][1])*Rvec[f][1]\
                  +(kvec[k][2]-kvec[kp][2])*Rvec[f][2])))\
                  -(function.conj(C_nc_k[d,c,k]*C_nv_k[e,vp,kp])\
                   * C_nv_k[d,v,k] * C_nc_k[e,cp,kp] \
                   * V_r_coul[f,i,j])
          arr[cp][vp][kp] = tmp/m2 + E_cvk
  return arr


@cython.boundscheck(False)
@cython.cdivision(True)
def grad_Hk_gamma( double [:,:] k,\
                 double complex [:,:,:] h_r_ab,\
                 double [:,:] r_ws,\
                 double [:,:] A, double [:,:] B,
                 double complex [:,:,:] grad_h_k_ab):
  """
  Convention II:
    H(k): \sum_{R} h_r_ab*e^{ik.R} (1j*R_x)
  Cythonized part of grad_Hk calculations
  Note that this implements grad_Hk calculations
  as you expect analytically.
  IM: Not sure what to do with Gamma point   
  """
  cdef int i, m, n, j, kmax, a, b, r
  cdef double complex tmp
  cdef int dim

  kmax = k.shape[0]
  a = h_r_ab.shape[1]
  b = h_r_ab.shape[2]
  r = h_r_ab.shape[0]

  dim = 3 
  cdef double [:] r_j = np.zeros((dim), dtype=float)
  cdef double [:] k_i = np.zeros((dim), dtype=float)

  with cython.nogil:
    for i in range(kmax):
      for m in range(a):
        for n in range(b):
          tmp = 0.0
          for j in range(r):
            # Matrix-multiply by hand
            r_j[0] = r_ws[j][0]*A[0][0]+r_ws[j][1]*A[1][0]+\
                     r_ws[j][2]*A[2][0]
            r_j[1] = r_ws[j][0]*A[0][1]+r_ws[j][1]*A[1][1]+\
                     r_ws[j][2]*A[2][1]
            r_j[2] = r_ws[j][0]*A[0][2]+r_ws[j][1]*A[1][2]+\
                     r_ws[j][2]*A[2][2]
            k_i[0] = k[i][0]*B[0][0]+k[i][1]*B[1][0]+\
                     k[i][2]*B[2][0]
            k_i[1] = k[i][0]*B[0][1]+k[i][1]*B[1][1]+\
                     k[i][2]*B[2][1]
            k_i[2] = k[i][0]*B[0][2]+k[i][1]*B[1][2]+\
                     k[i][2]*B[2][2]
            # Only x-components for now
            # \sum_{R} h_r_ab*e^{ik.R} (1j*R_x)
            tmp = tmp + (h_r_ab[j][m][n]* function.cexp(1j*\
                        ((r_j[0]*k_i[0]) + (r_j[1]*k_i[1])+\
                         (r_j[2]*k_i[2])))* 1j*r_j[0])
          grad_h_k_ab[i][m][n] = tmp
  return grad_h_k_ab
