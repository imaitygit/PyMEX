#-----------------------------|
#email: i.maity@imperial.ac.uk|
#author: Indrajit Maity       |
#-----------------------------|

import numpy as np
import sys, scipy, time
from scipy import special
from constants import *
from functools import partial
from wan90tobse import *
from generic_func import *
print_f = partial(print, flush=True)

#=========================|
# Potential ( multilayer )|
#=========================|


class POTENTIAL(object):
  """
  """
  # data attributes
  def __init__(self,wan2bse):
    self.wan2bse = wan2bse
    self.WF_loc = self.wan2bse.get_WF_loc()
    self.A = self.wan2bse.get_lattice()
    self.kgrid = self.wan2bse.get_kgrid()
    self.A_s = unit2super(self.kgrid, self.A)


  def get_Rvec(self):
    """
    Real-space grid/lattice points
    The Lattice points are generated sysmmetrically.
    """
    # Presumes a mxmx1 grid
    if self.kgrid[0]%2 != 0.0:
      m = self.kgrid[0]//2
      Rvec = []
      for i in range(-m, (m+1), 1):
        for j in range(-m, (m+1), 1):
          Rvec.append(i*self.A[0] + j*self.A[1])
      return np.array(Rvec)
    else:
      print_f("The even-grid in real-space is not implemented")
      print_f("Exiting...")
      sys.exit()

  def set_Rvec(self,m):
    """
    Cross-check quality of the Reciprocal-Real space conversion
    by setting the R-sapce yourself.
    """
    # Presumes a mxmx1 grid
    if m%2 != 0.0:
      Rvec = []
      for i in range(-m, (m+1), 1):
        for j in range(-m, (m+1), 1):
          Rvec.append(i*self.A[0] + j*self.A[1])
      return np.array(Rvec)
    else:
      print_f("The even-grid in real-space is not implemented")
      print_f("Exiting...")
      sys.exit()

#  def plot_Rvec(self):
#    """
#    Plot the Real-space grid/lattice points
#    Primarily for crosschecking purposes
#    """
#    Rvec = self.get_Rvec()
#    for i in range(Rvec.shape[0]):
#      plt.scatter(Rvec[i,0], Rvec[i,1], color="k",s=20)
#    plt.show()



  def pair_pbc(self, p1,p2):
    """
    For a given pair of points (in crystal coordinates)
    return separation vector, (p2-p1) and obeys the 
    periodic boundary conditions.
    @input
      p1,p2: Two points in crystal coordinates
    """ 
    d = p2-p1
    p2_un = np.zeros(p2.shape[0], dtype=float)
    for i in range(p2.shape[0]):
      if np.abs(d[i]) >= 0.5:
        if p2[i] >= p1[i]:
          p2_un[i] = p2[i] - 1.0
        else :
          p2_un[i] = p2[i] + 1.0
      else:
        p2_un[i] = p2[i]
    return p2_un-p1


  def r(self,R,n1,n3):
    """
    A vector between two sites (n1,n3) using periodic 
    boundary conditions.
    NOTE: PBC is implemented on the supercell.
    @input
      R: R-th lattice point (one of Rvec)
      n1,n3: WF locations within the unit-cell
    """
    # Two points(n1,n3) within unit-cell at R-th lattice point
    # (in angstroms)
    t_n1 = self.WF_loc[n1] + R
    t_n3 = self.WF_loc[n3]
    # (in-crystal coordinates)
    t_n1_c = ang2crys(self.A_s, t_n1)
    t_n3_c = ang2crys(self.A_s, t_n3)
    # separation vector, rvec in crystal coordinate
    rvec_c = self.pair_pbc(t_n1_c, t_n3_c)
    return np.linalg.norm(crys2ang(self.A_s, rvec_c))


  def get_dist(self, R,t_n1_m,t_n3_m):
    """
    Distance between two points with PBC on the supercell.
    See get_r_full for details of the input
    """
    # Two points(n1,n3) within unit-cell at R-th lattice point
    # (in angstroms)
    tmp1 = t_n1_m + R
    tmp2 = t_n3_m
    # (In crystal coordinates)
    tmp1_c = ang2crys(self.A_s, tmp1)
    tmp2_c = ang2crys(self.A_s, tmp2)
   # separation vector, rvec in crystal coordinate
    rvec_c = self.pair_pbc(tmp1_c, tmp2_c)
    return np.linalg.norm(crys2ang(self.A_s, rvec_c)[:2])
    

  def get_distvec(self, R,t_n1_m,t_n3_m):
    """
    Distance between two points without PBC on the supercell.
    """
    # Two points(n1,n3) within unit-cell at R-th lattice point
    # (in angstroms)
    tmp1 = t_n1_m + R
    tmp2 = t_n3_m
    # (In crystal coordinates)
    tmp1_c = ang2crys(self.A_s, tmp1)
    tmp2_c = ang2crys(self.A_s, tmp2)
   # separation vector, rvec in crystal coordinate (without pbc)
    rvec_c = tmp2_c - tmp1_c
    return crys2ang(self.A_s, rvec_c)


  def get_r_full(self):
    """
    Computes the distance between all the points
    within the moiré supercell. 
    r = R + (tn1_m - tn3_m)
    Note that, n1_m and n3_m represents the mapped indices
    corrsponding to n1,n3 respectively (atom-cenetered).
    NOTE: PBC is implemented on the supercell.
    @input
      R: R-th lattice point (one of Rvec)
      n1,n3: WF locations within the unit-cell
    """
    atoms = self.wan2bse.map_WF()
    Rvec = self.get_Rvec()
    dist = np.empty((Rvec.shape[0],atoms.shape[0],atoms.shape[0]),\
                     dtype=object)
    for i in range(Rvec.shape[0]):
      for n1_m in range(atoms.shape[0]):
        for n3_m in range(atoms.shape[0]):
          dist[i,n1_m,n3_m] = self.get_dist\
          (Rvec[i],atoms[n1_m,3:],atoms[n3_m,3:])
    return dist


  def get_rvec_full(self):
    """
    Computes the distance between all the points
    within the moiré supercell.
    r = R + (tn1_m - tn3_m)
    Note that, n1_m and n3_m represents the mapped indices
    corrsponding to n1,n3 respectively (atom-cenetered).
    NOTE: PBC is not implemented on the supercell.
    @input
      R: R-th lattice point (one of Rvec)
      n1,n3: WF locations within the unit-cell
    """
    atoms = WAN2BSE.map_WF()
    Rvec = self.get_Rvec()
    dist = np.empty((Rvec.shape[0],atoms.shape[0],\
                     atoms.shape[0],3),dtype=object)
    for i in range(Rvec.shape[0]):
      for n1_m in range(atoms.shape[0]):
        for n3_m in range(atoms.shape[0]):
          dist[i,n1_m,n3_m] = self.get_distvec\
          (Rvec[i],atoms[n1_m,3:],atoms[n3_m,3:])
    return dist


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

 
  def keldysh_intralayer_simple_rho(self, rho, rho_0,a0):
    """
    Keldysh potential at a particular point in real space;
    Monolayer TMDs and e-h separated in-plane.
    @input
      rho: in-plane distance between the e-h.
  """

    # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
    # Set up in a way that the potential is in eV.
    f = 22.59

    U = 1.0
    # Setting the on-site term
    # The in-plane lattice constant is used
    # Convention: Phys. Rev. B 91, 075310 (2015).
    if rho == 0.0:
      return -U*f*(self.Struve(self.a0/self.rho_0)-\
                   self.Bessel0(self.a0/self.rho_0))\
                 *1.0/(self.rho_0 * self.epsilon_d)
    else:
      return -f*(self.Struve(rho/self.rho_0)-\
                 self.Bessel0(rho/self.rho_0))\
                *1.0/(self.rho_0 * self.epsilon_d)


  def keldysh_monolayer_simple(self, epsilon_d,r0,a0):
    """
    Keldysh potential at a particular point in real space;
    Monolayer TMDs and e-h separated in a layer.
    @input
      r: distance between the e-h.
  """
    atoms = self.wan2bse.map_WF()
    Rvec = self.get_Rvec()
    Vr_full = np.zeros((Rvec.shape[0],atoms.shape[0],atoms.shape[0]),\
                     dtype=float)
    for i in range(Rvec.shape[0]):
      for n1_m in range(atoms.shape[0]):
        for n3_m in range(atoms.shape[0]):
          r = self.get_dist\
          (Rvec[i],atoms[n1_m,3:],atoms[n3_m,3:])

          # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
          # Set up in a way that the potential is in eV.
          f = 22.59
          U = 1.0
          # Setting the on-site term
          # Convention: Phys. Rev. B 91, 075310 (2015).
          if r == 0.0:
            Vr_full[i,n1_m,n3_m] = -U*f*(self.Struve(a0/r0)-\
                               self.Bessel0(a0/r0))\
                               *1.0/(r0 * epsilon_d)
          else:
            Vr_full[i,n1_m,n3_m] =  -f*(self.Struve(r/r0)-\
                               self.Bessel0(r/r0))\
                               *1.0/(r0 * epsilon_d)
    return Vr_full
   

  def keldysh_bilayer_simple(self, epsilon_d0,r0,a0,\
                                   epsilon_d1,r1,a1):
    """
    Keldysh potential at a particular point in real space;
    Homobilayer/Heterobilayer. 
    intralayer component: e-h stays within a layer.
    interlayer component: e-h are in different layer.
    Long-range interactions treated correctly
    @input
      epsilon_d0,r0,a0: Layer 1 parameters
      epsilon_d1,r1,a1: Layer 2 parameters
    @output
      Returns the Keldysh potential for long-range
      and clearly incorporates bilayer screening
    """
    atoms = self.wan2bse.map_WF()
    d = np.mean(atoms[:,5], dtype=object)
    a_av = (a0+a1)/2.
    Rvec = self.get_Rvec()
    Vr_full = np.zeros((Rvec.shape[0],atoms.shape[0],\
                        atoms.shape[0]),dtype=float)
    
    for i in range(Rvec.shape[0]):
      for n1_m in range(atoms.shape[0]):
        for n3_m in range(atoms.shape[0]):
          r = self.get_dist\
          (Rvec[i],atoms[n1_m,3:],atoms[n3_m,3:])

          # Layer 1 intralayer
          if atoms[n1_m,5] < d and atoms[n3_m,5] < d:
            # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
            # Set up in a way that the potential is in eV.
            f = 22.59; U = 1.0
            # Setting the on-site term
            # Convention: Phys. Rev. B 91, 075310 (2015).
            if r == 0.0:
              Vr_full[i,n1_m,n3_m] = -U*f*(self.Struve(a0/(r0+r1))-\
                                   self.Bessel0(a0/(r0+r1)))\
                                   *1.0/((r0+r1) * epsilon_d0)
            else:
              Vr_full[i,n1_m,n3_m] =  -f*(self.Struve(r/(r0+r1))-\
                                   self.Bessel0(r/(r0+r1)))\
                                   *1.0/((r0+r1) * epsilon_d0)

          # Layer 2 intralayer
          elif atoms[n1_m,5]>d and atoms[n3_m,5] >d:
            # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
            # Set up in a way that the potential is in eV.
            f = 22.59; U = 1.0
            # Setting the on-site term
            # Convention: Phys. Rev. B 91, 075310 (2015).
            if r == 0.0:
              Vr_full[i,n1_m,n3_m] = -U*f*(self.Struve(a1/(r0+r1))-\
                                   self.Bessel0(a1/(r0+r1)))\
                                   *1.0/((r0+r1) * epsilon_d0)
            else:
              Vr_full[i,n1_m,n3_m] =  -f*(self.Struve(r/(r0+r1))-\
                                   self.Bessel0(r/(r0+r1)))\
                                   *1.0/((r0+r1) * epsilon_d0)
          # Layer 1-2: Interlayer
          else:
            # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
            # Set up in a way that the potential is in eV.
            f = 22.59; U = 1.0; ils=7.0; 
            # Setting the on-site term
            # Convention: Phys. Rev. B 91, 075310 (2015).
            if r == 0.0:
              Vr_full[i,n1_m,n3_m] = -U*f*(self.Struve(a_av/(r0+r1+ils))-\
                                   self.Bessel0(a_av/(r0+r1+ils)))\
                                   *1.0/((r0+r1+ils) * epsilon_d0)
            else:
              Vr_full[i,n1_m,n3_m] =  -f*(self.Struve(r/(r0+r1+ils))-\
                                   self.Bessel0(r/(r0+r1+ils)))\
                                  *1.0/((r0+r1+ils) * epsilon_d0)
    return Vr_full


  def coulomb_simple(self, epsilon_d,r0,a0):
    """
    Unscreened Coulomb potential at a particular point in 
    real space;
    Monolayer TMDs and e-h separated in a layer.
    @input
      r: distance between the e-h.
  """
    atoms = self.wan2bse.map_WF()
    Rvec = self.get_Rvec()
    Vr_full = np.zeros((Rvec.shape[0],atoms.shape[0],atoms.shape[0]),\
                     dtype=float)
    for i in range(Rvec.shape[0]):
      for n1_m in range(atoms.shape[0]):
        for n3_m in range(atoms.shape[0]):
          r = self.get_dist\
          (Rvec[i],atoms[n1_m,3:],atoms[n3_m,3:])

          # f=(pi/2)*(e^2/4\pi\epsilon_0) is the prefactor
          # Set up in a way that the potential is in eV.
          f = 22.59
          U = 1.0
          # Setting the on-site term
          # Convention: In same spirit as Keldysh !Test!
          if r == 0.0:
            Vr_full[i,n1_m,n3_m] = -U*f*(1/a0)\
                               *1.0/(epsilon_d)
          else:
            Vr_full[i,n1_m,n3_m] =  -f*(1/r)\
                               *1.0/(epsilon_d)
    return Vr_full
   
   

### Following are for testing purposes; Always comment them out
##WAN2BSE = WAN2BSE("hetero.win", "hetero_u.mat", "hetero_hr.dat",\
##                  "hetero_wsvec.dat", "hetero.wout")
#WAN2BSE = WAN2BSE("WS2.win", "WS2_u.mat", "WS2_hr.dat",\
#                  "WS2_wsvec.dat", "WS2.wout")
#POTENTIAL = POTENTIAL(WAN2BSE)
#
## Tests
## Plotting the V_r
#import matplotlib.pyplot as plt
#
#rvec_full = POTENTIAL.get_rvec_full()
#rvec_full = rvec_full.reshape((rvec_full.shape[0]*\
#            rvec_full.shape[1]*rvec_full.shape[2],3))
#Vr_keld = POTENTIAL.keldysh_monolayer_simple(1.0,33,3.18)
#Vr_keld = Vr_keld.reshape((Vr_keld.shape[0]*\
#            Vr_keld.shape[1]*Vr_keld.shape[2]))
#Vr_coul = POTENTIAL.coulomb_monolayer_simple(1.0,33,3.18)
#Vr_coul = Vr_coul.reshape((Vr_coul.shape[0]*\
#            Vr_coul.shape[1]*Vr_coul.shape[2]))
#
#plt.subplot(211)
#R_sup = np.array([[1.0, 0.0],\
#                 [0.5, 0.8660254]])*9.96*3
#for i in range(-1, 2):
#  for j in range(-1, 2):
#    tmp = rvec_full[:,:2] + i*R_sup[0] + j*R_sup[1]
#    plt.scatter(tmp[:,0], tmp[:,1], c=Vr_keld, s=20)
#plt.colorbar()
#
##plt.subplot(212)
##plt.scatter(rvec_full[:,0], rvec_full[:,1], c=Vr_coul, s=20,\
##            label="Coulomb")
##plt.colorbar()
#plt.legend()
#plt.show()
