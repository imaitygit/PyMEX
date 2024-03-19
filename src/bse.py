#-----------------------------|
#email: i.maity@imperial.ac.uk|
#Author: Indrajit Maity       |
#-----------------------------|


import numpy as np
import os, sys
from wan90tobse import *
import time
from read_inp import *
from potential import *
from dft2bse import *
from functools import partial
from addsoc import * 
print_f = partial(print, flush=True)
import h5py
from cyfunc import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0



class BSE(object):
  """
  Construct the BSE matrix, diagonalize, and post-process
  """

  def __init__(self, inp_file):
    """
    Initializes the attributes
    """
    t = time.time()

    # Read and print input 
    self.inp_file = inp_file 
    self.calc, self.nature, self.dft, self.coeff, self.ham,\
    self.struc, self.interact, self.wout, self.kg, self.absorp,\
    self.ephparam, self.parallel = print_inp(self.inp_file)

    # Basic checks
    if rank == root:
      print_f(f"\n")
      print_f(60*"+")
      print_f("Start of Warning|Error messages".center(60))
      print_f()
      self.check_all()
      print_f()
      print_f("End of Warning|Error messages".center(60))
      print_f(60*"+")

    # ------- OUTPUT --------
    if rank == root:
      print_f()
      print_f("~~~Passed the preliminary checks~~~".center(60))
      print_f()
      print_f(24*"*"+"OUTPUT SECTION"+24*"*")
      print_f()

    # Stuffs from WAN90
    self.wan2bse = WAN2BSE(self.struc[1],self.coeff[0,1],\
                           self.ham[0,1],self.ham[0,2],\
                           self.wout[0,1])
    self.kvec = self.wan2bse.get_kpoints()
    self.atoms = self.wan2bse.map_WF()
    self.A = self.wan2bse.get_lattice()

    # Stuffs for Potential
    self.potential = POTENTIAL(self.wan2bse) 
    self.Rvec = self.potential.get_Rvec()
    if "monolayer" in self.interact[1].casefold():
      potname, epsilon_d0, r0, a0 = self.setup_potential()
    elif "bilayer" in self.interact[1].casefold():
      potname, epsilon_d0, r0, a0, epsilon_d1, r1, a1 \
                                  = self.setup_potential()
    # Initialise to zeros
    self.V_r_keld = np.zeros((self.Rvec.shape[0],\
                              self.atoms.shape[0],\
                              self.atoms.shape[0]),\
                              dtype=float)
    self.V_r_coul = np.zeros((self.V_r_keld.shape))

    # Direct: True
    if eval(self.interact[3]):
      if "keldysh_monolayer_simple" in potname:
        self.V_r_keld = self.potential.\
                   keldysh_monolayer_simple(epsilon_d0,r0,a0)
      elif "keldysh_bilayer_simple" in potname:
        self.V_r_keld = self.potential.\
                   keldysh_bilayer_simple(epsilon_d0,r0,a0,\
                                          epsilon_d1,r1,a1)
    # Exchange: True
    if eval(self.interact[4]):
      self.V_r_coul = self.potential.\
                      coulomb_simple(epsilon_d0,r0,a0)

    if rank == root:      
      print_f("El-Hole potential in real-space is done.")
      print_f("Time spent so far: %.4f secs."\
              %(time.time()-t))

    # Get Coefficients
    # IM: **Slows down when lots of proccess
    # read at the same time.** FIX ME
    if self.nature == "full":
      C_nm_k = self.wan2bse.get_C()
      for i in range(self.coeff.shape[0]):
        for j in range(self.coeff.shape[1]):
          if isinstance(self.coeff[i,j],str):
            if "cb" in self.coeff[i,j].casefold():
              self.C_nc_k = self.c_nc_k(C_nm_k,
                      int(self.coeff[i,2])+int(self.coeff[i,4]),\
                      int(self.coeff[i,3])+int(self.coeff[i,4]))
            elif "vb" in self.coeff[i,j].casefold():
              self.C_nv_k = self.c_nv_k(C_nm_k,
                      int(self.coeff[i,2])+int(self.coeff[i,4]),\
                      int(self.coeff[i,3])+int(self.coeff[i,4]))
      # Free C_nm_k as they are kept in coeffs.
      del C_nm_k
    else:
      if rank == root:
        print_f("Isolated style is not supported yet")
        print_f("Exiting...")
      comm.Abort(1)
    if rank == root:
      print_f("C_nm_k for valence and conduction bands done.")
      print_f("Time spent so far: %.4f secs."%(time.time()-t))

    # Single-particle energies
    self.E_ck = self.get_E_ck()
    self.E_vk = self.get_E_vk()
    if rank == root:
      print_f("Single-particle eigenvalues are receieved")
      print_f("Time spent so far: %.4f secs."%(time.time()-t))


  def check_all(self):
    """
    Preparatory safety checks
    """
    # Check 1 for "BSE:"
    errorcheck_calc(self.calc)
    # Check 2 for "Layer:"
    errorcheck_nature(self.nature)
    # Check 3 for "DFT:"/"Coefficients"/Hamiltonian
    errorcheck_dft(self.nature, self.dft)
    errorcheck_dft(self.nature, self.coeff)
    errorcheck_ham(self.nature, self.ham)
    # Check 4 for direct and exchnage terms
    errorcheck_pot(self.nature, self.interact)
    # Check 5 for k-grids
    errorcheck_kgrid(self.kg)
    # Check 6 for parallelization schemes
    errorcheck_parallel(self.parallel)


  def gaussian(self, E, Eex, sigma):
    """
    Gaussian function as a representation of the delta function
    @input
      E: Energy that's a variable
      Eex: Exciton discrete values
      sigma: Half-width at half-maximum
    @output
      probability function
    """
    return np.exp((-(E-Eex)**2.)/(2*sigma**2))*\
           1/(sigma*np.sqrt(2*np.pi))


  def sigma_xx(self):
    """
    To DO:
          Tidy this up.
          Currently, all the options are written in a it incoherent
          manner.
    """
    # Synchronize
    comm.Barrier()

    # Load the BSE eigenvalues
    f1 = h5py.File("Eigval.hdf5", 'r')
    eigval = np.array(f1['eigval'])
    f1.close()

    # Load the BSE eigenvectors
    f2 = h5py.File("Eigvec.hdf5", 'r')
    eigvec = np.array(f2['eigvec'])
    f2.close()

    # Eigenvector in the right-format
    eigvnew = np.zeros((len(eigval),self.C_nc_k.shape[1],\
                      self.C_nv_k.shape[1],
                      self.C_nc_k.shape[2]),\
                      dtype = complex)
    for i in range(len(eigval)):
      eigvnew[i] = eigvec[:,i].reshape((self.C_nc_k.shape[1],\
                      self.C_nv_k.shape[1],\
                      self.C_nc_k.shape[2]))

    t1 = time.time() 
    self.gradx_Hk = self.wan2bse.get_grad_Hk()
    self.Hk = self.wan2bse.get_Hk()
    pos = self.wan2bse.get_WF_loc()
    #print_f("pos_x", pos[:,0])

    if rank == root:
      print_f("Time taken for Grad_Hk:%.4f secs"%(\
               time.time()-t1))

    # Minimum and maximum window for conductivity calculations
    # All input parameters are in eV.
    xmin = np.min(eigval)
    E_ph = np.arange(xmin+self.ephparam[0], xmin+self.ephparam[1],\
                     self.ephparam[2])
    sigma = self.ephparam[3]
 
    # Photon enrgies to a file
    if rank == root: 
      g = h5py.File("E_ph.hdf5", "w", libver='latest')
      dset = g.create_dataset("photon",\
               ((E_ph.shape[0])),\
                 dtype='float')
      dset[:] = E_ph
      g.close()

    if rank == root:
      print_f("Parallelizing over photon enrgies")
    # Get local E_ph points for the ranks
    Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
    print_f("%d-th rank handles %d photon enrgies"%\
           (rank,(Elist[1]-Elist[0])))

    # Synchronize
    comm.Barrier()

    t2 = time.time()
    # Calculation runs without modifying exciton
    # eigenvalues; See the perturbative inclusion 
    # of spin-orbit-coupling below. 

    if self.absorp[1].casefold() == "full":
      # HDF5 in parallel
      f = h5py.File("SIGMA_full.hdf5", "w", libver='latest',\
                    driver='mpio',comm=comm)
      dset = f.create_dataset("sigma_xx",\
             ((E_ph.shape[0],)),\
             dtype='float')
      # Computation of \Sigma_xx
      # Cythonized version
      # Every rank computes conductivity
      # based on the photon energies it handles
      tmp_dim = Elist[1]-Elist[0]
      tmp = np.zeros((tmp_dim), dtype=float)
      t3 = time.time()
      for i in range(Elist[0], Elist[1]):

        # Modified - 13/11/2023.
        if self.parallel[1].casefold() == "thread":
          #print_f("positions x coord", pos[:,0])
          #print_f("sigma", sigma)
          #print_f("E_ph[i]", E_ph[i])
          tmp[i-Elist[0]] = sigma_xx_full_E_thread(eigval,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real
        else:
          tmp[i-Elist[0]] = sigma_xx_full_E(eigval,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real

      t4 = time.time()
      if rank == root:
        print_f("Calculations done. Will collect")
        print_f("Time for sigma calculations:%.4f secs."%(t4-t3))

      with dset.collective:
        # Collects from every rank
        dset[Elist[0]:Elist[1]] = tmp
      f.close()
      if rank == root:
        print_f("Conductivity data written to HDF5")
        print_f("Time for sigma-parallelization:%.4f secs"\
                 %(time.time()-t2))
      # Synchronize
      comm.Barrier()

    # Calculation runs with the modification of BSE 
    # eigenvalues in order to capture the SOC.
    # Based on: Phys. Rev. Lett. 111, 216805 (2013).
    elif self.absorp[1].casefold() == "perturbation":
      # Spin-Orbit-Coupling (SOC) class
      # -- SOC --
      # Look at the ordering
      if rank == root:
        print_f()
        print_f("----------------------------")
        print_f("Adding SOC as a perturbation")
        print_f("----------------------------")
      self.soc = SOC(self.dft[0,0], self.absorp[5],\
                 self.absorp[6],self.absorp[2],self.absorp[3],\
                 self.absorp[4])

      # Unfold the cvk for moiré to unit-cell c_uv_uk_u
      if self.absorp[2] == "True":
        if rank == root:
          print_f()
          print_f("Ideally requires unfolding")
          print_f("However, not implemented yet")
        comm.Abort(1)

      # Unit-cell calculations
      elif self.absorp[2]== "False":
        if rank == root:
          print_f("Don't require unfolding")
          print_f("Unit-cell calculations")
        # Get Delta_nsk for all the bands 
        self.soc.get_Delta_nsk_unit()
        # PLOT
        #import matplotlib.pyplot as plt
        #for i in range(Delta_cvsk.shape[2]):
        #  plt.plot(Delta_cvsk[1,1,i,:])
        #plt.show()
        # Compute \Delta_{vcsk} (see function)
        Delta_cvsk = self.get_Delta_cvsk_unit()
        eigval_s = np.zeros((eigval.shape[0],2))
        eigval_ = Eigval_perturb(eigval,eigvnew,\
                   Delta_cvsk, eigval_s)
        eigval_s[:,:] = eigval_
        # Minimum and maximum window for conductivity calculations
        # All input parameters are in eV.
        xmin = np.min(eigval_s)
        E_ph = np.arange(xmin+self.ephparam[0], xmin+self.ephparam[1],\
                         self.ephparam[2])
        sigma = self.ephparam[3]
        # Photon enrgies to a file
        if rank == root:
          g = h5py.File("E_ph_per.hdf5", "w", libver='latest')
          dset = g.create_dataset("photon",\
                 ((E_ph.shape[0])),\
                   dtype='float')
          dset[:] = E_ph
          g.close()

        if rank == root:
          print_f("Parallelizing over photon enrgies")
        # Get local E_ph points for the ranks
        Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
        print_f("%d-th rank handles %d photon enrgies"%\
               (rank,(Elist[1]-Elist[0])))

        # Synchronize
        comm.Barrier()

        # HDF5 in parallel
        f = h5py.File("SIGMA_per_unit.hdf5", "w", libver='latest',\
                      driver='mpio',comm=comm)
        dset = f.create_dataset("sigma_xx",\
               ((E_ph.shape[0],)),\
               dtype='float')
        # IM : Modified - 13/11/2023
        # Computation of \Sigma_xx
        # Cythonized version
        # Every rank computes conductivity
        # based on the photon energies it handles
        tmp_dim = Elist[1]-Elist[0]
        tmp = np.zeros((tmp_dim), dtype=float)
        for i in range(Elist[0], Elist[1]):
          if self.parallel[1].casefold() == "thread":
            tmp[i-Elist[0]] = sigma_xx_per_E_thread(eigval_s,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real
          else:
            tmp[i-Elist[0]] = sigma_xx_per_E(eigval_s,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real
        if rank == root:
          print_f("Calculations done. Will collect")
        # Delete un-necessary things
        del eigval_s; del eigvnew

        with dset.collective:
          # Collects from every rank
          dset[Elist[0]:Elist[1]] = tmp
        f.close()
        if rank == root:
          print_f("Conductivity data written to HDF5")
          print_f("Time for sigma-parallelization:%.4f secs"\
                   %(time.time()-t2))
        # Synchronize
        comm.Barrier()     

      # Based on unit-cell calculations
      # PMU approach
      elif self.absorp[2].casefold() == "pmu":
        if rank == root:
          print_f()
          print_f("WARNING: Moiré requires unfolding, ideally")
          print_f("Will be using a simplified version :D")
          print_f("Poor Man's Unfolding (PMU)")
          print_f()
        # 
        # Compute \Delta_{vcsk} (see function)
        Delta_cvsk = self.get_Delta_cvsk_pmu()
        eigval_s = np.zeros((eigval.shape[0],2))
        eigval_ = Eigval_perturb(eigval,eigvnew,\
                   Delta_cvsk, eigval_s)
        eigval_s[:,:] = eigval_
        # Minimum and maximum window for conductivity calculations
        # All input parameters are in eV.
        xmin = np.min(eigval_s)
        E_ph = np.arange(xmin+self.ephparam[0], xmin+self.ephparam[1],\
                         self.ephparam[2])
        sigma = self.ephparam[3]

        # Photon enrgies to a file
        if rank == root:
          g = h5py.File("E_ph_per_pmu.hdf5", "w", libver='latest')
          dset = g.create_dataset("photon",\
                 ((E_ph.shape[0])),\
                   dtype='float')
          dset[:] = E_ph
          g.close()

        if rank == root:
          print_f("Parallelizing over photon enrgies")
        # Get local E_ph points for the ranks
        Elist = self.distribute_E_ph(E_ph.shape[0],rank,size)
        print_f("%d-th rank handles %d photon enrgies"%\
               (rank,(Elist[1]-Elist[0])))

        # Synchronize
        comm.Barrier()

        # HDF5 in parallel
        f = h5py.File("SIGMA_per_pmu.hdf5", "w", libver='latest',\
                      driver='mpio',comm=comm)
        dset = f.create_dataset("sigma_xx",\
               ((E_ph.shape[0],)),\
               dtype='float')
        # Computation of \Sigma_xx
        # Cythonized version
        # Every rank computes conductivity
        # based on the photon energies it handles
        tmp_dim = Elist[1]-Elist[0]
        tmp = np.zeros((tmp_dim), dtype=float)
        for i in range(Elist[0], Elist[1]):
          if self.parallel[1].casefold() == "thread":
            tmp[i-Elist[0]] = sigma_xx_per_E_thread(eigval_s,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real
          else:
            tmp[i-Elist[0]] = sigma_xx_per_E(eigval_s,eigvnew,\
                            self.C_nc_k,self.C_nv_k, self.gradx_Hk,\
                            self.Hk, pos[:,0],\
                            E_ph[i],sigma).real
        if rank == root:
          print_f("Calculations done. Will collect")
        # Delete un-necessary things
        del eigval_s; del eigvnew

        with dset.collective:
          # Collects from every rank
          dset[Elist[0]:Elist[1]] = tmp
        f.close()
        if rank == root:
          print_f("Conductivity data written to HDF5")
          print_f("Time for sigma-parallelization:%.4f secs"\
                   %(time.time()-t2))
        # Synchronize
        comm.Barrier()     
      else:
        if rank == root:
          print_f("Unknwon keyword found in Absorption type")
          print_f("Exiting...")
        comm.Abort(1)  


  def get_Delta_cvsk_unit(self):
    """
    \Delta_{cvsk} = (E_ck - E_vk) +  \\ at the DFT and non-polar\
                    (\delta_csk - delta_vsk) + \\ SOC contrib.\
                    (rigid shift for GW at K-point)
                    s = 2
    NOTE: Rigid-shift is likely going to fail in general cases
          But for now, we will *not* use it.
          The c,v,k are the same as for DFT single-particle input.
          \delta_csk/\delta_vsk read from "DELTA.HDF5" file.
          Also, E_ck -E_vk is not included in this part.
    """
    #-----
    spin = 2
    shift = 0.0 # eV
    #-----
    Delta_cvsk = np.zeros((self.E_ck.shape[0], self.E_vk.shape[0],\
                           spin, self.E_ck.shape[1]), dtype=float)

    for c in range(self.E_ck.shape[0]):
      for v in range(self.E_vk.shape[0]):
        for s in range(spin):
          for k in range(self.E_ck.shape[1]):
            Delta_cvsk[c,v,s,k] = \
          (self.get_Delta_csk_unit()[c,s,k]-\
           self.get_Delta_vsk_unit()[v,s,k])+\
           shift 
          #+\
          #(self.E_ck[c,k] - self.E_vk[v,k])
    return Delta_cvsk


  def get_Delta_cvsk_pmu(self):
    """
    \Delta_{cvsk} = (E_ck - E_vk) +  \\ at the DFT and non-polar\
                    (\delta_csk - delta_vsk) + \\ SOC contrib.\
                    (rigid shift for GW at K-point)
                    s = 2
    NOTE: Rigid-shift is likely going to fail in general cases
          But for now, we will *not* use it.
          The c,v,k are the same as for DFT single-particle input.
          \delta_csk/\delta_vsk read from "DELTA.HDF5" file.
          Also, E_ck -E_vk is not included in this part.
    """
    #-----
    spin = 2
    shift = 0.0 # eV
    #-----
    Delta_cvsk = np.zeros((self.E_ck.shape[0], self.E_vk.shape[0],\
                           spin, self.E_ck.shape[1]), dtype=float)
    for c in range(self.E_ck.shape[0]):
      for v in range(self.E_vk.shape[0]):
        for s in range(spin):
          for k in range(self.E_ck.shape[1]):
            Delta_cvsk[c,v,s,k] = \
          (self.get_Delta_csk_pmu()[c,s,k]-\
           self.get_Delta_vsk_pmu()[v,s,k])+\
           shift 
          #+\
          #(self.E_ck[c,k] - self.E_vk[v,k])
    return Delta_cvsk



  def get_eigval_perturb_unit(self):
    """
    Omega_Ms = Omega_M + D_vcsk;
               D_vcsk = \sum_{vck} |A^M_{vcsk}|^{2} \Delta_{vcks}
               and Omega_M are the BSE eigenvalues with non-polar.
               DFT calculations. 
    NOTE: The valence and conduction band assignments are the same
          as with DFT single-particle eigenvalues. We don't use the
          WANNIER90 eigenvalues at all throughout. 
    Ref: Qiu et al., Phys. Rev. Lett. 111, 216805 (2013).
    """
     

  def factor(self):
    """
    """
    return 1.0


  def write_Ham_parallel(self, savefile="H_eh.hdf5"):
    """
    Save a file with HDF5 and mpi i/o
    NOTE: The inner loop is fully cythonized.
          ** Can be buggy, also try better algorithm **
    """

    if rank == root:
      print_f("Constructing the BSE Hamiltonian")
      print_f("Please wait...")
      print_f()

    # HDF5 in-parallel
    f = h5py.File(savefile, "w", driver='mpio', comm=comm)
    # For large systems play with the compression_opts
    dset = f.create_dataset("bse_ham",\
           ((self.C_nc_k.shape[1],self.C_nv_k.shape[1],\
             self.C_nc_k.shape[2],self.C_nc_k.shape[1],\
             self.C_nv_k.shape[1],self.C_nc_k.shape[2])),\
             dtype='complex')
             #dtype='complex',compression='gzip',\
             #compression_opts=4)

    if rank == root:
      print_f("Parallelization Information")
      print_f()

    if self.parallel[0] == "kpoints":
      if rank == root:
        print_f("Choosing parallelization over-kpoints")
      # Get local-kpoints for the ranks
      klist = self.distribute_k(self.C_nc_k.shape[2],rank,size)
      print_f("%d-th rank handles %d kpoints"%\
            (rank,(klist[1]-klist[0])))

      # Synchronize
      comm.Barrier()
      t1 = time.time()
      myatoms = self.atoms.astype(int)
      for c in range(self.C_nc_k.shape[1]):
        for v in range(self.C_nv_k.shape[1]):
          #----------------------------
          # Hbrid version: python+cython
          #----------------------------
          tmp = H_optk(self.C_nc_k, self.C_nv_k,\
                  self.E_ck, self.E_vk, c,v,klist,\
                  myatoms, self.Rvec, self.kvec,\
                  self.V_r_keld, self.V_r_coul)
          if rank == root:
            print_f("Calculations done. Will collect")
          with dset.collective:
            # Collects from every rank
            dset[c,v,klist[0]:klist[1],:,:,:] = tmp
#            #----------------------
#            # Pure python version
#            #----------------------
#            #dset[c,v,klist[0]:klist[1],:,:,:] \
#            #         = self.H_k(c,v,klist)
          if rank == root:
            print_f("Written to HDF5")
            print_f("Upto c=%d, v=%d"%(c,v))
            print_f("Time for k-parallelization:%.4f secs"\
                    %(time.time()-t1))
          # Synchronize again
          comm.Barrier()

    elif self.parallel[0] == "bands":
    # Parallelize over bands
      if rank == root:
        print_f("Choosing parallelization over-bands")
      # Distribute bands for processes
      blist = self.distribute_band(self.C_nc_k.shape[1]*\
                      self.C_nv_k.shape[1],rank,size)
      # "Vectorizes" the number of bands to compute
      # and distributes them suitably. Scales almost
      # linearly as there's no interaction.
      myelem = []
      for i in range(self.C_nc_k.shape[1]):
        for j in range(self.C_nv_k.shape[1]):
          myelem.append([i,j])
      myelem = np.array(myelem)
      print_f("%d-th rank handles %d bands"%\
               (rank,(blist[1]-blist[0])))
      # Synchronize
      comm.Barrier()
      t1 = time.time()
      myatoms = self.atoms.astype(int)
      for m in range(blist[1]-blist[0]):
        c = myelem[blist[0]+m][0]
        v = myelem[blist[0]+m][1]
        #------------------------------
        # Hybrid version: python+cython
        #----------------------------
        tmp = H_optb(self.C_nc_k, self.C_nv_k,\
                  self.E_ck, self.E_vk, c,v,\
                  myatoms, self.Rvec,self.kvec,\
                  self.V_r_keld, self.V_r_coul)
        if rank == root:
          print_f("Calculations done. Will collect")

        # HDF5 in-parallel
        with dset.collective:
          dset[c,v,:,:,:,:] = tmp
          if rank == root:
            print_f("Written to HDF5")
            print_f("Upto c=%d, v=%d"%(c,v))
            print_f("Time for band-parallelization:%.4f secs"\
                    %(time.time()-t1))
        # Synchronize
        comm.Barrier()


    elif self.parallel[0] == "full":
      # Parallelize over bands and kpoints
      if rank == root:
        print_f("Choosing parallelization over-bands and kpoints")
      # Distribute points for processes
      plist = self.distribute_full(self.C_nc_k.shape[1]*\
                      self.C_nv_k.shape[1]*self.C_nc_k.shape[2],\
                      rank,size)
      # "Vectorizes" the number of points to compute
      # and distributes them suitably. Scales almost
      # linearly as there's no interaction.
      myelem = []
      for i in range(self.C_nc_k.shape[1]):
        for j in range(self.C_nv_k.shape[1]):
          for k in range(self.C_nc_k.shape[2]):
            myelem.append([i,j,k])
      print_f("%d-th rank handles %d items"%\
               (rank,(plist[1]-plist[0])))
      # Synchronize
      comm.Barrier()
      t1 = time.time()
      myatoms = self.atoms.astype(int)
      for m in range(plist[1]-plist[0]):
        c = myelem[plist[0]+m][0]
        v = myelem[plist[0]+m][1]
        k = myelem[plist[0]+m][2]
        #------------------------------
        # Hybrid version: python+cython
        #----------------------------
        if self.parallel[1].casefold() == "thread":
          tmp = H_optfull_thread(self.C_nc_k, self.C_nv_k,\
                  self.E_ck, self.E_vk, c,v,k,\
                  myatoms, self.Rvec, self.kvec,\
                  self.V_r_keld, self.V_r_coul)
        else:
          tmp = H_optfull(self.C_nc_k, self.C_nv_k,\
                  self.E_ck, self.E_vk, c,v,k,\
                  myatoms, self.Rvec, self.kvec,\
                  self.V_r_keld, self.V_r_coul)
        if rank == root:
          print_f("Calculations done. Will collect")

        # HDF5 in-parallel
        with dset.collective:
          dset[c,v,k,:,:,:] = tmp
          if rank == root:
            print_f("Written to HDF5")
            print_f("Upto c=%d, v=%d"%(c,v))
            print_f("Time for band-parallelization:%.4f secs"\
                    %(time.time()-t1))
        # Synchronize again
        comm.Barrier()

    # Close the HDF5 file
    f.close()


  def diagon_BSE(self, hamfile="H_eh.hdf5"):
    """
    Loads the BSE hamiltonian and diagonalizes with LAPACK
    Should work fine for small systems. 
    Since we don't really need huge matrices to diagonalize
    we can safely use LAPACK for now.
    To DO: Scalapack for FUTURE
    """
    t = time.time()
    if rank == root:
      print_f("Setting up the BSE Hamiltonian")
    # One-process
    if rank == root:
      f = h5py.File(hamfile, 'r')
      # my dataset
      H_eh = np.array(f['bse_ham'])
      re = self.C_nc_k.shape[1]*self.C_nv_k.shape[1]*\
           self.C_nc_k.shape[2]
      H_eh = H_eh.reshape((re,re))
      f.close()
 
      # Check if the matrix is hermitian
      self.is_hermitian(H_eh)
      eigval, eigvec = np.linalg.eigh(H_eh)
      print_f("Diagonalization of the BSE Hamiltonian done")
      print_f("First 16 eignevalues", eigval[:16])

      # Store eigenvalues
      g = h5py.File("Eigval.hdf5", "w")
      dset = g.create_dataset("eigval",\
             ((re,)),\
               dtype='float',compression='gzip',\
               compression_opts=4)
      dset[:] = eigval
      g.close() 

      # Store eigenvectors
      h = h5py.File("Eigvec.hdf5", "w")
      dset = h.create_dataset("eigvec",\
             ((re,re)),\
               dtype='complex',compression='gzip',\
               compression_opts=4)
      dset[:,:] = eigvec
      h.close()
      print_f("Time taken to diagonalize the BSE:%.3f secs"\
              %(time.time()-t))


  def H_b(self,c,v):
    """
    Construct the Bethe-Salpeter-Hamiltonian 
    for a given c,v using the Direct
    term.
    """
    tmp = np.zeros((self.C_nc_k.shape[2], self.C_nc_k.shape[1],\
                    self.C_nv_k.shape[1],self.C_nc_k.shape[2]),\
                    dtype=complex)
    for k in range(self.C_nc_k.shape[2]):
      if rank == root:
        if k%2 == 0:
          print_f("Completed %d points for c=%d,v=%d"%(k,c,v))
      for cp in range(self.C_nc_k.shape[1]):
        for vp in range(self.C_nv_k.shape[1]):
          for kp in range(self.C_nc_k.shape[2]):
            # Single-particle electronic structure
            # DFT/TB/Wannier90/GW shouldn't matter
            if cp == c and vp == v and kp == k:
              E_cvk = (self.E_ck[c][k] - self.E_vk[v][k])
            else:
              E_cvk = 0.0
            # Pure python version
            tmp[k][cp][vp][kp] =\
              self.D_cvk_cpvpkp(c,v,k,cp,vp,kp) + E_cvk      
              # Old hybrid version
              #self.D__(c,v,k,cp,vp,kp) + E_cvk       
    return tmp


  def H_k(self,c,v,klist):
    """
    Construct the Bethe-Salpeter-Hamiltonian 
    for a given set of k-points using the Direct
    term.
    """
    tmp = np.zeros((klist[1]-klist[0], self.C_nc_k.shape[1],\
                    self.C_nv_k.shape[1],self.C_nc_k.shape[2]),\
                    dtype=complex)
    for k in range(klist[0],klist[1]):
      for cp in range(self.C_nc_k.shape[1]):
        for vp in range(self.C_nv_k.shape[1]):
          for kp in range(self.C_nc_k.shape[2]):
            # Single-particle electronic structure
            # DFT/TB/Wannier90/GW shouldn't matter
            if cp == c and vp == v and kp == k:
              E_cvk = (self.E_ck[c][k] - self.E_vk[v][k])
            else:
              E_cvk = 0.0
            # Pure python version
            tmp[k-klist[0]][cp][vp][kp] =\
              self.D_cvk_cpvpkp(c,v,k,cp,vp,kp) + E_cvk
            if kp%100 == 0:
              if rank == root:
                print_f("Completed %d kprime_points for cp=%d,vp=%d"%(kp,cp,vp))
              # old hybrid version-
              #self.D__(c,v,k,cp,vp,kp) + E_cvk      
    return tmp


  def distribute_band(self, num_b, rank, size):
    """
   *Variable* block cyclic distribution to take
    care of the load-balancing and maintain easy
    data access; This is intended for nested loop
    optimization. 
    NOTE: Only works when number of bands is integer divisible 
          by number of mpi processes. This is temporary needs work.
    """
    if num_b%size != 0:
      if rank == root:
        print_f("Bad Parallelization!")
        print_f("Number of bands need to be divisible by mpi proc.")
        print_f("Exiting...")
      comm.Abort(1)
    # Easy integer division
    b_own = num_b // size
    # Check if there's a remainder
    b_rem = num_b % size
    # Starting point for each process as long
    # as the rank < size-k_rem
    b_init = rank*b_own
    # These ranks will handle 1 extra band
    if rank >= (size - b_rem):
      b_init = (rank*b_own) + (rank-(size-b_rem))
      b_own = b_own+1
      k_end   = b_init+b_own
    b_end   = b_init+b_own
    return np.array([b_init,b_end])



  def distribute_k(self, num_k, rank, size):
    """
    *Variable* block cyclic distribution to take
    care of the load-balancing and maintain easy
    data access;
    """
    # Warning
    if num_k < size:
      if rank == root:
        print_f("Bad Parallelization! Some processes will be idle")
        print_f("Exiting...")
      comm.Abort(1)
    # Easy integer division
    k_own = num_k // size
    # Check if there's a remainder
    k_rem = num_k % size
    # Starting point for each process as long
    # as the rank < size-k_rem
    k_init = rank*k_own
    # These ranks will handle 1 extra k-points
    if rank >= (size - k_rem):
      k_init = (rank*k_own) + (rank-(size-k_rem))
      k_own = k_own+1
      k_end   = k_init+k_own
    k_end   = k_init+k_own
    return np.array([k_init,k_end])


  def distribute_full(self, num_p, rank, size):
    """
    *Variable* block cyclic distribution to take
    care of the load-balancing and maintain easy
    data access;
    num_p: Number of points=num_b*num_k
    NOTE: Simplified version for now
    """
    # Warning
    if num_p%size != 0:
      if rank == root:
        print_f("Bad Parallelization!")
        print_f("Number of bands need to be divisible by mpi proc.")
        print_f("Exiting...")
      comm.Abort(1)
    # Easy integer division
    p_own = num_p // size
    # Check if there's a remainder
    p_rem = num_p % size
    # Starting point for each process as long
    # as the rank < size-p_rem
    p_init = rank*p_own
    # These ranks will handle 1 extra points
    if rank >= (size - p_rem):
      p_init = (rank*p_own) + (rank-(size-p_rem))
      p_own = p_own+1
      p_end   = p_init+p_own
    p_end   = p_init+p_own
    return np.array([p_init, p_end])


  def distribute_E_ph(self, num_E, rank, size):
    """
    Distribution of photon energies for the optical 
    conductivity calculations
    ---------
    *Variable* block cyclic distribution to take
    care of the load-balancing and maintain easy
    data access.
    """
    # Warning
    if num_E < size:
      if rank == root:
        print_f("Bad Parallelization! Some processes will be idle")
        print_f("Exiting...")
        comm.Abort(1)
    # Easy integer division
    E_own = num_E // size
    # Check if there's a remainder
    E_rem = num_E % size
    # Starting point for each process as long
    # as the rank < size-E_rem
    E_init = rank * E_own
    # These ranks will handle 1 extra photon enrgies
    if rank >= (size - E_rem):
      E_init = (rank*E_own) + (rank-(size-E_rem))
      E_own = E_own+1
      E_end   = E_init+E_own
    E_end   = E_init+E_own
    return np.array([E_init,E_end])


  def get_E_ck(self):
    """
    Returns the Electronic single particle bands
    for conduction bands
    """
    for i in range(self.dft.shape[0]):
      for j in range(self.dft.shape[1]):
        if isinstance(self.dft[i,j],str):
          if "cb" in self.dft[i,j].casefold():
            if "qe" in self.dft[i,0].casefold():
              return E_qe(self.dft[i,1])\
                     [int(self.dft[i,2])+int(self.dft[i,4]):\
                      int(self.dft[i,3])+int(self.dft[i,4])]
            elif "siesta" in self.dft[i,0].casefold():
              return E_siesta(self.dft[i,1])\
                     [int(self.dft[i,2])+int(self.dft[i,4]):\
                      int(self.dft[i,3])+int(self.dft[i,4])]
    
  
  def get_E_vk(self):
    """
    Returns the Electronic single particle bands
    for valence bands
    """
    for i in range(self.dft.shape[0]):
      for j in range(self.dft.shape[1]):
        if isinstance(self.dft[i,j],str):
          if "vb" in self.dft[i,j].casefold():
            if "qe" in self.dft[i,0].casefold():
              return E_qe(self.dft[i,1])\
                       [int(self.dft[i,2])+int(self.dft[i,4]):\
                        int(self.dft[i,3])+int(self.dft[i,4])]
            elif "siesta" in self.dft[i,0].casefold():
              return E_siesta(self.dft[i,1])\
                       [int(self.dft[i,2])+int(self.dft[i,4]):\
                        int(self.dft[i,3])+int(self.dft[i,4])]


  def get_Delta_csk_unit(self):
    """
    Returns the SOC effects with respect to non-polar.
    calculations for the unit cell. 
    """
    # Load the appropriate file
    f1 = h5py.File("DELTA.hdf5", 'r')
    Delta_nsk = np.array(f1['D_nsk_unit'])
    f1.close() 

    for i in range(self.dft.shape[0]):
      for j in range(self.dft.shape[1]):
        if isinstance(self.dft[i,j],str):
          if "cb" in self.dft[i,j].casefold():
            # SIESTA only
            #if "siesta" in self.dft[i,0].casefold():
            #  return \
            #  Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
            #  int(self.dft[i,3])+int(self.dft[i,4]),:,:]
            return \
              Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
              int(self.dft[i,3])+int(self.dft[i,4]),:,:]


  def get_Delta_vsk_unit(self):
    """
    Returns the SOC effects with respect to non-polar.
    calculations for the unit cell. 
    """ 
    # Load the appropriate file
    f1 = h5py.File("DELTA.hdf5", 'r')
    Delta_nsk = np.array(f1['D_nsk_unit'])
    f1.close() 

    for i in range(self.dft.shape[0]):
      for j in range(self.dft.shape[1]):
        if isinstance(self.dft[i,j],str):
          if "vb" in self.dft[i,j].casefold():
            return \
              Delta_nsk[int(self.dft[i,2])+int(self.dft[i,4]):\
              int(self.dft[i,3])+int(self.dft[i,4]),:,:]


  def get_Delta_vsk_pmu(self):
    """
    Returns the SOC effects with respect to non-polar.
    calculations. This is a poor man's unfolding version.
    ONLY use if you know very well what you are doing!!!
    ** Under approximation that the valence bands contributing
       to the excitons come from K point and G-point
       At the K-point, the splitting is about 445 meV.
       At the G-point, the splitting is zero.
       One needs to look at the band-structure and analyze\
       the wave-functions further to confirm this hypotheis.
       ****Twist-angle dependent****
    # Ideally this could be done within the source code by calling 
    # the following lines, however, as these numbers are twist-angle
    # dependent and needs care to use, I will use a separate file
    # for convencience. 
    """
    Delta_vsk = np.load(self.absorp[3])[0]
    return Delta_vsk 


  def get_Delta_csk_pmu(self):
    """
    Returns the SOC effects with respect to non-polar.
    calculations. This is a poor man's unfolding version.
    ONLY use if you know very well what you are doing!!!
    ** Under approximation that the conduction bands contributing
       to the excitons come from K point
       At the conduction bands, the splitting is about 40 meV.
    # Ideally this could be done within the source code by calling 
    # the following lines, however, as these numbers are twist-angle
    # dependent and needs care to use, I will use a separate file
    # for convencience. 
    """
    Delta_csk = np.load(self.absorp[3])[1]
    return Delta_csk 


  def phasefix(self, C_nm_k):
    """
    Fixes the phase of single-particle wave-functions
    by choosing the sum of basis-set coeffs of wfns 
    to be a real number. 
    Ref: Rohlfing and Louie, PRB 62 (2000).
    @input
      C_nm_k: Linear Combination of Atomic or Atomic-like
              Orbital coefficients
    @output
      C_nm_k: Phase fixed coeeficients
    """
    if rank == 0:
      print_f()
      print_f("Fixing the phases of singel-particle wfns.")
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
          print_f()
          print_f("Phase fixing didn't work properly")
          print_f("Exiting...")
          sys.exit()
    return C_nm_k


  def c_nc_k(self,C_nm_k,cb_min,cb_max):
    """
    Separate Conduction band manifold
    """
    return self.phasefix(C_nm_k)[:, cb_min:cb_max, :]  


  def c_nv_k(self,C_nm_k,vb_min,vb_max):
    """
    Separate Valence band manifold
    """
    return self.phasefix(C_nm_k)[:, vb_min:vb_max, :]



  def setup_potential(self):
    """
    Based on the input as in pymex.inp figure out the
    potential parameter details
    """
    # Potential name
    pot = str(self.interact[0].casefold())+ str("_")+\
          str(self.interact[1].casefold())+ str("_")+\
          str(self.interact[2].casefold())+str("_r")
    # Intralayer
    if "monolayer" in self.interact[1].casefold():
      # Singlelayer
      if len(self.interact)-5 == 3:
        return pot, self.interact[5],self.interact[6],\
               self.interact[7]
      # Bilayer
      elif len(self.interact)-5 == 6:
        print_f("Potential format not supported")
        print_f("Exiting...")
        sys.exit()
      else:
        print_f("Potential not supported!")
        print_f("Exiting...")
        sys.exit()
    elif "bilayer" in self.interact[1].casefold():
      if len(self.interact)-5 == 6:  
        return pot, self.interact[5], self.interact[6],\
               self.interact[7], self.interact[8],\
               self.interact[9], self.interact[10]
      else:
        print_f("Potential not supported!")
        print_f("Exiting...")
        sys.exit()


  def D__(self, c,v,k,cp,vp,kp):
    """
    Status: Deprecated
    Optimized Cythonized version
    """
    myatoms = self.atoms.astype(int)
    return Direct(c,v,k,cp,vp,kp,myatoms,\
                           self.Rvec,self.kvec,self.C_nc_k,\
                           self.C_nv_k,self.V_r_full)


  def D_cvk_cpvpkp_old(self,c,v,k,cp,vp,kp):
    """
    Status: Deprecated
    NOTE: Very slow method. Looks at all the 
    Wannier Function position explicitly to compute
    the Direct term. Is only kept for benchmarking.

    Direct term for <cvk|D|cpvpkp>
    * Can be a bottleneck when n1,n3 is very large
    @ input:
      c,cp: conduction bands;
      v,vp: valence bands;
      k,kp: k-points;
      All are indices;
    @output:
      Direct e-h interaction term
    """
    t1 = time.time()
    potname, epsilon_d, r0, a0 = self.setup_potential()
    R_ = self.Rvec()
    if "keldysh_intralayer_simple" in potname:
      tmp = 0.0 + 0j
      for d in range(self.n1):
        for e in range(self.n3):
          for f in range(R_.shape[0]):
            diff_k = np.subtract(self.kvec[k],self.kvec[kp])
            tmp = tmp+(np.conjugate(self.C_nc_k[d,c,k])*\
            self.C_nc_k[d,cp,kp]*self.C_nv_k[e,v,k]\
            *np.conjugate(self.C_nv_k[e,vp,kp])*\
            self.potential.keldysh_intralayer_simple_r\
            (self.potential.r(R_[f],d,e),epsilon_d,r0,a0)*\
            np.exp(1j*np.dot(diff_k, R_[f])))
      print("Time taken for old code", time.time()-t1)
      return tmp/np.prod(self.kg)
    else:
      print_f("Haven't implemented the full one")
      print_f("Exiting...")
      sys.exit()


  def D_cvk_cpvpkp(self,c,v,k,cp,vp,kp):
    """
    Status: Only use when cythonized version doesn't work
    Direct term for <cvk|D|cpvpkp>
    This is *pure python version of the new-algorithm*
    * Can be a bottleneck when n1,n3 is very large
    @ input:
      c,cp: conduction bands;
      v,vp: valence bands;
      k,kp: k-points;
      All are indices;
    @output:
      Direct e-h interaction term
    """
    tmp = 0.0 + 0j
    for i in range(self.atoms.shape[0]):
      for d in range(self.atoms[i,1],self.atoms[i,2]):
        for j in range(self.atoms.shape[0]):
          for e in range(self.atoms[j,1],self.atoms[j,2]):
            for f in range(self.Rvec.shape[0]):
              diff_k = np.subtract(self.kvec[k],self.kvec[kp])
              tmp = tmp+(np.conjugate(self.C_nc_k[d,c,k])*\
                  self.C_nc_k[d,cp,kp]*self.C_nv_k[e,v,k]\
                  *np.conjugate(self.C_nv_k[e,vp,kp])*\
                  self.V_r_full[f,i,j]*\
                  np.exp(1j*np.dot(diff_k, self.Rvec[f])))
    return tmp/np.prod(self.kg)



  def is_hermitian(self, M):
    """
    Checks if a matrix, M is Hermitian or not.
    @input
      M: Matrix
  """
    M = np.matrix(M)
    if not np.allclose(M, M.H):
      raise ValueError('BSE Hamiltonian is not Hermitian')
      print_f("Exiting...") 
      sys.exit()
    else:
      pass


  def get_hole_density(self, i_vb):
    """
    Extracts the Wannier90 derived hole
    density for a specific valence band
    @input
      i_vb: Intended valence band index 
            0: Valence band maximum
            1: 1st band below VBM
            ....
    @output
      loc_r: (k,x,y,z) format
             k is the momentum
             x is the orbital location along x
             y is the orbital location along y
             z is the charge density associated 
               with the in-plane positions
    NOTE: All counting starts from 0 based on Python
    """
    if rank == root:
      map_wf = self.wan2bse.map_WF()
      pos_wf = map_wf[:,3:6].astype(float)
      ind_wf = map_wf[:,0:3].astype(int)
      # Extract the valence band index ranging
      # from valence band maximum to below
      # That's why i_vb is subtracted.
      print_f()
      n_vb = self.C_nv_k.shape[1]-(i_vb+1)
      print_f("NOTE: ")
      print_f("Extracting data of the %d-th valence band"%(n_vb))
      print_f("Counting from top and %d VB is mapped"%(i_vb))
      print_f()
      loc_r = np.zeros((self.C_nv_k.shape[2],\
                        ind_wf.shape[0], 4),\
                        dtype=float)
      # Extract density for all k-points
      for k in range(self.C_nv_k.shape[2]):
        for i in range(ind_wf.shape[0]):
          tmp = 0.0
          for j in range(ind_wf[i,1], ind_wf[i,2]):
            tmp = tmp + np.square(np.abs(self.C_nv_k[j,n_vb,k]))
          loc_r[k,i] = np.array([pos_wf[i,0],\
                                 pos_wf[i,1],\
                                 pos_wf[i,2],\
                                 tmp])

      savename = "Hole_" + str(i_vb)+ ".hdf5"
      with h5py.File(savename, 'w') as f:
        f.create_dataset('density', data=loc_r)
      print_f("Written the Hole density to a HDF5 file")
    return None
      

  def get_electron_density(self, i_cb):
    """
    Extracts the Wannier90 derived electron
    density for a specific conduction band.
    @input
      i_cb: Intended conduction band index 
            0: Conduction band minimum
            1: 1st band above CBM
            ....
    @output
      loc_r: (k,x,y,z) format
             k is the momentum
             x is the orbital location along x
             y is the orbital location along y
             z is the charge density associated 
               with the in-plane positions
    NOTE: All counting starts from 0 based on Python
    """
    if rank == root:
      # This is going to be used for n1
      map_wf = self.wan2bse.map_WF()
      pos_wf = map_wf[:,3:6].astype(float)
      ind_wf = map_wf[:,0:3].astype(int)
      print_f()
      print_f("Extracting data of the %d-th conduction band"%(i_cb))
      print_f()
      loc_r = np.zeros((self.C_nc_k.shape[2],\
                        ind_wf.shape[0],4),\
                        dtype=float)
      # Extract density for all k-points
      for k in range(self.C_nc_k.shape[2]):
        for i in range(ind_wf.shape[0]):
          tmp = 0.0
          for j in range(ind_wf[i,1], ind_wf[i,2]):
            tmp = tmp + np.square(np.abs(self.C_nv_k[j,i_cb,k]))
          loc_r[k,i] = np.array([pos_wf[i,0],\
                                 pos_wf[i,1],\
                                 pos_wf[i,2],\
                                 tmp])
      savename = "Electron_" + str(i_cb)+ ".hdf5"
      with h5py.File(savename, 'w') as f:
        f.create_dataset('density', data=loc_r)
      print_f("Written the Electron density to a HDF5 file")
    return None


  def exciton_r(self, S, rh, savename):
    """
    Computes the exciton wave-function in real-space
    for a fixed position of the hole, rh
    @input:
      S: Exciton index 
      rh: (x,y,z)- 3 dimensional position in Angstrom
      savename: File savename
    """
    # get the wannier function location with a good
    # map, so that it's easier
    # NOTE: Normally positions are outputted in angstrom 
    n3 = self.get_hole_loc(rh)
    if n3 is None:
      print_f("Use a hole position (rh) that falls within 0.2 Ang")
      comm.Abort(1)
    else:
      if rank == root:
        print_f("Based on the hole-location-")
        print_f("WF from [%d:%d] will be used"%(n3[1], n3[2]))
        print_f()
    # This is going to be used for n1
    map_wf = self.wan2bse.map_WF()
    if rank == root:
      print_f("Mapped Wannier functions", map_wf)
    pos_wf = map_wf[:,3:6].astype(float)
    ind_wf = map_wf[:,0:3].astype(int)

    # Load the BSE eigenvalues
    f1 = h5py.File("Eigval.hdf5", 'r')
    eigval = np.array(f1['eigval'])
    f1.close()

    # Load the BSE eigenvectors
    f2 = h5py.File("Eigvec.hdf5", 'r')
    eigvec = np.array(f2['eigvec'])
    f2.close()

    # Exciton in the right-format
    S_cvk = np.zeros((len(eigval),self.C_nc_k.shape[1],\
                      self.C_nv_k.shape[1],
                      self.C_nc_k.shape[2]),\
                      dtype = complex)
    # Extract the BSE eigenvectors for processing
    # A^S_cvk get the associated S value from input
    AS_cvk = eigvec[:,S].reshape((\
                         self.C_nc_k.shape[1],\
                         self.C_nv_k.shape[1],\
                         self.C_nc_k.shape[2]))

    # Old checks
    #self.Rvec = self.potential.set_Rvec(5)

    # Cythonized calculations with thread-support
    data = np.empty((ind_wf.shape[0], self.Rvec.shape[0]),\
                   dtype=complex)
    #data[:,:] = opt_exciton_r(AS_cvk, self.Rvec, self.kvec,\
    #                     self.C_nc_k, self.C_nv_k,\
    #                     ind_wf, pos_wf, n3[0:3].astype(int))
    # Use thread always to get things quickly done
    data[:,:] = opt_exciton_r_thread(AS_cvk, self.Rvec, self.kvec,\
                         self.C_nc_k, self.C_nv_k,\
                         ind_wf, pos_wf, n3[0:3].astype(int),\
                         rh)
    if rank == root:
      # File for exciton wave-function in real space
      np.save(savename, data)
      # Rvector file
      np.save("Rvec", self.Rvec)
      # Position of Wannier centers at the home cell
      np.save("pos_wf", map_wf)
  

  def get_hole_loc(self, rh):
    """
    For a given hole-location, find the Wannier functions
    that could be correspond to n3
    Derivations can be found: Mypaper
    """
    map_wf = self.wan2bse.map_WF()
    for i in range(map_wf.shape[0]):
      pos = map_wf[i,3:6]
      # Closest Wannier function location for a given hole
      # coordinate
      if np.linalg.norm(np.subtract(pos, rh)) < 0.5:
        return map_wf[i,:]
