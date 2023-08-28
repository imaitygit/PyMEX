#-----------------------------|
#email: i.maity@imperial.ac.uk|
#Author: Indrajit Maity       |
#-----------------------------|

import numpy as np
import sys
from functools import partial
from dft2bse import *
print_f = partial(print, flush=True)
import h5py

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

# |=====================================|
# |Class to Add Spin-Orbit-Coupling (SOC)|
# |as a perturbation                     |
# |=====================================|
class SOC(object):

  def __init__(self, engine, file_unpol_u, file_soc_u,\
               moire2unit, file_moire2unit, file_delta):
    """
    @input
    engine: Band-structure calculated using what-
            (QE or SIESTA or TB);
    file_unpol_u: Filename for bands without soc
                  included for a unit-cell of 2D
                  material.
    file_soc_u: Filename for bands with soc included
                for a unit-cell of 2D material.
    moire2unit: Map the moire bands to unit cell and 
                read the soc effects on bands from a
                file "file_moire2unit"
    file_moire2unit: See above.
    file_delta: Delta_soc_nks file for the unit-cell
    """
    self.engine = engine  
    self.file_unpol_u = file_unpol_u
    self.file_soc_u = file_soc_u
    self.moire2unit = moire2unit
    self.file_moire2unit = file_moire2unit
    self.file_delta = file_delta

  def get_Delta_nsk_unit(self):
    """
    Computes for the unit-cell:
    D_nks = \eps_{nks}^{SOC} - \eps_{nks}^{Spin-nonpol}
    NOTE: We try to be consistent with the 
          Qiu et al., Phys. Rev. Lett. 111, 216805 (2013)/
          Qiu et al., Phys. Rev. B 93, 235435 (2015),
          by using non-polarized calculations. Thus,
          we need to *have* s explicitly in the subtraction.
    """
    if self.engine.casefold() == "siesta":
      # Call SIESTA readers
      # Non-polarized calculations at unit-cell
      E_nk = E_siesta(self.file_unpol_u)
      # SOC calculations at unit-cell
      E_nk_soc = E_siesta(self.file_soc_u)
    elif self.engine.casefold() == "qe":
      # Call QE readers
      # Non-polarized calculations at unit-cell
      E_nk = E_qe(self.file_unpol_u)
      # SOC calculations at unit-cell
      E_nk_soc = E_qe(self.file_soc_u)
    else:
      if rank == root:
        print_f("%s not supproted for SOC perturbation yet")
        print_f("Exiting...")
      comm.Abort(1)

    E_nsk = E_nk_soc.reshape(E_nk.shape[0], 2, E_nk.shape[1])
    Delta_nsk = np.zeros((E_nsk.shape))
    for i in range(Delta_nsk.shape[0]):
      for spin in range(2):
        for j in range(Delta_nsk.shape[2]):
          Delta_nsk[i,spin,j] = E_nsk[i,spin,j] - E_nk[i,j]

    # Write the data to a *HDF5 file
    if rank == root:
      print_f("Writing the Delta_nsk to a file")
      print_f()
      # HDF5
      f = h5py.File(self.file_delta, "w")
      # For large systems play with the compression_opts
      dset = f.create_dataset("D_nsk_unit",\
             ((Delta_nsk.shape[0], Delta_nsk.shape[1],\
               Delta_nsk.shape[2])),dtype='float')
      dset[:,:,:] = Delta_nsk
    comm.Barrier()
    return None    
    

  def soc_unit(self):
    """
    Computes the \Delta^{soc_u}_{vcks} of the unit-cell
    v: Valence band of unit-cell
    c: Conduction band of unit-cell
    k: k-point (Keep it sufficiently fine)
    s: spin; In-principle, we simply double v,c at every k
    """ 
    return None
   
