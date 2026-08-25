import os
import sys
sys.path.append("/Users/indrajitmaity/Codes/GitHub/PyMEX/develop/hBN/Paulina_updated")
from bse import BSE
import time
from functools import partial
print_f = partial(print, flush=True)
import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0


#------------
# Calculations
#-------------
t1 = time.time()
BSE = BSE("pymex_tb.yaml")
BSE.write_exciton_H()
BSE.diagon_BSE()
BSE.optical_conductivity()

#rh = np.array([0, 1.1, 0.0])
#BSE.get_exciton_wfn_at_rh(rh, num_S=10)
#BSE.get_electron_density()
#BSE.get_hole_density()
#BSE.get_electron_for_exciton(0)

#---------------
if rank == root:
  print_f("Time taken on %d processes :%.3f secs."%(size,\
       time.time()-t1))
