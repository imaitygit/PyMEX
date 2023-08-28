#-----------------------------|
#email: i.maity@imperial.ac.uk|
#Author: Indrajit Maity       |
#-----------------------------|

import sys
sys.path.append("/work/e05/e05/imaity/codes/pymex/src")
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
BSE = BSE("pymex.inp")
BSE.write_Ham_parallel()
BSE.diagon_BSE()
BSE.sigma_xx()

#Plotting etc.
# S, c, v
#BSE.plot_exciton_k(0,0,0)
#BSE.plot_exciton_k(1,0,0)
#BSE.plot_exciton_k(2,0,0)
#BSE.plot_exciton_k(3,0,0)
#BSE.plot_exciton_k(4,0,0)
#BSE.plot_exciton_k(5,0,0)
#---------------
if rank == root:
  print_f("Time taken on %d processes :%.3f secs."%(size,\
       time.time()-t1))
