import os
import sys
from pathlib import Path
pymex_src_path = "/Users/indrajitmaity/Codes/GitHub/PyMEX/src"
pymex_src_path = (
    Path(pymex_src_path).expanduser().resolve()
    if pymex_src_path
    else Path.cwd()
)
sys.path.insert(0, str(pymex_src_path))

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
#---------------
if rank == root:
  print_f("Time taken on %d processes :%.3f secs."%(size,\
       time.time()-t1))
