import os
import sys
sys.path.append("/work/e89/e89/imli/codes/pymex_plus/src")

from bse import BSE
import time
from functools import partial
import numpy as np
from mpi4py import MPI

print_f = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0

t1 = time.time()
BSE = BSE("pymex_tb.yaml")
BSE.diagon_BSE()

if rank == root:
  print_f("Time taken on %d processes: %.3f secs." % (size, time.time() - t1))
