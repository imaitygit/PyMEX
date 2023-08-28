import numpy as np
import sys
from functools import partial
print_f = partial(print, flush=True)


def unit2super(m,A):
  """
  Returns the supercell lattice vectors of a 
  ginven unit-cell lattice vectors.
  @input
    m: array of integers (for supercell)
    A: Unit-cell lattice vectors
  """
  S = np.zeros((3,3))
  np.fill_diagonal(S,m)
  return np.matmul(A,S)


def ang2crys(A, pos):
  """
  Converts position(s) of an atom (or series of atoms)
  to crystal coordinates
  @input
    A: Lattice vectors
    pos: positions in ansgtrom
  """
  Ainv = np.linalg.inv(A)
  if len(pos.shape) == 1:
    pos_c = np.dot(pos, Ainv)
    return pos_c
  elif len(pos.shape) == 2:
    pos_c = np.zeros((pos.shape[0], pos.shape[1]))
    for i in range(pos.shape[0]):
      pos_c[i] = np.dot(pos[i], Ainv)
    return pos_c
  else:
    print_f("Unrecongnized data format!")
    print_f("Exiting...")
    sys.exit()



def crys2ang(A, pos_c):
  """
  Converts position(s) of an atom (or series of atoms)
  from crystal coordinates to angstroms.
  @input
    A: Lattice vectors
    pos_c: positions in crystal coordinates
  """
  if len(pos_c.shape) == 1:
    pos = np.dot(pos_c, A)
    return pos
  elif len(pos_c.shape) == 2:
    pos = np.zeros((pos_c.shape[0], pos_c.shape[1]))
    for i in range(pos_c.shape[0]):
      pos[i] = np.dot(pos_c[i], A)
    return pos
  else:
    print_f("Unrecongnized data format!")
    print_f("Exiting...")
    sys.exit()

# line counter
def linecounter(num_l, L):
  """
  Computes the number of lines to that writes/reads L
  elements/points with num_l of elements per line.
  """
  rem = L%num_l
  if rem != 0.0:
    l = L//num_l + 1
  else:
    l = L//num_l
  return l


# Chek unitarity of a matrix
def is_unitary(M):
  """
  Checks if a matrix, M is unitary or not.
  Works for both square and rectangular M.
  @input
   M: Matrix
  @output
    bool-True if the matrix is unitary
  """
  M = np.matrix(M)
  I = M.H * M
  return np.allclose(np.eye(I.shape[0]), I)
