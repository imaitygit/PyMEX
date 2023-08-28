# Author: Indrajit Maity
# email: indrajit.maity02@gmail.com
#-------------------------------

# NOTE: Please use the proper E_siesta..() to read
# SIESTA output files

import numpy as np

def E_siesta_MAX(filename):
  """
  Reads SIESTA output when BandPoints are used
  ------------------------------------
  -----------For MAX versions---------
  ------------------------------------
  @input
    filename: SIESTA bandstructure file(standard)
  @output
    E: bands without excluding any bands
       (nkp, nbnd) array
  """
  f = open(filename, "r")
  lines = f.readlines()[:-5]
  f.close()

  # Exract nkp and nbnd
  for i in range(len(lines)):
    nbnd = eval(lines[3].split()[0])
    nkp = eval(lines[3].split()[2])

  # Number of lines for each k-points
  # 8 is standard and thus, hard-coded
  num_l = 10; sh = 1
  l = nbnd%num_l
  if l != 0.0:
    nline = nbnd//num_l + 1
  else:
    nline = nbnd//num_l

  print(nline)
  # Extract the eigenvalues
  E = []
  for i in range(4, len(lines), nline):
    for j in range(i, i+nline):
#      print(lines[j])
      if j == i:
        for k in range(1, 1+num_l):
          E.append(eval(lines[j].split()[k]))
      else:
        for k in range(len(lines[j].split())):
          E.append(eval(lines[j].split()[k]))
  return np.moveaxis(np.array(E).reshape(nkp, nbnd), 0, -1)


def E_qe(filename):
  """
  Reads Quantum ESPRESSO output from bands calculations
  @input
    filename: QE bands (bands.x output)
  @output
    E: bands without excluding any bands
       (nkp, nbnd) array
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  for i in range(len(lines)):
    if "nbnd" in lines[i].casefold():
      nbnd = eval(lines[i].split()[2])[0]
      nkp = eval(lines[i].split()[4])

  # Number of lines for each k-points
  # 8 is standard and thus, hard-coded
  num_l = 10
  l = nbnd%num_l
  if l != 0.0:
    nline = nbnd//num_l + 1
  else:
    nline = nbnd//num_l

  # Extract the eigenvalues
  E = []
  for i in range(1, len(lines), nline+1):
    for j in range(i+1, i+nline+1):
      for k in range(len(lines[j].split())):
        E.append(eval(lines[j].split()[k]))
  return np.moveaxis(np.array(E).reshape(nkp, nbnd), 0, -1)


def E_siesta(filename):
  """
  Reads SIESTA output when BandPoints are used
  @input
    filename: SIESTA bandstructure file(standard)
  @output
    E: bands without excluding any bands
       (nkp, nbnd) array
  NOTE: Checked against siesta-4.1.5 version
  """
  f = open(filename, "r")
  lines = f.readlines()
  f.close()

  # Exract nkp and nbnd
  for i in range(len(lines)):
    nbnd = eval(lines[2].split()[0])
    nkp = eval(lines[2].split()[2])

  # Number of lines for each k-points
  # 8 is standard and thus, hard-coded
  num_l = 8; sh = 1
  l = nbnd%num_l
  if l != 0.0:
    nline = nbnd//num_l + 1
  else:
    nline = nbnd//num_l

  # Extract the eigenvalues
  E = []
  for i in range(3, len(lines), nline):
    for j in range(i, i+nline):
#      print(lines[j])
      if j == i:
        for k in range(3, 3+num_l):
          E.append(eval(lines[j].split()[k]))
      else:
        for k in range(len(lines[j].split())):
          E.append(eval(lines[j].split()[k]))
  return np.moveaxis(np.array(E).reshape(nkp, nbnd), 0, -1)
