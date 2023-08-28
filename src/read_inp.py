#!/usr/bin/python


import sys,string
import numpy as np
from datetime import datetime
from functools import partial
print_f = partial(print, flush=True)

#MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
root = 0


def read_inp(filename):
  """
  Reads the input file from pymex.inp
   * Strings are case in-sensitive
   * Ignores blank lines
   * Ignores anything after '#'
  @input:
    filename: pymex.inp
  """
  f = open(filename)
  # ignore blank spaces
  lines = [line for line in f.readlines() if line.strip()]
  f.close()

  # Be default, bilayers
  l = 2
  for i in range(len(lines)):
    c = lines[i].partition("#")[0]
    if "bse:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      calc = tmp.split()[0]
    elif "layer:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      nature = tmp.split()[0]

    # Eignvalues information
    if "dft:" in c.casefold():
      dft = np.empty((l, 7), dtype=object)
      for j in range(i+1, i+1+l):
        tmp = lines[j].partition("#")[0]
        for k in range(2):
          dft[j-i-1][k] = tmp.split()[k]
        for k in range(2,5):
          dft[j-i-1][k] = int(tmp.split()[k])
        for k in range(5,6):
          dft[j-i-1][k] = tmp.split()[k]
        for k in range(6,7):
          try:
            dft[j-i-1][k] = eval(tmp.split()[k])
          except IndexError:
            dft[0][k] = 0.0
            dft[1][k] = 100.0


    # Coefficients information
    elif "coefficients:" in c.casefold():
      coeff = np.empty((l, 6), dtype=object)
      for j in range(i+1, i+1+l):
        tmp = lines[j].partition("#")[0]
        for k in range(2):
          coeff[j-i-1][k] = tmp.split()[k]
        for k in range(2,5):
          coeff[j-i-1][k] = int(tmp.split()[k])
        for k in range(5,6):
          coeff[j-i-1][k] = tmp.split()[k]

    # Hamiltonian information
    elif "hamiltonian:" in c.casefold():
      ham = np.empty((l, 4), dtype=object)
      for j in range(i+1, i+1+l):
        tmp = lines[j].partition("#")[0]
        for k in range(4):
          ham[j-i-1][k] = tmp.split()[k]

    # Structure information
    elif "structure:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      struc = np.empty((2), dtype=object)
      for k in range(2):
        struc[k] = tmp.split()[k]

    # Interaction information
    elif "interaction:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      interact = np.empty((len(tmp.split())), dtype=object)
      for k in range(5):
        interact[k] = tmp.split()[k]
      for k in range(5,len(tmp.split())):
        interact[k] = eval(tmp.split()[k])
 
    # Wannier Function locations
    elif "location:" in c.casefold():
      wout = np.empty((l,3), dtype=object)
      for j in range(i+1, i+1+l):
        tmp = lines[j].partition("#")[0]
        for k in range(3):
          wout[j-i-1][k] = tmp.split()[k]

    # Kgrid/Real-space gridding
    elif "kgrid:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      kg = np.array([int(tmp.split()[0]),\
                     int(tmp.split()[1]),\
                     int(tmp.split()[2])])
 
    # Absorption calculations
    elif "absorption:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      absorp = np.array([tmp.split()[0],\
                         tmp.split()[1],\
                         tmp.split()[2],\
                         tmp.split()[3],\
                         tmp.split()[4],\
                         tmp.split()[5],\
                         tmp.split()[6]]) 

    # Absorption calc. window and parameters
    elif "ephoton:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      ephparam = np.array([eval(tmp.split()[0]),\
                         eval(tmp.split()[1]),\
                         eval(tmp.split()[2]),\
                         eval(tmp.split()[3])])

    # Parallelization options
    elif "parallel:" in c.casefold():
      tmp = lines[i+1].partition("#")[0]
      parallel = np.array([tmp.split()[0],\
                           tmp.split()[1]])

  return calc, nature, dft, coeff, ham, struc, interact,\
         wout, kg, absorp, ephparam, parallel


def print_inp(filename):
  """
  Print the input parameters
  """
  # IM: Formatting is very old; Spend some time to fix
  # those!!
  calc, nature, dft, coeff, ham, struc, interact, wout, kg,\
  absorp, ephparam, parallel = read_inp(filename)

  if rank == root:
    print_f(f"\n")
    print_f("|"+60*"-"+"|")
    print_f("""
  _______              ____    ____   ________   ____  ____
 |_   __ \            |_   \  /   _| |_   __  | |_  _||_  _|
   | |__) |   _   __    |   \/   |     | |_ \_|   \ \  / /
   |  ___/   [ \ [  ]   | |\  /| |     |  _| _     > `' <
  _| |_       \ '/ /   _| |_\/_| |_   _| |__/ |  _/ /'`\ \_
 |_____|    [\_:  /   |_____||_____| |________| |____||____|
             \__.' 
       """)     
    print_f("|"+60*" "+"|")
    print_f("|"+ "PyMEX (Python package for Moiré EXcitons)".center(60)+"|")
    date = datetime.now().date().strftime("%d/%m/%Y")
    time = datetime.now().time().strftime("%H:%M:%S")
    print_f("|"+f"{'Date: ' + date} {'Time: '+ time}".center(60)+"|")
    print_f("|"+60*"-"+"|")
    print_f("\n")
    print_f("|"+60*"-"+"|")
    print_f("|"+"INPUT PARAMETERS".center(60)+"|")
    print_f("|"+60*"-"+"|")

    # Full calculations or separate intra/inter-layer excitons
    if nature == "full":
      print_f("|"+"Both VB and CB will be read from the same file".\
              center(60)+"|")
    # Likely going to be deprecated-
    elif nature == "isolated":
      print_f("|"+"VB and CB will be read from different files".\
              center(60)+"|")

    # Hard-coded
    #**********
    l = 2
    #**********
    for i in range(l):
      print_f("|"+60*" "+"|")
      if dft[i][5].casefold() == "vb":
        print_f("|"+"%-30s %-29s"%\
        ("Assigned as: ", dft[i][5])+"|")
        print_f("|"+60*" "+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Bands from:", dft[i][0])+\
                    "{0:<18} {1:<11}".format("Filename:", dft[i][1])+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Bandmin:", dft[i][2])+\
                    "{0:<18} {1:<11}".format("Bandmax:", dft[i][3])+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Skip_coreband:", dft[i][4])+\
                    "{0:<18} {1:<11}".format("E_l1", dft[i][6])+"|")
        for i in range(l):
          if coeff[i][5].casefold() == "vb":
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k from:", coeff[i][0])+\
                    "{0:<11} {1:<18}".format("Filename:", coeff[i][1])+"|")
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k min:", coeff[i][2])+\
                    "{0:<18} {1:<11}".format("C_nm_k max:", coeff[i][3])+"|")
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k skip:", coeff[i][4])+\
                    "{0:<18} {1:<11}".format("", "")+"|")
        for i in range(l):
          if ham[i][3].casefold() == "vb":
            print_f("|"+"{0:<18} {1:<11}".format("Ham. from:", ham[i][0])+\
                    "{0:<11} {1:<18}".format("Filename:", ham[i][1])+"|")
        for i in range(l):
          if wout[i][2].casefold() == "vb":
            print_f("|"+"{0:<18} {1:<11}".format("WF. from:", wout[i][0])+\
                    "{0:<14} {1:<15}".format("Filename:", wout[i][1])+"|")
      elif dft[i][5].casefold() == "cb":
        print_f("|"+"%-30s %-29s"%\
        ("Assigned as", dft[i][5])+"|")
        print_f("|"+60*" "+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Bands from:", dft[i][0])+\
                    "{0:<18} {1:<11}".format("Filename:", dft[i][1])+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Bandmin:", dft[i][2])+\
                    "{0:<18} {1:<11}".format("Bandmax:", dft[i][3])+"|")
        print_f("|"+"{0:<18} {1:<11}".format("Skip_coreband:", dft[i][4])+\
                    "{0:<18} {1:<11}".format("E_l1", dft[i][6])+"|")
        for i in range(l):
          if coeff[i][5].casefold() == "cb":
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k from:", coeff[i][0])+\
                    "{0:<14} {1:<15}".format("Filename:", coeff[i][1])+"|")
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k min:", coeff[i][2])+\
                    "{0:<18} {1:<11}".format("C_nm_k max:", coeff[i][3])+"|")
            print_f("|"+"{0:<18} {1:<11}".format("C_nm_k skip:", coeff[i][4])+\
                    "{0:<18} {1:<11}".format("", "")+"|")
        for i in range(l):
          if ham[i][3].casefold() == "cb":
            print_f("|"+"{0:<18} {1:<11}".format("Ham. from:", ham[i][0])+\
                    "{0:<14} {1:<15}".format("Filename:", ham[i][1])+"|")
        for i in range(l):
          if wout[i][2].casefold() == "cb":
            print_f("|"+"{0:<18} {1:<11}".format("WF. from:", wout[i][0])+\
                    "{0:<14} {1:<15}".format("Filename:", wout[i][1])+"|")
      else:
        print_f("Unrecognized forrmat.")
        print_f("Exiting...") 
        sys.exit()      
    print_f("|"+"".center(60)+"|")
    print_f("|"+"{0:<18} {1:<11}".format("Potential:", interact[0])+\
                    "{0:<14} {1:<15}".format("", "")+"|")
    print_f("|"+60*" "+"|")
    print_f("|"+"{0:<20} {1:<9}".format("Potential Parameters", "")+\
                    "{0:<14} {1:<15}".format("", "")+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Type:", interact[1])+\
                    "{0:<14} {1:<15}".format("", "")+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Kind:", interact[2])+\
                "{0:<14} {1:<15}".format("", "")+"|")
    print_f("|"+"{0:<19} {1:<10}".format("Include Direct(D_):", interact[3])+\
               "{0:<19} {1:<10}".format("Exchange(X_):", interact[4])+"|")
    for i in range(int(np.ceil(len(interact[5:])/3))):
      print_f("|"+"{0:30} {1:9} {2:9} {3:9}".format(
              str("L"+str(i+1)+" (epsilon, r0, a0)"),\
              interact[5+(i*3)],\
              interact[6+(i*3)], interact[7+(i*3)])+"|")

    print_f("|"+"".center(60)+"|")
    print_f("|"+"WF centers for e-h inter. are read from structure file"\
             .center(60)+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Structure:", struc[0])+\
                "{0:<14} {1:<15}".format("Filename:", struc[1])+"|")
    print_f("|"+"{0:<14} {1:<15}".format("MP-grid:", str(kg))+\
                "{0:<14} {1:<15}".format("", "")+"|")
    print_f("|"+60*" "+"|")
    print_f("|"+"Optical conductivity calculations".center(60)+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Spin-orbit:", absorp[0])+\
                "{0:<14} {1:<15}".format("Nature:", absorp[1])+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Unfold:", absorp[2])+\
                "{0:<14} {1:<15}".format("File1:", absorp[3])+"|")

    print_f("|"+60*" "+"|")
    print_f("|"+"Optical conductivity parameters (in eV)".center(60)+"|")
    print_f("|"+"{0:<14} {1:<15}".format("Minimum:", ephparam[0])+\
                "{0:<14} {1:<15}".format("Maximum:", ephparam[1])+"|")
    print_f("|"+"{0:<14} {1:<15}".format("DeltaE:", ephparam[2])+\
                "{0:<14} {1:<12}".format("Sigma for Gaussian", ephparam[3])+"|")

    print_f("|"+" ".center(60)+"|")
    print_f("|"+"Parallelization options for PyMEX".center(60)+"|")
    print_f("|"+"{0:<14} {1:<15}".format("BSE:",\
           parallel[0])+ "{0:<14} {1:<15}".format("Conductivity:",\
           parallel[1])+"|")
    print_f("|"+60*"-"+"|")

  return calc, nature, dft, coeff, ham, struc, interact,\
         wout, kg, absorp, ephparam, parallel


def errorcheck_calc(arg):
  """
  Check for errors in calculation methods
  """
  if rank == root:
    if str(arg).casefold() == "dft":
      pass
    elif str(arg).casefold() == "tb":
      print_f("Tight-Binding interface is not implemented yet!")
      print_f("Exiting...")
      comm.Abort(1)
    else:
      print_f("Unrecognizable option")
      print_f("Exiting...")
      comm.Abort(1)


def errorcheck_nature(arg):
  """
  Check for errors in nature of the calculations.
  If "isolated" then the C_nm_k, Bands_QE etc.
  will require separate files. 
  If "full" then everything read from one file.
  Please use the same name for VB and CB in "pymex.inp"
  """
  if rank == root:
    if str(arg).casefold() == "isolated":
      print_f("Not supported for now")
      print_f("Exiting...")
      comm.Abort(1)
    elif str(arg).casefold() == "full":
      pass
    else:
      print_f("Unrecognizable option")
      print_f("Exiting...")
      comm.Abort(1)


def errorcheck_dft(nature, arg):
  """
  Errorcheck on DFT eigenvalues, which are single-
  particle energies;
  """
  if rank == root:
    if str(nature).casefold() == "isolated":
      if str(arg[0,1]) != str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    elif str(nature).casefold() == "full":
      if str(arg[0,1]) == str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    else:
      print_f("Did not pass the checks")
      print_f("Exiting...")
      comm.Abort(1)



def errorcheck_coeff(nature, arg):
  """
  Error-check on Wannier90 coefficients
  """
  if rank == root:
    if str(nature).casefold() == "isolated":
      if str(arg[0,1]) != str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    elif str(nature).casefold() == "full":
      if str(arg[0,1]) == str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    else:
      print_f("Did not pass the checks")
      print_f("Exiting...")
      comm.Abort(1)


def errorcheck_ham(nature, arg):
  """
  Errorcheck on Hamiltonians
  """
  if rank == root:
    if str(nature).casefold() == "isolated":
      if str(arg[0,1]) != str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    elif str(nature).casefold() == "full":
      if str(arg[0,1]) == str(arg[1,1]):
        pass
      else:
        print_f("Incompatible options!")
        print_f("Exiting...")
        comm.Abort(1)
    else:
      print_f("Did not pass the checks")
      print_f("Exiting...")
      comm.Abort(1)


def errorcheck_pot(nature, arg):
  """
  Errorcheck on Keldysh potential
  """
  if rank == root:
    if arg[0].casefold() != "keldysh":
      print_f("Only Keldysh potentials are implemented") 
      print_f("Exiting...")
      comm.Abort(1)
    if eval(arg[3]) and eval(arg[4]):
      pass
    else:
      print_f("Both Direct and Exchange terms are recommended")
      print_f("WARNING: Use with caution")
      pass


def errorcheck_kgrid(arg):
  """
  Errorchecks for the k-grids
  """
  if rank == root:
    # Works only for odd-grids
    if np.all(arg%2!=0):
      pass
    else:
      print_f("Only works for odd MP k-grids")
      print_f("Exiting...")
      comm.Abort(1)


def errorcheck_parallel(arg):
  """
  Errorchecks for the parallelization options
  """
  if rank == root:
    for i in range(arg.shape[0]):
      if i == 0:
        if arg[i].casefold() == "full"\
        or arg[i].casefold() == "kpoints"\
        or arg[i].casefold() == "bands":
          pass
        else:
          print_f("Unrecognizable parallel BSE option")
          print_f("Exiting...")
          comm.Abort(1)
      elif i == 1:
        if arg[1].casefold() == "thread" or\
         arg[1].casefold() == "nothread":
          pass
        else:
          print_f("Unrecognizable parallel conductivity option")
          print_f("Exiting...")
          comm.Abort(1)
