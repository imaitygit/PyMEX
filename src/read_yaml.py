#!/usr/bin/env python3

import os, sys
import yaml
import numpy as np
from datetime import datetime
from functools import partial
from mpi4py import MPI
print_f = partial(print, flush=True)

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
root = 0

import os


def num_if_list(x):
  """
  Returns the first element if x is a list, 
  otherwise returns x itself.
  """
  if isinstance(x, list):
    return x[0]
  return x


def list_if_num(x):
  """
  Ensure that x is a list. If it is a scalar, wrap it in a list.
  """
  if isinstance(x, list):
    return x
  return [x]

def is_filename(s):
  """
  Heuristic to detect if a string is a filename.
  """
  return (
    isinstance(s, str)
    and (
      any(s.endswith(ext) for ext in ['.dat', '.hdf5', 
                                      '.bands', '.win', 
                                      '.wout', '.mat'])
      or '/' in s
      or '\\' in s
      or '.' in os.path.basename(s)
    )
  )

def lowercase_keys_except_filenames(obj):
  """
  Recursively lowercase all dict keys and string values, except filenames.
  """
  if isinstance(obj, dict):
    return {
      k.lower(): lowercase_keys_except_filenames(v)
      for k, v in obj.items()
    }
  elif isinstance(obj, list):
    return [lowercase_keys_except_filenames(i) for i in obj]
  elif isinstance(obj, str):
    return obj if is_filename(obj) else obj.lower()
  else:
    return obj

def read_yaml(yaml_file):
  """
  Read a YAML configuration file and return its contents
  with all keys lowercased. Skips lines starting with '#' or '!'.
  """
  try:
    with open(yaml_file, 'r') as file:
      yaml_lines = [
        line for line in file
        if line.strip() and not line.strip().startswith(('#', '!'))
      ]
      cleaned_yaml = '\n'.join(yaml_lines)

    config_raw = yaml.safe_load(cleaned_yaml) or {}
    return lowercase_keys_except_filenames(config_raw)

  except FileNotFoundError:
    if rank == 0:
      print_f(f"Error: YAML file '{yaml_file}' not found.")
    comm.Abort(1)

  except yaml.YAMLError as e:
    if rank == 0:
      print_f(f"Error parsing YAML file: {e}")
    comm.Abort(1)


def print_config(config, indent=0):
  spacer = '  ' * indent
  if isinstance(config, dict):
    for key, value in config.items():
      if isinstance(value, dict):
        print_f(f"{spacer}{key}:")
        print_config(value, indent + 1)
      else:
        print_f(f"{spacer}{key}: {value}")
  elif isinstance(config, list):
    for i, item in enumerate(config):
      print_f(f"{spacer}- [{i}]")
      print_config(item, indent + 1)
  else:
    print_f(f"{spacer}{config}")


def print_yaml(yaml_file):
  """
  Print the contents of a YAML file in a formatted way.
  """
  config = read_yaml(yaml_file)
  width = 70
  if rank == root:
    print_f(f"\n")
    print_f("|"+width*"-"+"|")
    print_f("""
  _______              ____    ____   ________   ____  ____            
 |_   __ \            |_   \  /   _| |_   __  | |_  _||_  _|  
   | |__) |   _   __    |   \/   |     | |_ \_|   \ \  / /   
   |  ___/   [ \ [  ]   | |\  /| |     |  _| _     > `' <   
  _| |_       \ '/ /   _| |_\/_| |_   _| |__/ |  _/ /'`\ \_   
 |_____|    [\_:  /   |_____||_____| |________| |____||____|   
             \__.'                                                 
""")
    print_f("|"+width*" "+"|")
    print_f("|"+ "PyMEX+ (Python package for Moiré EXcitons plus beyond)".center(width)+"|")
    date = datetime.now().date().strftime("%d/%m/%Y")
    time = datetime.now().time().strftime("%H:%M:%S")
    print_f("|"+f"{'Date: ' + date} {'Time: '+ time}".center(width)+"|")
    print_f("|"+width*"-"+"|")
    print_f("\n")
    print_f("|"+width*"-"+"|")
    print_f("|"+"INPUT PARAMETERS".center(width)+"|")
    print_f("|"+width*"-"+"|")  
    print_config(config)
    print_f("|"+width*"-"+"|")
    print_f(("|"+"End of INPUT PARAMETERS".center(width)+"|"))
    print_f("|"+width*"-"+"|")
  return config


def check_excitation(excitation):
  """
  Check if the excitation type is valid.
  """
  valid_excitations = {'exciton'}
  if excitation not in valid_excitations:
    print_f(f"ERROR: Invalid excitation type '{excitation}'. Must be one of: {', '.join(valid_excitations)}")
    comm.Abort(1)


def check_material(material):
  """
  Check if the material dimensionality is valid.
  """
  valid_material = {'3d', '2d', '1d', '0d'}
  if material not in valid_material:
    print_f(f"ERROR: Invalid material '{material}'. Must be one of: {valid_material}")
    comm.Abort(1)


def check_file_exists(filename):
  """
  Check if a file exists and print an error message if it does not.
  """
  if not os.path.exists(filename):
    print_f(f"ERROR: File not found: {filename}")
    comm.Abort(1)


def check_wannier_io(bse):
  """
  Check if the wannier90 input/output files are specified correctly.
  """
  valid_wannier = {'wannier90'}
  if bse["wannier_io"] is None:
    print_f("ERROR: 'wannier_io' must be specified.")
    comm.Abort(1)
  else:
    if bse["wannier_io"]["engine"] not in valid_wannier:
      print_f(f"ERROR: Invalid engine '{bse['wannier_io']['engine']}'. Must be one of: {valid_wannier}")
      comm.Abort(1)
    check_file_exists(bse["wannier_io"]["hr_file"])
    check_file_exists(bse["wannier_io"]["wsvec_file"])
    check_file_exists(bse["wannier_io"]["wout_file"])
    check_file_exists(bse["wannier_io"]["win_file"])
  

def check_bse(bse):
  """
  Check if the BSE type is valid.
  """
  valid_bse = {'dft', 'tb'}
  # Validate bse method
  if bse["method"] not in valid_bse:
    print_f(f'ERROR: Invalid BSE type {bse["method"]}. Must be one of: {valid_bse}')
    comm.Abort(1)
  else:
    if bse["method"] == 'dft':
      # Validate dft eigenvalues
      if bse["dft"]["bands"] is None:
        print_f("ERROR: 'bands' must be specified for dft bse method.")
        comm.Abort(1)
      else:
        valid_dft_engine = {'vasp', 'quantum_espresso', 'siesta'}
        if bse["dft"]["bands"]["engine"] not in valid_dft_engine: 
          print_f(f'ERROR: Invalid engine {bse["dft"]["bands"]["engine"]}. Must be one of: {valid_dft_engine}')
          comm.Abort(1)
        check_file_exists(bse["dft"]["bands"]["file"])

      # Validate wannier90 eigenvectors/cnmk
      if bse["dft"]["cnmk"] is None:
        print_f("ERROR: 'cnmk' must be specified for dft bse method.")
        comm.Abort(1)
      else:
        valid_dft_engine = {'wannier90'}
        if bse["dft"]["cnmk"]["engine"] not in valid_dft_engine: 
          print_f(f'ERROR: Invalid engine {bse["dft"]["cnmk"]["engine"]}. Must be one of: {valid_dft_engine}')
          comm.Abort(1)
        check_file_exists(bse["dft"]["cnmk"]["file"])

    elif bse["method"] == 'tb':
      # Validate only tb 
      if bse["tb"]["bands"] is None:
        print_f("ERROR: 'bands' must be specified for tb bse method.")
        comm.Abort(1)
      valid_tb_engine = {'wannier90'}
      if bse["tb"]["bands"]["engine"] not in valid_tb_engine: 
        print_f(f'ERROR: Invalid engine {bse["tb"]["bands"]["engine"]}. Must be one of: {valid_tb_engine}')
        comm.Abort(1)

def check_eh_interaction(eh_interaction):
  """
  Check if the electron-hole interaction type is valid.
  """
  if eh_interaction is None:
    print_f("ERROR: 'eh_interaction' must be specified.")
    comm.Abort(1)
  else:
    valid_space = {'real', 'reciprocal'}
    if eh_interaction["space"] not in valid_space:
      print_f(f'ERROR: Invalid space {eh_interaction["space"]}. Must be one of: {valid_space}')
      comm.Abort(1)
    else:
      if eh_interaction["space"] == 'real':
        # Validate real space potential
        valid_potential = {'keldysh'}
        if eh_interaction["real"]["potential"] not in valid_potential:
          print_f(f'ERROR: Invalid potential {eh_interaction["real"]["potential"]}. Must be one of: {valid_potential}')
          comm.Abort(1)
      elif eh_interaction["space"] == 'reciprocal':
        # Validate reciprocal space potential
        if eh_interaction["reciprocal"]["file"] is None:
          print_f("ERROR: 'file' must be specified for reciprocal space eh interaction.")
          comm.Abort(1)
        check_file_exists(eh_interaction["reciprocal"]["file"])
    if not eh_interaction["include"]["direct"]:
      print_f("WARNING: Direct interaction is not included!")
    if not eh_interaction["include"]["exchange"]:
      print_f("WARNING: Exchange interaction is not included!")
  return None 

def check_absorption(absorption):
  """
  Check if the absorption type is valid.
  """
  valid_absorption = {'unpolarized', 'polarized', 'spinorbit'}
  if absorption["spin"] not in valid_absorption:
    print_f(f'ERROR: Invalid absorption type {absorption}. Must be one of: {valid_absorption}')
    comm.Abort(1)
  elif absorption["spin"] == 'polarized':
    print_f("WARNING: Run twice (like unpolarised, but for up and down spin)!")
    print_f("Change eigenvalues and eigenvectors for spin up and down!")
  elif absorption["spin"] == 'spinorbit':
    # Validate spinorbit absorption
    valid_soc_type = {'perturbation', 'full'}
    if absorption["spinorbit"]["type"] not in valid_soc_type:
      #print_f(f"ERROR: Invalid spinorbit type '{absorption["spinorbit"]["type"]}'. Must be one of: {', '.join(valid_soc_type)}")
      comm.Abort(1)
    elif absorption["spinorbit"]["type"] == 'perturbation':
      if absorption["spinorbit"]["unfold"]:
        print_f("WARNING: Unfolding is not implemented for perturbation spinorbit absorption.")
        comm.Abort(1)
      else:
        check_file_exists(absorption["spinorbit"]["band_file_unpolarized"])
        check_file_exists(absorption["spinorbit"]["band_file_soc"])
  if absorption["photon_energy"] is None:
    print_f("ERROR: 'photon_energy' must be specified.")
    comm.Abort(1)


def check_system(system):
  """
  Check if the system parameters are valid.
  """
  valid_system = {'cpu', 'gpu'}
  if system not in valid_system:
    #print_f(f"ERROR: Invalid system type '{system}'. Must be one of: {', '.join(valid_system)}")
    comm.Abort(1)
  elif system == 'gpu':
    print_f("WARNING: GPU support is experimental and may not work as expected.")
    comm.Abort(1)

def check_diagonalization(diagonalize):
  """
  Check if the diagonalization parameters are valid.
  """
  valid_library = {'primme', 'lapack', 'elpa', 'slepc'}
  if diagonalize["library"] not in valid_library:
    #print_f(f"ERROR: Invalid diagonalization library '{diagonalize["library"]}'. Must be one of: {', '.join(valid_library)}")
    comm.Abort(1)
