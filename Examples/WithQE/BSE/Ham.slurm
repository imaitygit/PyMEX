#!/bin/bash --login

#SBATCH --job-name=pymex
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=81
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

#SBATCH --account=e05-nano-lis
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --reservation=shortqos

# Load the Python module
module load PrgEnv-gnu
module load cray-python
module load cray-hdf5-parallel

# Instead of virtual environment try 
VE=/work/e05/e05/imaity/venv_updated
source $VE/bin/activate

# Build cythonized funcs using setup.py
PYMEXSRC="/work/e05/e05/imaity/codes/pymex/src"
python3 ${PYMEXSRC}/setup.py build_ext --inplace
export PATH=${PYMEXSRC}/build:$PATH

export OMP_NUM_THREADS=1

# Parallel-BSE done in two-steps
# With more processors get BSE eigenvalues
# and eigenvectors very quickly.
srun python3 calc_Ham.py >& bse_out

# Compute the exciton BSE wave-functions in real space
#srun python3 calculations.py >& wfn_out
#mkdir datafiles; mv EX_*.npy pos*.npy R*.npy datafiles

# Get the spectra with a fewer processors
#srun python3 calculations.py >& sigma_out
