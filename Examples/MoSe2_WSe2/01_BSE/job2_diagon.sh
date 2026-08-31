#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=pymex_diagon
#SBATCH --nodes=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --account=e89-ic_m

#SBATCH --partition=standard
#SBATCH --qos=short

# Modules
module load PrgEnv-gnu/8.4.0
module load cray-python/3.10.10
module load cray-hdf5-parallel/1.12.2.7
module load petsc/3.18.5
module load slepc/3.18.3
module load cray-mpich/8.1.27

# Environment
source /work/e89/e89/imli/codes/venv_dec25/bin/activate
export PYTHONPATH=/work/e89/e89/imli/codes/elpa-2025.06.002_omp/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/work/e89/e89/imli/codes/elpa-2025.06.002_omp/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "OMP_NUM_THREADS     = $OMP_NUM_THREADS"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"

# Run
srun --distribution=block:block --hint=nomultithread \
     python3 calc_diagon.py &>> pymex_plus_out
