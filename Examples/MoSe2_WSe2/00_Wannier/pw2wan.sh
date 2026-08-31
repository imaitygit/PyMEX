#!/bin/bash --login
#SBATCH --job-name=quadrupole_pw2wan
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

#SBATCH --account=e89-ic_m
#SBATCH --partition=standard
#SBATCH --qos=standard

module load epcc-job-env
module load quantum_espresso/7.3.1

echo "=== PW2WANNIER90 START: $(date) ==="

srun --distribution=block:block --hint=nomultithread --unbuffered \
     pw2wannier90.x -pd .true. -in hetero.pw2wan >& pw2wan.out

echo "=== PW2WANNIER90 DONE: $(date) ==="
