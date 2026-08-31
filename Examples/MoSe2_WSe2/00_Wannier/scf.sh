#!/bin/bash --login
#SBATCH --job-name=quadrupole_scf
#SBATCH --nodes=2
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

#SBATCH --account=e89-ic_m
#SBATCH --partition=standard
#SBATCH --qos=standard

module load epcc-job-env
module load quantum_espresso/7.3.1

echo "=== SCF START: $(date) ==="

srun --distribution=block:block --hint=nomultithread --unbuffered \
     pw.x -pd .true. -nk 4 -in hetero.scf >& scf.out

echo "=== SCF DONE: $(date) ==="
