#!/bin/bash --login
#SBATCH --job-name=quadrupole_wanpp
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

#SBATCH --account=e89-ic_m
#SBATCH --partition=standard
#SBATCH --qos=short

module load epcc-job-env
module load quantum_espresso/7.3.1

WAN90="/work/e89/e89/imli/codes/wannier90-3.1.0/bin"

echo "=== WANNIER90 -pp START: $(date) ==="

srun --distribution=block:block --hint=nomultithread --unbuffered \
     ${WAN90}/wannier90.x hetero

echo "=== WANNIER90 -pp DONE: $(date) ==="
