#!/bin/bash --login

#SBATCH --job-name=wse2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=e05-nano-lis
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --reservation=shortqos


# Ensure that the environment is set correctly for the job
module load epcc-job-env

# Load quantum espresso
module load quantum_espresso/6.8
WAN90="/work/e05/e05/imaity/codes/wannier90-3.1.0/bin"

# Load the xthi module to get access to the xthi program
#module load xthi

# srun launches the parallel program based on the SBATCH options
# The SBATCH options are provided above
## This is an "only MPI" job.
#srun --distribution=block:block --hint=nomultithread --unbuffered pw.x -pd .true. -nk 3 -in WSe2.scf >& scf.out
## NSCF
#srun --distribution=block:block --hint=nomultithread --unbuffered pw.x -pd .true. -nk 3 -in WSe2.nscf >& nscf.out
## Run wannier90 for .nnkp
#srun --distribution=block:block --hint=nomultithread --unbuffered ${WAN90}/wannier90.x -pp WSe2
## PW2WAN
#srun --distribution=block:block --hint=nomultithread --unbuffered pw2wannier90.x -pd .true. -in WSe2.pw2wan >& pw2wan.out
## bands
#srun --distribution=block:block --hint=nomultithread --unbuffered bands.x -pd .true. -in bands.in

# Run wannier90 to minimize spread
srun --distribution=block:block --hint=nomultithread --unbuffered ${WAN90}/wannier90.x WSe2
