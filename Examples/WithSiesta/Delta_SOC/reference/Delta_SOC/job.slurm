#!/bin/bash --login
#SBATCH --job-name=wse2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=e05-nano-lis
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --reservation=shortqos


# Ensure that the environment is set correctly for the job
module load epcc-job-env

# Load the xthi module to get access to the xthi program
#module load xthi

# srun launches the parallel program based on the SBATCH options
# The SBATCH options are provided above
# This is an "only MPI" job.

EXECDIR="/work/e05/e05/imaity/codes/siesta/cpe_21.03_test/Obj_mpi"
BANDS="/work/e05/e05/imaity/codes/siesta/cpe_21.03_test/Util/Bands"
WAN90="/work/e05/e05/imaity/codes/wannier90-3.1.0/bin"

# Run SIESTA for unpolar calculations
srun --distribution=block:block --hint=nomultithread --unbuffered ${EXECDIR}/siesta WSe2_unpol.fdf >& WSe2_unpolar.out 
echo "Moving the bands file"
mv WSe2.bands WSe2_unit_unpolar.bands

# Run SIESTA for band-structure
#srun --distribution=block:block --hint=nomultithread --unbuffered ${BANDS}/gnubands WSe2_unit_unpolar.bands 

# Run SIESTA for soc calculations
srun --distribution=block:block --hint=nomultithread --unbuffered ${EXECDIR}/siesta WSe2.fdf >& WSe2.out 
echo "Moving the bands file"
mv WSe2.bands WSe2_unit.bands

# Run SIESTA for band-structure
#srun --distribution=block:block --hint=nomultithread --unbuffered ${BANDS}/gnubands WSe2_unit.bands 
