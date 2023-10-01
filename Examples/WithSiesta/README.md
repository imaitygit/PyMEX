Author: Indrajit Maity
Email: indrajit.maity02@gmail.com

## Example to solve the BSE with PyMEX for a relatively coarse grid
## NOTE: The spin-orbit coupling is included perturbatively.

**Steps for the calculations**

1. Run Wannier90 to create the k-grid;
 
*PATH-2-WAN90/utility/kmesh.pl 9 9 1 wannier >> kpoints_wannier*

You will find 81 kpoints written in `kpoints_wannier` file. We
will utilize these k-points in all out future calculations.


2. Run Wannier90 to generate the `WSe2.nnkp` file; 

*PATH-2-WAN90//wannier90.x -pp WSe2*

You will find `WSe2.nnkp` and other files as output. 

If you are not familiar with WANNIER90 input, please take a look 
and make sure it makes sense. At this stage make sure the 
following lines are commented out:
`
!restart = default
!bands_plot = true
!write_u_matrices = .true
!write_hr = .true
!wannier_plot = .true.
!wannier_plot_supercell = 3
`
