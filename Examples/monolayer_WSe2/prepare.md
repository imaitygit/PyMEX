**Author: Indrajit Maity  
Email: indrajit.maity02@gmail.com**

Example to solve the BSE with PyMEX for a relatively coarse grid. 
NOTE: The spin-orbit coupling is not included here. You can skip the Wannierisation steps altogether if you are familiar with it already. 

## Steps for the calculations
### Wannierization (5 Steps)

1. Run Wannier90 to create the k-grid within the [00_Wannier](./00_Wannier) 
folder;
 
*cd 00_Wannier*  
*PATH-2-WAN90/utility/kmesh.pl 15 15 1 wannier >> kpoints_wannier*

You will find 225 kpoints written in `kpoints_wannier` file. We
will utilize these k-points in all out future calculations.


2. Run Wannier90 to generate the `WSe2.nnkp` file; 

*PATH-2-WAN90/wannier90.x -pp WSe2*

You will find `WSe2.nnkp` and other files as output. 

If you are not familiar with WANNIER90 input, please take a look 
and make sure it makes sense. At this stage make sure the 
following lines are commented out (i.e., the use of `!`):
`
!restart = default
!bands_plot = true
!write_u_matrices = .true
!write_hr = .true
!wannier_plot = .true.
!wannier_plot_supercell = 3
`

3. SIESTA calculations for generating inputs of Wannier90; 

*PATH-2-SIESTA/siesta WSe2.fdf >& WSe2.out*

SIESTA calculations with inputs required for Wannier90. Please 
take a look and familiarize yourself with the input and the 
keywords required to generate the Wannier90 input. Also, 
we are using 9x9x1 k-grid for the SCF calculations. 


4. Wannier90 one-shot projections;

*cp WSe2.eigW WSe2.eig*  
*PATH-2-WAN90/wannier90.x WSe2*

5. Wannier90 data to necessary files; Before you run `wannier90.x`
executable please uncomment the following lines in `WSe2.win`:
`
restart = default
bands_plot = true
write_u_matrices = .true
write_hr = .true
`

*PATH-2-WAN90/wannier90.x WSe2*

### Bethe-Salpeter-Equation (2 steps) 

6. Copy the necessary files to [01_BSE](./01_BSE) folder and create the BSE Hamiltonian and diagonalize; Take a look at `pymex_tb.yaml` inside the folder. Set it up as needed and copy the necessary files. At the moment, all the files are in `01_BSE`, so you can directly run these calculations. Moreover, if you want to see how to run the same calculations on your Mac/Linux versus HPC supercomputers, such as ARCHER2 in the UK, take a look at the example [`MacvsHPC`](../MacvsHPC).

*cd ../BSE*  
*ln -s ../00_Wannier/WSe2_u.mat ./*  
*ln -s ../00_Wannier/WSe2_hr.dat ./*  
*ln -s ../00_Wannier/WSe2_wsvec.dat ./*  
*ln -s ../00_Wannier/WSe2.win ./*  
*ln -s ../00_Wannier/WSe2.wout ./*  
*ln -s ../00_Wannier/WSe2.bands ./*  
*python3 PATH-2-PYMEX-SRC/setup.py build_ext --inplace*  
*export PATH=${PYMEXSRC}/build:$PATH*  
*mpirun -np numprocess python3 calc_all.py >& pymex_out*

The lines above creates soft links for required files, complies the source codes (for cythonized part), and  runs the BSE Hamiltonian construction. 


All Done!! 
