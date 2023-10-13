**Author: Indrajit Maity  
Email: indrajit.maity02@gmail.com**

Example to solve the BSE with PyMEX for a relatively coarse grid  
NOTE: The spin-orbit coupling is included in the electronic
      structure calculations explicitly. This is a non-perutbative
      calculation. For a perturbative treatment, see the example 
      [WithSiesta](../WithSiesta).

      YOU MUST change the number of orbital within the
      `wan90tobse.py` to 10 for W and 6 for Se. Will be automated
      in the future! 

## Steps for the calculations
### WANNIERIZE (5 Steps)

1. Run Wannier90 to create the k-grid within the [WANNIERIZE](./WANNIERIZE) 
folder;
 
*cd WANNIERIZE*  
*PATH-2-WAN90/utility/kmesh.pl 9 9 1 >> kpoints_qe*
*PATH-2-WAN90/utility/kmesh.pl 9 9 1 wannier >> kpoints_wannier*

You will find 81 kpoints written in `kpoints_wannier` and
`kpoints_qe` file. We will utilize these k-points in all out future
calculations.


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

3. Quantum ESPRESSO (QE) calculations for generating inputs of
Wannier90; 

*PATH-2-QE/pw.x -pd .true. -nk 3 -in WSe2.scf >& scf.out*  
*PATH-2-QE/pw.x -pd .true. -nk 3 -in WSe2.nscf >& nscf.out*  
*PATH-2-QE/bands.x -pd .true. -in bands.in*  
*PATH-2-QE/pw2wannier90.x -pd .true. -in WSe2.pw2wan >& pw2wan.out*  

QE calculations with inputs required for Wannier90. Please 
take a look and familiarize yourself with the input and the 
keywords required to generate the Wannier90 input. Also, 
we are using 9x9x1 k-grid for the SCF calculations. 


4. Wannier90 one-shot projections;

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

### Bethe-Salpeter-Equation (3 steps) 

6. Copy the necessary files to [BSE](./BSE) folder and create the 
BSE Hamiltonian and diagonalize; 

*cd ../BSE*  
*ln -s ../WANNIERIZE/WSe2_u.mat ./*  
*ln -s ../WANNIERIZE/WSe2_hr.dat ./*  
*ln -s ../WANNIERIZE/WSe2_wsvec.dat ./*  
*ln -s ../WANNIERIZE/WSe2.win ./*  
*ln -s ../WANNIERIZE/WSe2.wout ./*  
*ln -s ../WANNIERIZE/Bands_QE ./*  
*python3 PATH-2-PYMEX-SRC/setup.py build_ext --inplace*  
*export PATH=${PYMEXSRC}/build:$PATH*  
*mpirun -np numprocess python3 calc_Ham.py >& bse_out*

The lines above creates soft links for required files, 
complies the source codes (for cythonized part), and 
runs the BSE Hamiltonian construction. 
**NOTE:** Ideally, the numprocess can be anything <= 
Numberofkpoints x NumberofValencebands x NumberofCondbands; 
In this case, it is= 9 x 9 x 2 x 2 = 324. 
However, we found the h5py-parallel works only for a integer 
divisor of Numberofkpoints x Numberofbands. Therefore,
the numprocess here can be 1, 2, 3, 4, 6, etc. upto 324. 
However, if you enter 5 as numprocess the code will throw
an error! 


7. Perform the optical conductivity calculations. 

*python3 PATH-2-PYMEX-SRC/setup.py build_ext --inplace*  
*export PATH=${PYMEXSRC}/build:$PATH*  
*mpirun -np numprocess python3 calc_Absorb.py >& bse_out*  


All Done!! You can now analyze the data and do cool things with 
it :)

