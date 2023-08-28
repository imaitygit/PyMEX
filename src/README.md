# PyMEX: Python package for Moiré EXciton calculations
------------------------------------------------------

## Files
1. pymex.inp: Input file 
   pymex_detailed.inp: All the input files are exaplined in details.

2. read_inp.py: Reads and prints input files

3. wan90tobse.py: Conversion of Wannier90 outputs to PYMEX code.
4. potential.py: Real-space electron-hole interactions (and Fourier
                 Transforms etc.)
5. dft2bse.py: DFT output to PYMEX conversion
               (BandLines are used in SIESTA)
               QE is also supported.
6. function.pxd: External calls for cython 
                 Helpful for High-Performance-Machine arch.
7. generic_func.py: A few generic functions.
8. cyfunc.pyx: Most expensive for loops are cythonized for speed.
9. constants.py: Some constants
10. bse.py  

## ToDO

+ Implement the ability to compute finite momentum excitons;

+ Implement numerical derivative calculator so that
Gamma point is supported. [**High priority**]

+ Check Wannier Hamiltonian construction [**Very high-priority**]

+ When SOC is true and full then guess the number of orbitals.

+ Unfolding simple and possibly, siesta [**low-priority**]

+ Matrix parallelizations [**low-priority**]
