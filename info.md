## PyMEX: Python package for Moiré EXciton calculations

* Files in [src](./src)

1. `pymex_tb.yaml`: Input configuration file (replaces the old `pymex.inp`/
                   `pymex_detailed.inp` format).
2. `read_yaml.py`: Reads and parses the YAML configuration file.
3. `wan90tobse.py`: Conversion of Wannier90 outputs to PyMEX input format.
4. `potential.py`: Real-space electron-hole interactions (and Fourier
                  Transforms, etc.)
5. `dft2bse.py`: DFT output to PyMEX conversion
                (BandLines are used in SIESTA; QE is also supported).
                Only needed for "dft" method. "tb" method does not need
                this.
6. `function.pxd`: External calls for Cython.
                  Helpful for High-Performance-Computing architectures.
7. `generic_func.py`: A few generic utility functions.
                      Includes transfer matrix via sympy. 
8. `cyfunc.pyx`: Most expensive for-loops are cythonized for speed.
9. `constants.py`: Some physical constants.
10. `bse.py`: Core module solving the Bethe-Salpeter Equation; computes
             exciton eigenvalues, eigenvectors, and related properties.
11. `calc_Ham.py`: Driver script to build and diagonalize the exciton
                  Hamiltonian. 
                  Can be broken to multiple parts to take adavantage of
                  different parallelization schemes for different functions.
                  See Examples for illustrations.
12. `addsoc.py`: Adds spin-orbit coupling contributions. Requires work to 
                 translate from the deprecated to new version. 
13. `write_pmu_file.py`: Same as addsoc.py 
14. `setup.py`: Build script for compiling the Cython extensions
               (`python setup.py build_ext --inplace`).
               User/use-cases dependent. Tune for your laptop/HPC. 
15. Additional post-processing analysis can be found in the Utility folder.

