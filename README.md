
```
     _______              ____    ____   ________   ____  ____
    |_   __ \            |_   \  /   _| |_   __  | |_  _||_  _|
      | |__) |   _   __    |   \/   |     | |_ \_|   \ \  / /
      |  ___/   [ \ [  ]   | |\  /| |     |  _| _     > `' <
     _| |_       \ '/ /   _| |_\/_| |_   _| |__/ |  _/ /'`\ \_
    |_____|    [\_:  /   |_____||_____| |________| |____||____|
                \__.'
```

## 🏋️ Package
**PyMEX**: Python package for Moiré EXcitons

## 📖 Description
**PyMEX** is a Python package designed to solve the
*Bethe-Salpeter Equation (BSE)* for exciton properties in moiré systems. It
leverages *Wannier functions* as a basis to compute moiré excitons
efficiently.
 
In the Wannier function basis, the BSE Hamiltonian can be approximated by using
the localized and orthogonal nature of Wannier functions, along with the
translational invariance of Coulomb interactions:

![BSE Hamiltonian](./images/BSE_zero.png)

Please note that the released code is specifically designed for calculating
zero-momentum excitons. However, it is straightforward to implement finite
momentum excitons.

## 🚀 Features

### Scientific Features
- Compute zero-momentum **BSE eigenvalues** and **eigenvectors**  
- Calculate **optical conductivity**  
- Compute **excitonic wavefunctions**  

### Performance Optimizations
- Hybrid implementation using **Python** and **Cython** for efficient
  **looping** and performance enhancement.
- Optimized for parallel computing using **MPI** and **OpenMP**.
- Efficient memory management with **h5py-parallel**.

## 📂 Directory Structure

```
root/
├── src/              # Source code
├── Examples/         # Examples
├── Utility/          # Utility scripts
├── docs/             # Documentation (experimental)
├── README.md         # Project details
├── info.md           # Minimum to get started
└── .gitignore        # Git ignore rules
```

## 🛠️ Installation
**PyMEX** has been tested on Python 3.8, 3.9, and 3.11.

### Requirements:
- numpy  
- scipy  
- matplotlib (optional)  
- cython  
- mpi4py  
- h5py-parallel  

We will be moving to a pip-installable version soon. For now, please install
the dependencies manually.

## 📬 Support
If you have any questions or encounter any bugs, please feel free to reach out.
You can contact us via the following email:
[indrajit.maity02@gmail.com](mailto:indrajit.maity02@gmail.com)

## ⌨️ Authors
This package is written and maintained by **Indrajit Maity**. If you use this
package or any part of the source code, please cite the following paper for
which this code was developed:

**Atomistic theory of twist-angle dependent intralayer and interlayer exciton
properties in twisted bilayer materials** [arXiv](https://arxiv.org/abs/2406.11098) [Peer-reviewed@NPJ2DMaterials&Apps.](https://doi.org/10.1038/s41699-025-00538-4)  

### Citation:

```bibtex
@Article{Maityatomistic2025,
author={Maity, Indrajit
and Mostofi, Arash A.
and Lischner, Johannes},
title={Atomistic theory of twist-angle dependent intralayer and interlayer exciton properties in twisted bilayer materials},
journal={npj 2D Materials and Applications},
year={2025},
month={Mar},
day={04},
volume={9},
number={1},
pages={20},
issn={2397-7132},
doi={10.1038/s41699-025-00538-4},
url={https://doi.org/10.1038/s41699-025-00538-4}
}
```

## ⭐ Acknowledgments

- Special thanks to **Prof. Johannes Lischner** and **Prof. Arash Mostofi** for
  their continuous support and encouragement.
- This work was supported by funding from the **European Union’s Horizon 2020**
  research and innovation program under the **Marie Skłodowska-Curie Grant
  Agreement No. 101028468**.

