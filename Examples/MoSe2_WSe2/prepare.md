**Author: Indrajit Maity  
Email: indrajit.maity02@gmail.com**

Example: Solving the BSE for a MoSe₂/WSe₂ Heterobilayer with PyMEX.

Note: Spin–orbit coupling is included in this example. The spinor wavefunctions are explicitly used to construct and solve the spinor-BSE. In this example, we use Quantum Espresso to set up Wannier calculations. 

## Steps for the calculations
### Wannierization (Five steps)

1. **SCF** (Run `scf.sh`): self-consistent field run to converge the charge density.
2. **NSCF** (Run `nscf.sh`): non-self-consistent run on a uniform, unshifted k-grid.
3. **Wannier90 pre-processing** (Run `wannier_pp.sh`): `wannier90.x -pp` generates the `.nnkp` file.
4. **PW2Wannier90** (Run `pw2wan.sh`): computes overlap (`Mmn`) and projection (`Amn`) matrices from the `.nnkp` file.
5. **Wannier90** (`wannier.sh`): disentanglement + localization, producing the maximally localized Wannier functions.

Alternatively, if you configure the `submit_chain.sh` file, you can run them with a single step. 

```bash
./submit_chain.sh
```
Runs steps 1–5 as a SLURM dependency chain, each stage starting only after
the previous one completes successfully.

## Final run

Once step 5 has converged, comment out the following lines in `seedname.win`:

```
restart = default
bands_plot = .true.
write_u_matrices = .true.
write_hr = .true.
```

Then resubmit `wannier.sh` once more for the final production run.


### Bethe-Salpeter-Equation (3 steps) 

6. Copy the necessary files to [01_BSE](./01_BSE) folder and create the BSE Hamiltonian and diagonalize; Take a look at `pymex_tb.yaml` inside the folder. Set it up as needed and copy the necessary files. You can achieve that with the following:

*cd ../BSE*  
*ln -s ../00_Wannier/WSe2_u.mat ./*  
*ln -s ../00_Wannier/WSe2_hr.dat ./*  
*ln -s ../00_Wannier/WSe2_wsvec.dat ./*  
*ln -s ../00_Wannier/WSe2.win ./*  
*ln -s ../00_Wannier/WSe2.wout ./*  
*ln -s ../00_Wannier/WSe2.bands ./*  

The lines above creates soft links for required files. n this example, we exploit the different scaling strategies available in PyMEX to efficiently solve the spinor-BSE for the MoSe₂/WSe₂ heterobilayer. 

**Parallelization and Job Workflow**

The calculation is split into three SLURM jobs:

* **Job 1 — `write_H`**: uses **hybrid MPI + OpenMP** parallelization.
* **Job 2 — `diagon`**: uses **ELPA** for the diagonalization and is run with **MPI only**. We experimented with OpenMP threading for this step, but encountered segmentation faults in most cases. ELPA threading has not yet been rigorously tested, so **MPI-only execution is currently recommended**.
* **Job 3 — `optical`**: uses **hybrid MPI + OpenMP** parallelization.

The detailed resource specifications and execution commands are provided in the corresponding SLURM scripts. In stead of 3 steps, you can get your results in one step with `submit_all.sh`. 

### ELPA dependency

The ELPA diagonalization requires **`pyelpa`**, which must be installed separately and available in the Python environment used to run PyMEX.

All Done!! 
