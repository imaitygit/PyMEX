## PyMEX Examples

1. [`monolayer_WSe2`](./monolayer_WSe2): Calculate intralayer excitons in monolayer WSe₂. This is the simplest example for 2D materials. See `prepare.md` in this example for details on generating Wannier files, setting up, and running PyMEX.

2. [`MoSe2_WSe2`](./MoSe2_WSe2): Calculate excitons in a MoSe₂/WSe₂ heterobilayer. The conduction and valence bands come from different materials, forming a type-II heterojunction. The lowest-energy excitons are interlayer excitons. See `prepare.md` in this example for details on generating Wannier files, setting up, and running PyMEX.

3. [`Moire_WSe2`](./Moire_WSe2): Calculate intralayer excitons in WSe₂ with moiré-induced atomic rearrangements. See `prepare.md` in this example for details on generating Wannier files, setting up, and running PyMEX.

4. [`MacvsHPC`](./MacvsHPC): Run PyMEX on a Mac and an HPC system and compare the two setups. See `prepare.md` in this example for details on running PyMEX locally on your laptop and on an HPC system.

Start with `monolayer_WSe2` if you're new to PyMEX, then move to the heterobilayer and moiré examples as you become familiar with the config format.

We are preparing a paper covering technical details, scaling, and more examples for arXiv submission in December 2026.