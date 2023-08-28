import numpy as np
import matplotlib.pyplot as plt
from dft2bse import *
import h5py
from dft2bse import *


# SOC as a perturbation

f1 = h5py.File("E_ph.hdf5", 'r')
E_ph_np = np.array(f1['photon'])
f1.close()

f2 = h5py.File("SIGMA_full.hdf5", 'r')
sigma_xx_np = np.array(f2['sigma_xx'])
f2.close()

f1 = h5py.File("E_ph_per.hdf5", 'r')
E_ph = np.array(f1['photon'])
f1.close()

f2 = h5py.File("SIGMA_per_unit.hdf5", 'r')
sigma_xx = np.array(f2['sigma_xx'])
f2.close()

f1 = h5py.File("E_ph_per_pmu.hdf5", 'r')
E_ph_pmu = np.array(f1['photon'])
f1.close()

f2 = h5py.File("SIGMA_per_pmu.hdf5", 'r')
sigma_xx_pmu = np.array(f2['sigma_xx'])
f2.close()

plt.plot(E_ph_np-np.min(E_ph_np), sigma_xx_np,\
         color="r", lw=3, ls="--", label="Non-polarized")

plt.plot(E_ph-np.min(E_ph), sigma_xx,\
         color="b", lw=3, ls="--", label="Perturbation")

plt.plot(E_ph_pmu-np.min(E_ph_pmu), sigma_xx_pmu,\
         color="c", lw=3, ls="--", label="Perturbation@PMU")

plt.legend()
plt.savefig("comparison.png", dpi=200)
plt.show()


#f1 = h5py.File("DELTA.hdf5", 'r')
#E_nsk = np.array(f1['D_nsk_unit'])
#f1.close()
#
#E_nonpol = E_siesta("WS2_unit_unpolar.bands")
#E_soc = E_siesta("WS2_unit.bands")
#
#D_nk = E_nsk.reshape(E_nsk.shape[0]*E_nsk.shape[1], E_nsk.shape[2])
#E_nk = np.zeros((D_nk.shape[0], D_nk.shape[1]))
#
#for i in range(D_nk.shape[0]):
#  for j in range(D_nk.shape[1]):
#    E_nk[i,j] = E_nonpol[i//2, j] + D_nk[i,j]
#
#plt.subplot(211)
#for i in range(13, 21):
#  plt.plot(E_nk[i,:], color="k")
#  plt.plot(E_soc[i,:]+0.01, color="g")
##plt.ylim(-6, -2)
#plt.subplot(212)
#for i in range(18, 20):
#  # i == 16, 17 VBM
#  # i = 18,19 CBM
#  if i == 18:
#    plt.plot(D_nk[i,:], color="r", label="UP_")
#  else:
#    plt.plot(D_nk[i,:], color="b", label="DOWN_")
#plt.legend()
#plt.show()
