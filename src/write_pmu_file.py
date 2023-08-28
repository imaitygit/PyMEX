import numpy as np


# For conduction band manifolds
# nc : Number of conduction bands
# nk : Number of k-points
# s : spin = 2 
nc =  1; nk = 81; s = 2
Delta_csk = np.zeros((nc,s,nk),dtype=float)
for i in range(Delta_csk.shape[0]):
  for j in range(Delta_csk.shape[2]):
    Delta_csk[i,0,j] = -0.02
    Delta_csk[i,1,j] = 0.02

# For Valence band manifolds
# nv : Number of conduction bands
# nk : Number of k-points 
# s : spin = 2 
nv =  1; nk = 81; s = 2
Delta_vsk = np.zeros((nv,s,nk), dtype=float)
# WSe2: K splits by -0.216 eV and 0.229 eV.
# WS2: K splits by -0.212 eV and 0.186 eV.
for i in range(Delta_vsk.shape[0]):
  for j in range(Delta_vsk.shape[2]):
    Delta_vsk[i,0,j] = -0.212
    Delta_vsk[i,1,j] = 0.186

Delta_f = np.stack((Delta_vsk, Delta_csk))
np.save("Moire2unit", Delta_f)
