import numpy as np
import sys
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM")
from basis_funcs import *
from singBasis_wedges import *
import gauss_quadrature_tetra as q
from gaussQuadratureSingular import *
from assemble_mats import getWedgeLocs
import matplotlib.pyplot as plt

## Validate that the curls of basis functions are correct using finite differencing

basisOrder = 2
basisOrder_long = 2

ep0 = 1e-9 / (36 * np.pi)
mu0 = 4e-7 * np.pi
coords = np.array([[1,0,0], [0,1,0], [0,0,0], [1,0,1], [0,1,1], [0,0,1]], dtype=np.float64)
vertDir = 2
# coords = np.load("./Test_Files/Singular_testing/wedgeCoords_v2.npy")
globNodeNums = np.arange(6, dtype=np.int32)

center_loc = np.sum(coords[:3], axis=0)/3 # Center of the triangle
test_loc = center_loc + coords[0]/10 - coords[1]/10 # Offset slightly from the center
fd_shift = 1e-9
triLoc = np.zeros((3,3), dtype=np.float64)
triLoc[0] = test_loc
for ii in range(1, 3):
  triLoc[ii] = test_loc
  triLoc[ii,ii-1] += fd_shift
vertLoc = np.zeros((2,), dtype=np.float64)
vertLoc[0] = (coords[0,vertDir] + coords[3,vertDir]) / 2 # Vertical center of the prism
vertLoc[0] += (coords[0,vertDir] + coords[3,vertDir]) / 10 # Offset slightly from the center
vertLoc[1] = vertLoc[0] + fd_shift

if np.all((basisOrder,basisOrder_long) == (2,2)):
  vecBasis, curlBasis = basis3DVWedge_2x2(coords, globNodeNums, triLoc, vertLoc, vertDir)
elif np.all((basisOrder,basisOrder_long) == (2,1)):
  vecBasis, curlBasis = basis3DVWedgeQuad(coords, globNodeNums, triLoc, vertLoc, vertDir)

# Evaluate the curls numerically from the values of the function
curlBasis_numeric = np.zeros((len(curlBasis), 3), dtype=np.float64)

for ii in range(len(vecBasis)):
  dx = (vecBasis[ii,1] - vecBasis[ii,0])/fd_shift
  dy = (vecBasis[ii,2] - vecBasis[ii,0])/fd_shift
  dz = (vecBasis[ii,3] - vecBasis[ii,0])/fd_shift
  curlBasis_numeric[ii] = np.array([dy[2]-dz[1], dz[0]-dx[2], dx[1]-dy[0]])
curlBasis_error = np.linalg.norm(curlBasis[:,0,:] - curlBasis_numeric, axis=1)/np.linalg.norm(curlBasis_numeric, axis=1)

print(f"Relative error in the curls of the singular basis functions: {curlBasis_error}")
print(f"Max relative error in the curls: {np.max(curlBasis_error)}")
