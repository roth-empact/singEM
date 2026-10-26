import numpy as np
import sys
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM")
from ReadAbaqusMesh import readAbq

# Creates a file containing a set of points to evaluate the fields at for the qubit geometry
targetNodeCoords = np.array([-0.141, 0, 1e-4]) # Top edge of the left pad
targetNodeCoords[0] -= 1e-8 # Offset so it's not directly at the interface
targetNodeCoords[2] -= 1e-8

rectDims = np.array([5e-2, 5e-2, 0])
normDir = np.where(rectDims == 0)[0][0] # Direction normal to the rectangle
tanDirs = np.where(rectDims != 0)[0]
numPts = 101 # Number of points in each direction
pts = targetNodeCoords.reshape(3,1) + np.outer(rectDims, np.linspace(-1/2, 1/2, numPts))
totPoints = np.zeros((numPts**2,3), dtype=np.float64)
for ii in range(numPts):
  totPoints[ii*numPts:(ii+1)*numPts, normDir] = pts[normDir, 0]
  totPoints[ii*numPts:(ii+1)*numPts, tanDirs[0]] = pts[tanDirs[0]]
  totPoints[ii*numPts:(ii+1)*numPts, tanDirs[1]] = pts[tanDirs[1], ii]

np.save("./Meshes/Qubit/EvalPts/qubitEvalPoints_topEdge.npy", totPoints)