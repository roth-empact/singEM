import numpy as np
import sys
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM")
from ReadAbaqusMesh import readAbq

# Creates a file containing a set of points to evaluate the fields at for the IBMSurfPart mesh
# This version of the code picks points in a rectangular region surrounding a particular node
meshFile = "./Meshes/surfPart/IBMSurfPart.inp"
nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, surfStartInds, lineList, lineStartInds, nodesetList, nodesetStartInds = readAbq(meshFile)
tetList -= 1 # -1 to match python indexing
tetList = np.sort(tetList, axis=1) # Sort node numbers in ascending order in each element

# targetNode = 89 # Along one of the top edges of the pad
# targetNode = 491 # At the same position as 89, but along the bottom face
# targetNode = 19 # At a corner of the top side of the pad
targetNode = 366 # At the same position as 19, but along the bottom face
# targetNode = 2 # Corner of the port
targetNodeCoords = nodeList[targetNode, 1:]
targetNodeCoords[0] -= 1e-8 # Offset so it's not directly at the interface
targetNodeCoords[1] += 1e-8
targetNodeCoords[2] -= 1e-8
# targetNodeCoords[0] -= 2.5e-2 + 1e-5 # Offset so it's not on the PEC

# rectDims = np.array([0, 5e-4, 1e-4]) # Normal to x
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

np.save("qubitSurfPartEvalPoints_botCorner.npy", totPoints)