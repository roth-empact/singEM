import numpy as np
import sys
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM")
from ReadAbaqusMesh import readAbq

# Creates a file containing a set of points to evaluate the fields at for the IBMSurfPart mesh
# This version of the code picks along the whole planar surface

# rectDims = np.array([0, 5e-4, 1e-4]) # Normal to x
rectLims = np.array([[0.310, -1.0, 1.1e-4], [1.26, 0.10, 1.1e-4]], dtype=np.float64)
rectDims = rectLims[1] - rectLims[0]
centerPt = (rectLims[1] + rectLims[0]) / 2
normDir = np.where(rectDims == 0)[0][0] # Direction normal to the rectangle
tanDirs = np.where(rectDims != 0)[0]
numPts = 1001 # Number of points in each direction
pts = centerPt.reshape(3,1) + np.outer(rectDims, np.linspace(-0.5 + 1e-6, 0.5 - 1e-6, numPts))
totPoints = np.zeros((numPts**2,3), dtype=np.float64)
for ii in range(numPts):
  totPoints[ii*numPts:(ii+1)*numPts, normDir] = pts[normDir, 0]
  totPoints[ii*numPts:(ii+1)*numPts, tanDirs[0]] = pts[tanDirs[0]]
  totPoints[ii*numPts:(ii+1)*numPts, tanDirs[1]] = pts[tanDirs[1], ii]

# rectLength, rectWidth = [5e-2, 5e-2] # Units are mm
# numX, numY = [101, 101] # Number of points in each direction
# xPts = targetNodeCoords[0] + np.linspace(-rectLength/2, rectLength/2, numX)
# yPts = targetNodeCoords[1] + np.linspace(-rectWidth/2, rectWidth/2, numY)
# if np.isclose(targetNodeCoords[2], 0.0, atol=1e-5):
#   # Surface is the bottom of the metal, so shift down to avoid being directly on it
#   zVal = targetNodeCoords[2] - 1.0e-8 
# elif np.isclose(targetNodeCoords[2], 1.0e-4, atol=1e-5):
#   # Surface is the top of the metal, so shift up to avoid being directly on it
#   zVal = targetNodeCoords[2] + 1.0e-8 

# totPoints = np.zeros((numX*numY,3), dtype=np.float64)
# for ii in range(numX):
#   totPoints[ii*numY:(ii+1)*numY, 0] = xPts[ii]
#   totPoints[ii*numY:(ii+1)*numY, 1] = yPts
#   totPoints[ii*numY:(ii+1)*numY, 2] = zVal

np.save(f"./Meshes/surfPart/EvalPts/IBMSurfPartEvalPoints_plane_{numPts}points.npy", totPoints)