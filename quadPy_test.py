import numpy as np
import quadpy as qp
from singBasis_tets import vecBasisSingular_orderN
from assemble_mats import rotateCoords2D
from ReadAbaqusMesh import readAbq

coords = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=np.float64)
origTriCoords = coords[:3]
normVec = np.cross(origTriCoords[0] - origTriCoords[1], origTriCoords[0] - origTriCoords[2])
normVec /= np.linalg.norm(origTriCoords)
triCoords, rotMat = rotateCoords2D(origTriCoords, normVec, True)
basisOrder, singOrder = [2,3]
coeff = 1/2

def singIntEval(coords, coeff, points_2D, basisOrder, singOrder, rotMat, rotCoordZ):
  points_3D = np.vstack([points_2D, rotCoordZ*np.ones((1,points_2D.shape[1]), dtype=np.float64)])
  loc = (points_3D.T @ rotMat)
  fvals, _ = vecBasisSingular_orderN(coords, coeff, coeff, loc, basisOrder, singOrder)
  return fvals[200,:,0]*fvals[201,:,1]

minOrder = 3
maxOrder = 21
print(f"Using quadrature orders between {minOrder} and {maxOrder}")
intVals = np.zeros((maxOrder-minOrder,), dtype=np.float64)
for ii in range(minOrder, maxOrder):
  # scheme = qp.t2.get_good_scheme(ii)
  if ii < 10:
    ii_str = f"0{ii}"
  else:
    ii_str = f"{ii}"
  scheme = getattr(qp.t2._xiao_gimbutas, f"xiao_gimbutas_{ii_str}")()
  intVals[ii-minOrder] = scheme.integrate(lambda x: singIntEval(coords, coeff, x, basisOrder, singOrder, rotMat, triCoords[0,2]), triCoords[:,:2])
# Rescale integrals over this face because it has an area of sqrt(3) (the rest have area 1)
intVals /= np.sqrt(3)

print("No Splits:")
print(intVals)


##### Repeat with splitting
numSplit = 5
splitCoords, _, _, _, _, totTriList, surfStartInds, _, _, _, _ = readAbq(f"./Test_Files/Singular_testing/tetSplit_numSplit={numSplit}_tris.inp")
totTriList -= 1 # Shift by 1 to account for python vs Cubit numbering
splitCoords = splitCoords[:,1:] # Remove the node indices
splitTriList = [totTriList[surfStartInds[ii]:surfStartInds[ii+1]] for ii in range(len(surfStartInds)-1)]
splitTriList.append(totTriList[surfStartInds[-1]:len(totTriList)])

faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)
faceInd, face = [0, faceNums[0]]
rotTriCoords, rotMat = rotateCoords2D(coords[face], normVec, True)
splitLoc, splitTriCoords = [[], []]
for nodes in splitTriList[faceInd]:
  splitTriCoords.append(splitCoords[nodes] @ rotMat.T)
splitTriCoords = np.array(splitTriCoords)
numEle = len(splitTriList[0])

intVals = np.zeros((maxOrder-minOrder,), dtype=np.float64)
for ii in range(minOrder, maxOrder):
  if ii < 10:
    ii_str = f"0{ii}"
  else:
    ii_str = f"{ii}"
  scheme = getattr(qp.t2._xiao_gimbutas, f"xiao_gimbutas_{ii_str}")()
  for jj in range(numEle):
    intVals[ii-minOrder] += scheme.integrate(lambda x: singIntEval(coords, coeff, x, basisOrder, singOrder, rotMat, rotTriCoords[0,2]), splitTriCoords[jj,:,:2])

# Rescale integrals over this face because it has an area of sqrt(3) (the rest have area 1)
intVals /= np.sqrt(3)
print(f"{numSplit} splits:")
print(intVals)