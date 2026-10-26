"""
   Copyright 2026 Samuel Elkin, Ghazi Khan, Thomas Roth

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import gauss_quadrature_tetra as q
from multiprocessing import Pool, cpu_count
from itertools import repeat
from singBasis_tets import vecBasisSingular_orderN
import time
from ReadAbaqusMesh import readAbq

# Derives integrals of the field magnitude over each surface
def singFaceInts(coeff, cornerCoeff, basisOrder, singOrder, numSplit):
  # Set up integration bounds for a unit tetrahedron
  coords = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=np.float64)

  # Run the basis function generator once to get the number of functions
  tempLoc = q.gauss_quadrature_trisimp_init(9, coords[0], coords[1], coords[2])
  fvals, _ = vecBasisSingular_orderN(coords, coeff, cornerCoeff, tempLoc, basisOrder, singOrder)
  numBasis = len(fvals)

  surfInts = np.zeros((4, 6, numBasis, numBasis), dtype=np.float64) # Integrals over each of the 4 triangles
  mnDirs_num = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]], dtype=np.int32)
  faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)

  # Split the triangles into a set of smaller triangles, and define integration points on each
  splitCoords, _, _, _, _, totTriList, _, surfStartInds, _, _, _, _, _ = readAbq(f"./Test_Files/Singular_testing/tetSplit_numSplit={numSplit}_tris.inp")
  totTriList -= 1 # Shift by 1 to account for python vs Cubit numbering
  splitCoords = splitCoords[:,1:] # Remove the node indices
  splitTriList = [totTriList[surfStartInds[ii]:surfStartInds[ii+1]] for ii in range(len(surfStartInds)-1)]
  splitTriList.append(totTriList[surfStartInds[-1]:len(totTriList)])

  # Loop through the triangular faces and repeat the process on each
  for faceInd, face in enumerate(faceNums):
    splitLoc, splitTriCoords = [[], []]
    for nodes in splitTriList[faceInd]:
      splitTriCoords.append(splitCoords[nodes])
      splitLoc.append(q.gauss_quadrature_trisimp_init(9, *splitTriCoords[-1]))
    splitLoc = np.array(splitLoc)
    splitTriCoords = np.array(splitTriCoords)

    # # No splitting, just integrate over a single triangle
    # splitTriList = np.array([face])
    # splitLoc = np.array([q.gauss_quadrature_trisimp_init(9, coords[face[0]], coords[face[1]], coords[face[2]])])
    # splitTriCoords = np.array([coords[face]])
    
    p = Pool(np.min([cpu_count(), numBasis]))
    start_time = time.time()

    # Evaluate the functions at the integration locations
    numEle = len(splitTriList[0])
    for ii in range(numEle):
      fvals, _ = vecBasisSingular_orderN(coords, coeff, cornerCoeff, splitLoc[ii], basisOrder, singOrder)

      # totMats = singBasisMatrices_loop(0, numEle, numBasis, fvals, cvals, mnDirs_num, splitPrismCoords, splitVertLoc)
      curInt = list(p.starmap(singFaceInts_loop, zip(np.arange(numBasis), repeat(numBasis), repeat(fvals), \
                                                          repeat(mnDirs_num), repeat(splitTriCoords[ii]))))
      
      for jj in range(numBasis):
        surfInts[faceInd][:,jj,:] += curInt[jj]
    if faceInd == 0:
      # Rescale integrals over this face because it has an area of sqrt(3) (the rest have area 1)
      surfInts[faceInd] /= np.sqrt(3)

    print(f"Finished calculating integrals for face {faceInd} in {(time.time() - start_time):.5f} s.")

  return surfInts

def singFaceInts_loop(ii, numBasis, fvals, mnDirs, triCoords):
  """
  Function for enabling parallelization of the loop in singFaceInts()

  """

  subFaceInt = np.zeros((6, numBasis), dtype=np.float64)

  for jj in range(numBasis):
    # print(f"(ii, jj) = {(ii, jj)}")
    for mn in range(6):
      vecVal = fvals[ii,:,mnDirs[mn][0]]*fvals[jj,:,mnDirs[mn][1]] \
                + fvals[ii,:,mnDirs[mn][1]]*fvals[jj,:,mnDirs[mn][0]]

      if mnDirs[mn][0] == mnDirs[mn][1]:
        theta = 1/2
      else:
        theta = 1
      vecVal *= theta

      subFaceInt[mn,jj] += q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], vecVal)

  return subFaceInt

def saveSingFaceInts(filename, singCoeff, cornerSingCoeff, basisOrder, singOrder, numSplit):
  start_time = time.time()
  surfInts = singFaceInts(singCoeff, cornerSingCoeff, basisOrder, singOrder, numSplit)
  np.savez(filename, surfInts=surfInts)
  print(f"Finished running for singularity coefficient {singCoeff:.3f} with {numSplit} splits in {(time.time() - start_time):.5f} s")

if __name__ == "__main__":
  basisOrder, singOrder = [1, 1]
  numSplit = 5
  singCoeffList = np.array([2/3], dtype=np.float64)
  cornerSingCoeffList = np.array([2/3], dtype=np.float64) 
  filenames = [f"./BasisMats/singFaceInts_tets_0p{int(np.round(100*singCoeffList[ii]))}_p={basisOrder}_s={singOrder}_numSplit={numSplit}_shiftedGrid.npz" for ii in range(len(singCoeffList))]
  for ii in range(len(singCoeffList)):
    saveSingFaceInts(filenames[ii], singCoeffList[ii], cornerSingCoeffList[ii], basisOrder, singOrder, numSplit)
    
