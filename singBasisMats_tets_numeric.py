import numpy as np
import gauss_quadrature_tetra as q
from multiprocessing import Pool, cpu_count
from itertools import repeat, product
from assemble_mats import splitTet
from singBasis_tets import vecBasisSingular_orderN
import time
from ReadAbaqusMesh import readAbq

# Derives matrices for evaluating the singular basis functions using gaussian quadrature rules
# This version evaluates the basis functions numerically rather than using symbolic definitions

def singBasisMatrices(coeff, cornerCoeff, basisOrder, singOrder, numSplit):
  # Set up integration bounds for a unit tetrahedron
  coords = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=np.float64)

  # Run the basis function generator once to get the number of functions
  tempLoc = q.gauss_quadrature_tetra_numba_loc(coords, 8)
  fvals, _ = vecBasisSingular_orderN(coords, coeff, cornerCoeff, tempLoc, basisOrder, singOrder)
  numBasis = len(fvals)

  Tmat = np.zeros((6, numBasis, numBasis), dtype=np.float64)
  Smat = np.zeros((6, numBasis, numBasis), dtype=np.float64)
  mnDirs_num = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]], dtype=np.int32)

  # Split the tet into a set of smaller tets, and define integration points on each
  # splitCoords, splitTetList = splitTet(coords, 3)
  splitCoords, splitTetList, _, _, _, _, _, _, _, _, _ = readAbq(f"./Test_Files/Singular_testing/tetSplit_numSplit={numSplit}.inp")
  splitCoords = splitCoords[:,1:] # Remove the node indices
  splitTetList -= 1 # Shift by 1 to account for python vs Cubit numbering

  splitLoc, splitTetCoords = [[], []]
  for nodes in splitTetList:
    splitTetCoords.append(splitCoords[nodes])
    splitLoc.append(q.gauss_quadrature_tetra_numba_loc(splitTetCoords[-1], 8))
  splitLoc = np.array(splitLoc)
  splitTetCoords = np.array(splitTetCoords)

  # splitLoc = np.array([q.gauss_quadrature_tetra_numba_loc(coords, 8)])
  # splitTetCoords = np.array([coords])
  
  p = Pool(np.min([cpu_count(), numBasis]))

  # Evaluate the functions at the integration locations
  numEle = len(splitTetList)
  for ii in range(numEle):
    start_time = time.time()
    fvals, cvals = vecBasisSingular_orderN(coords, coeff, cornerCoeff, splitLoc[ii], basisOrder, singOrder)
    print(f"Evaluated basis functions for split tet {ii} in {(time.time() - start_time):.5f} s.")

    start_time = time.time()
    # totMats = singBasisMatrices_loop(0, numEle, numBasis, fvals, cvals, mnDirs_num, splitPrismCoords, splitVertLoc)
    totMats = list(p.starmap(singBasisMatrices_loop, zip(np.arange(numBasis), repeat(numBasis), repeat(fvals), repeat(cvals), \
                                                        repeat(mnDirs_num), repeat(splitTetCoords[ii]))))
    
    for jj in range(numBasis):
      Tmat[:,jj,:] += totMats[jj][0]
      Smat[:,jj,:] += totMats[jj][1]
    print(f"Filled matrices for split tet {ii} in {(time.time() - start_time):.5f} s.")

  return Tmat, Smat

def singBasisMatrices_loop(ii, numBasis, fvals, cvals, mnDirs, tetCoords):
  """
  Function for enabling parallelization of the loop in singBasisMatrices()

  """

  subTmat = np.zeros((6, numBasis), dtype=np.float64)
  subSmat = np.zeros((6, numBasis), dtype=np.float64)

  for jj in range(numBasis):
    # print(f"(ii, jj) = {(ii, jj)}")
    for mn in range(6):
      vecVal = fvals[ii,:,mnDirs[mn][0]]*fvals[jj,:,mnDirs[mn][1]] \
                + fvals[ii,:,mnDirs[mn][1]]*fvals[jj,:,mnDirs[mn][0]]
      curlVal = cvals[ii,:,mnDirs[mn][0]]*cvals[jj,:,mnDirs[mn][1]] \
                + cvals[ii,:,mnDirs[mn][1]]*cvals[jj,:,mnDirs[mn][0]]

      if mnDirs[mn][0] == mnDirs[mn][1]:
        theta = 1/2
      else:
        theta = 1
      vecVal *= theta
      curlVal *= theta

      # loc = q.gauss_quadrature_tetra_numba_loc(numCoords, 8)
      subTmat[mn,jj] += q.gauss_quadrature_tetra_numba_I(tetCoords, 8, vecVal)
      subSmat[mn,jj] += q.gauss_quadrature_tetra_numba_I(tetCoords, 8, curlVal)

  return subTmat, subSmat

def saveSingMats(filename, singCoeff, cornerSingCoeff, basisOrder, singOrder, numSplit):
  startTime = time.time()
  singTmn, singSmn = singBasisMatrices(singCoeff, cornerSingCoeff, basisOrder, singOrder, numSplit)
  np.savez(filename, singTmn=singTmn, singSmn=singSmn)
  print(f"Finished running for singularity coefficient {singCoeff:.5f}. Total elapsed time: {time.time() - startTime} s.")

if __name__ == "__main__":
  basisOrder, singOrder = [1,1]
  numSplit = 5
  singCoeffList = np.array([2/3], dtype=np.float64)
  cornerSingCoeffList = np.array([2/3], dtype=np.float64) 
  filenames = [f"./BasisMats/singBasisMats_tets_0p{int(np.round(100*singCoeffList[ii]))}_p={basisOrder}_s={singOrder}_numSplit={numSplit}_shiftedGrid.npz" for ii in range(len(singCoeffList))]
  for ii in range(len(singCoeffList)):
    saveSingMats(filenames[ii], singCoeffList[ii], cornerSingCoeffList[ii], basisOrder, singOrder, numSplit)
    
