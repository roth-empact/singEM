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

import gauss_quadrature_tetra as q
import numpy as np
import scipy.sparse as spa
import sys
from basis_funcs import *
from HigherOrder3D import basis3DSquad_numba, basis3DVquad
from singBasis_tets import vecBasisSingular_orderN
import time
from itertools import repeat

def assembleMats(eleList, eleEdgeList, eleSingEdgeList, eleFaceList, startEle, singEleLists, volP, nodeList, basisMaps, \
                 basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder, abcList, simType, eleType, bc, meshStats):
  """
  A function to construct matrices arising from mesh information for the specified finite element method

  Inputs:
    eleList         - Contains node numbers associated with each element, as well as material properties in each element
    eleEdgeList     - Edge numbers associated with each element
    eleSingEdgeList - Singular edge numbers associated with each element
    eleFaceList     - Face numbers associated with each element (only used if basisOrder == 2)
    startEle        - Marks the element number of the first entry of eleList (needed for parallelization to work correctly)
    singEleLists    - Dictionary containing information about singular lines and basis functions
    volP            - Parameters of each volume associated with the current element type
    nodeList        - List of renumbered nodes (to account for PECs) and their coordinates
    basisMaps       - Contains information on how basis functions map to geometric entities
    abcList         - List of node numbers forming each triangle on the absorbing boundary
    basisOrder      - The order of basis functions to use. Currently '1' and '2' are supported for tetrahedral elements and wedge elements.
    singOrder       - The order of singular basis functions (if used)
    basisOrder_long - Order of the nonsingular basis functions in the longitudinal direction (for wedges/prisms)
    singOrder_long  - Order of the singular basis functions in the longitudinal direction (for wedges/prisms)
    quadOrder       - The order of gaussian quadrature to use for numerical integration
    abcList         - List of triangles forming the absorbing boundary (if used)
    simType         - The type of simulation to use (either 'E', 'A', or 'TCS'. Note that some features are not supported in 'TCS')
    eleType         - The type of elements specified by the connectivity lists. Currently either 'Tet' for tetrahedrons, or 'Wedge' for wedges (prisms)
    bc              - The boundary condition for the outer boundary
    meshStats       - Statistics on the number of each mesh entity
  
  Outputs:
    matStruct - Dictionary containing the following sparse matrices:
      K1              - Curl-curl matrix for edge basis functions
      K2              - Dot product matrix for edge basis functions
      Ree             - Loss matrix for edge basis functions
      Bee             - ABC matrix for edge basis functions
      Ken             - Coupling matrix between edges and nodes (only for A-phi and TCS)
      Ren             - Loss matrix with coupling between edges and nodes (only for A-phi and TCS)
      Kne             - Coupling matrix between edges and nodes (only for A-phi and TCS)
      Gnn             - Scalar product matrix for nodal basis functions (only for A-phi and TCS)
      Ben             - ABC matrix with coupling between edges and nodes (only for A-phi and TCS)
      Bne             - ABC matrix with coupling between nodes and edges (only for TCS)
      Bnn             - ABC matrix for nodal basis functions (only for TCS)

  """
  ep0 = 1e-9 / (36 * np.pi)
  mu0 = 4e-7 * np.pi
  vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap = [basisMaps["vecEdgeMap"], basisMaps["vecSingEdgeMap"], basisMaps["vecFaceMap"], basisMaps["vecVolMap"]]
  scalNodeMap, scalEdgeMap = [basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"]]
  N_tet = meshStats["N_tet"] # Number of tetrahedra in the mesh

  if eleType == "Tet":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 6
        numScalBasis = 4
      case 2:
        numVecBasis = 20 
        numScalBasis = 10
    
    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

  elif eleType == "Wedge":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 9
        numScalBasis = 6
      case 2:
        numVecBasis = 22
        numScalBasis = 12 # 15

    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]])

  numVecUn, numScalUn = getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap)

  ## Retrieve mappings for singular elements
  singEleMap = singEleLists[f"sing{eleType}Map"]
  if len(singEleLists[f"sing{eleType}s"]) != 0:
    singEleCoeffs = singEleLists[f"sing{eleType}Coeffs"]
    
    # Only include map entries for elements on the current processor
    numEle = len(eleList)
    singEleMap = singEleMap[startEle:startEle+numEle]

    # Get local matrices for each set of singular coefficients
    singCoeffs = np.unique(singEleCoeffs[singEleCoeffs != 0])
    singMats = {}
    for ii in range(len(singCoeffs)):
      coeffStr = f"0p{int(np.round(100*singCoeffs[ii]))}"
      if eleType.lower() == "tet":
        singMats[coeffStr] = np.load(f"BasisMats/singBasisMats_{eleType.lower()}s_{coeffStr}_p={basisOrder}_s={singOrder}_numSplit=6_shiftedGrid.npz")
      elif eleType.lower() == "wedge":
        raise ValueError("Singular wedge/prism elements are not supported in the current version of this package.")
      for jj in range(ii+1, len(singCoeffs)):
        # Only get entries with the lower number first since that's assumed to be the corner 
        # (and the higher number the edge)
        coeffStr = f"0p{int(np.round(100*singCoeffs[ii]))}_0p{int(np.round(100*singCoeffs[jj]))}"
        singMats[coeffStr] = np.load(f"BasisMats/singBasisMats_{eleType.lower()}_{coeffStr}_p={basisOrder}_s={singOrder}_numSplit=6_shiftedGrid.npz")

  tanList = np.array([volP[f"vol{ii}"]["tanD"] for ii in range(volP["num"])])
  if np.any(tanList != 0): # Determine whether there is a nonzero loss tangent (causing matrices to be complex)
    dataType = np.complex128
    tanFlag = True
  else:
    dataType = np.float64
    tanFlag = False

  # Initialize vectors for building sparse matrices
  numAppend = int(1e6) #num of zeros to append to each vector each time it needs to be increased
  rows = np.zeros((numAppend,), dtype=np.int32)
  cols = np.zeros((numAppend,), dtype=np.int32)
  K1_vals = np.zeros((numAppend,), dtype=np.float64)
  K2_vals = np.zeros((numAppend,), dtype=dataType)

  sigmaList = np.array([volP[f"vol{ii}"]["sigma"] for ii in range(volP["num"])])
  if np.any(sigmaList != 0): # Include conductivity if it's in the mesh info
    Ree_vals = np.zeros((numAppend,), dtype=np.float64)
    condFlag = True # Mark that conductivity is being used
  else:
    condFlag = False

  if bc == "ABC":
    rows_Bee = np.zeros((numAppend,), dtype=np.int32)
    cols_Bee = np.zeros((numAppend,), dtype=np.int32)
    Bee_vals = np.zeros((numAppend,), dtype=np.float64)
  if simType == "A" or simType == "TCS" or simType == "Phi":
    rows_Ken = np.zeros((numAppend,), dtype=np.int32)
    cols_Ken = np.zeros((numAppend,), dtype=np.int32)
    Ken_vals = np.zeros((numAppend,), dtype=dataType)
    Kne_vals = np.zeros((numAppend,), dtype=dataType)
    rows_Gnn = np.zeros((numAppend,), dtype=np.int32)
    cols_Gnn = np.zeros((numAppend,), dtype=np.int32)
    Gnn_vals = np.zeros((numAppend,), dtype=dataType)
    Tnn_vals = np.zeros((numAppend,), dtype=dataType)
    if condFlag:
      Ren_vals = np.zeros((numAppend,), dtype=np.float64)
    if bc == "ABC":
      rows_Bnn = np.zeros((numAppend,), dtype=np.int32)
      cols_Bnn = np.zeros((numAppend,), dtype=np.int32)
      Bnn_vals = np.zeros((numAppend,), dtype=np.float64)
      if simType == "TCS":
        rows_Ben = np.zeros((numAppend,), dtype=np.int32)
        cols_Ben = np.zeros((numAppend,), dtype=np.int32)
        Ben_vals = np.zeros((numAppend,), dtype=np.float64)

  ind = 0 # Index for K1 and K2
  ind_Bee = 0
  curLen = numAppend # Tracks the current length of K1 and K2 
  curLen_Bee = numAppend
  if simType == "A" or simType == "TCS" or simType == "Phi":
    ind_Ken = 0 # Index for Ken and Kne
    ind_Gnn = 0 # Index for Gnn
    curLen_Ken = numAppend
    curLen_Gnn = numAppend
    if bc == "ABC":
      ind_Bnn = 0
      curLen_Bnn = numAppend
      if simType == "TCS":
        ind_Ben = 0
        curLen_Ben = numAppend

  # Loop through each volume and update material parameters
  for vol in range(volP["num"]): 
    volStartInd = volP[f"vol{vol}"]["startInd"] - startEle
    if vol == volP["num"] - 1:
      volEndInd = len(eleList)
    else:
      volEndInd = volP[f"vol{vol+1}"]["startInd"] - startEle

    if volEndInd <= 0:
      continue # If the element list doesn't include the current volume, move to the next
    elif volEndInd > len(eleList):
      volEndInd = len(eleList) # If the volume exceeds the element list, shift the end index

    if volStartInd < 0:
      volStartInd = 0 # If the element list starts in the middle of the current volume, shift the start index

    ep = volP[f"vol{vol}"]["epr"]*ep0
    muInv = volP[f"vol{vol}"]["murInv"]/mu0
    if np.any(volP[f"vol{vol}"]["tanD"] != 0):
      ep -= 1j*volP[f"vol{vol}"]["tanD"]
    sigma = volP[f"vol{vol}"]["sigma"]

    ## Loop through elements in the current volume and update the matrices
    for eleInd in range(volStartInd, volEndInd):
      # Get coordinates and global node numbers for the element
      coords = nodeList[eleList[eleInd]][:,1:]
      nodeNums = nodeList[eleList[eleInd], 0] # Also get global node numbers (including PEC)

      # Check whether the current element is singular or not and get indices for the rows and columns
      singFlag = singEleMap[eleInd]
      if singFlag != -1:
        singCoeffs = singEleCoeffs[singFlag]
      else:
        singCoeffs = []
      vecInds, scalInds = \
        selectInds3D(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, scalNodeMap, scalEdgeMap, vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, \
                     eleList[eleInd], eleEdgeList[eleInd], eleSingEdgeList[eleInd], eleFaceList[eleInd], eleInd+startEle, nodeNums, edgeNums, \
                     numVecBasis, numScalBasis, N_tet, singFlag, singCoeffs, simType)
      numVecInds, numScalInds = [len(vecInds), len(scalInds)]

      # Evaluate basis functions at each integration point
      if eleType == "Tet":
        if singEleMap[eleInd] != -1: # For singular elements, get precomputed matrices
          gradProducts, eProducts, eleVol, curTmn, curSmn = \
            basisSetupSingTet(coords, ep, muInv, singMats, singEleCoeffs[singEleMap[eleInd]], eleInd)
        else:
          loc = np.array(q.gauss_quadrature_tetra_numba_loc(coords, quadOrder)) # Determine integration points in the tetrahedron
          vecVals, curlVals, scalVals, gradVals = basisSetupTet(coords, loc, simType, basisOrder, edgeNums, singOrder=singOrder)
        
      elif eleType == "Wedge":
        if singEleMap[eleInd] != -1: # For singular elements, get precomputed matrices
          gradProducts, eProducts, eleVol, curTmn, curSmn \
            = basisSetupSingWedge(coords, ep, muInv, singMats, singEleCoeffs[singEleMap[eleInd]], eleInd, eleList[eleInd])
        else:
          vertDir, triLoc, vertLoc = getWedgeLocs(coords, quadOrder) # Integration points in each direction
          vecVals, curlVals, scalVals, gradVals = basisSetupWedge(coords, eleList[eleInd], eleEdgeList[eleInd], triLoc, vertLoc, vertDir, simType, basisOrder, basisOrder_long)

      # Determine if any faces lie on the ABC (if used) (Different than PEC because there can be multiple faces on ABC)
      ABC_loc = [] # Used as an indicator when forming matrices
      if bc == "ABC":
        ABC_surfs = []
        ABC_edgeVals = []
        ABC_nodeVals = []
        ABC_gradVals = []
        ABC_norm = []
        surfNums = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] # Nodes associated with each unique surface of the tet
        for surf in surfNums:
          if np.any(np.all(eleList[eleInd, surf] == abcList, axis=1)): # Check if the surface nodes form a boundary triangle
            ABC_surfs.append(surf)
            ABC_loc.append(q.gauss_quadrature_trisimp_init(
              N=3, p1=coords[surf[0]], p2=coords[surf[1]], p3=coords[surf[2]])) # Create integration points along each boundary
            
            # Evaluate basis functions
            ABC_edgeVals.append([basis_vector_numba(coords, edgeNums[ii][0], edgeNums[ii][1], ABC_loc[-1])[0] for ii in range(6)])
            ABC_nodeVals.append([basis3Dscalar_numba(coords, ii, ABC_loc[-1]) for ii in range(4)])
            ABC_gradVals.append([ABC_nodeVals[-1][ii][1] for ii in range(4)])
            ABC_nodeVals[-1] = [ABC_nodeVals[-1][ii][0] for ii in range(4)]

            # Determine the direction normal to the boundary surface
            totGradN = np.zeros((3,))
            for jj in range(3):
              totGradN += ABC_gradVals[-1][surf[jj]]
            ABC_norm.append(-totGradN / np.linalg.norm(totGradN))

      if simType == "E" or simType == "A" or simType == "TCS":
        for ii in range(numVecInds):
          ## Determine which row the entry should be placed in
          rowVal = vecInds[ii]
          if rowVal == -1:
            continue
          
          for jj in range(numVecInds):
            ## Determine which column the entry should be placed in
            colVal = vecInds[jj]
            if colVal == -1:
              continue

            ## Append the vectors for sparse matrix construction (if necessary)
            if (ind == curLen):
              rows = np.append(rows, np.zeros((numAppend,)))
              cols = np.append(cols, np.zeros((numAppend,)))
              K1_vals = np.append(K1_vals, np.zeros((numAppend,), dtype=np.float64))
              K2_vals = np.append(K2_vals, np.zeros((numAppend,), dtype=dataType))
              if condFlag:
                Ree_vals = np.append(Ree_vals, np.zeros((numAppend,), dtype=np.float64))
              
              curLen += numAppend

            rows[ind] = rowVal
            cols[ind] = colVal

            if singEleMap[eleInd] == -1: 
              if eleType == "Tet":
                K1_vals[ind] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.sum((curlVals[ii] @ muInv) * curlVals[jj], axis = 1))
                K2_vals[ind] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.sum((vecVals[ii] @ ep) * vecVals[jj], axis = 1))
              elif eleType == "Wedge":
                K1_vals[ind] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.sum((curlVals[ii] @ muInv) * curlVals[jj], axis = 1))
                K2_vals[ind] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.sum((vecVals[ii] @ ep) * vecVals[jj], axis = 1))
            else:
              K1_vals[ind] = eleVol*np.sum(eProducts*curSmn[:,ii,jj])
              K2_vals[ind] = eleVol*np.sum(gradProducts*curTmn[:,ii,jj])
            if condFlag and sigma != 0:
              Ree_vals[ind] = np.real(K2_vals[ind]*sigma/ep)
            ind += 1

            if len(ABC_loc) != 0: # Append ABC matrices if applicable
              ## Append the vectors for sparse matrix construction (if necessary)
              if ind_Bee == curLen_Bee: 
                rows_Bee = np.append(rows_Bee, np.zeros((numAppend,)))
                cols_Bee = np.append(cols_Bee, np.zeros((numAppend,)))
                Bee_vals = np.append(Bee_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Bee += numAppend

              rows_Bee[ind_Bee] = rowVal
              cols_Bee[ind_Bee] = colVal
              for kk in range(len(ABC_loc)): # Loop through each boundary surface of the tet
                Bee_vals[ind_Bee] += np.sqrt(ep0/mu0) * q.gauss_quadrature_trisimp(coords[ABC_surfs[kk][0]], coords[ABC_surfs[kk][1]], coords[ABC_surfs[kk][2]],
                                                                    np.sum(np.cross(ABC_norm[kk], ABC_edgeVals[kk][ii]) * np.cross(ABC_norm[kk], ABC_edgeVals[kk][jj]), axis=1))
              ind_Bee += 1

          if simType == "A" or simType == "TCS":
            for jj in range(numScalInds):
              ## Find the correct column of the matrix to use
              colVal = scalInds[jj]
              if colVal == -1:
                continue

              ## Append the vectors for sparse matrix construction (if necessary)
              if (ind_Ken == curLen_Ken):
                rows_Ken = np.append(rows_Ken, np.zeros((numAppend,)))
                cols_Ken = np.append(cols_Ken, np.zeros((numAppend,)))
                Ken_vals = np.append(Ken_vals, np.zeros((numAppend,), dtype=dataType))
                if condFlag:
                  Ren_vals = np.append(Ren_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Ken += numAppend

              rows_Ken[ind_Ken] = rowVal
              cols_Ken[ind_Ken] = colVal

              if eleType == "Tet":
                Ken_vals[ind_Ken] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.sum((vecVals[ii] @ ep) * gradVals[jj], axis = 1))
              elif eleType == "Wedge":
                Ken_vals[ind_Ken] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.sum((vecVals[ii] @ ep) * gradVals[jj], axis = 1))

              if condFlag:
                Ren_vals[ind_Ken] = np.real(Ken_vals[ind_Ken] * sigma/ep)

              ind_Ken += 1

              if len(ABC_loc) != 0 and simType == "TCS": # Add a term for ABCs if using TCS
                ## Append the vectors for sparse matrix construction (if necessary)
                if ind_Ben == curLen_Ben: 
                  rows_Ben = np.append(rows_Ben, np.zeros((numAppend,)))
                  cols_Ben = np.append(cols_Ben, np.zeros((numAppend,)))
                  Ben_vals = np.append(Ben_vals, np.zeros((numAppend,), dtype=np.float64))
                  curLen_Ben += numAppend
                
                rows_Ben[ind_Ben] = eleEdgeList[eleInd, ii]
                cols_Ben[ind_Ben] = colVal
                
                for kk in range(len(ABC_loc)):
                  Ben_vals[ind_Ben] += np.sqrt(ep0/mu0) * q.gauss_quadrature_trisimp(coords[ABC_surfs[kk][0]], coords[ABC_surfs[kk][1]], coords[ABC_surfs[kk][2]],
                                                                                    np.sum(np.cross(ABC_norm[kk], ABC_edgeVals[kk][ii]) * np.cross(ABC_norm[kk], ABC_gradVals[kk][jj]), axis=1))
                ind_Ben += 1
      
      if simType == "A" or simType == "TCS" or simType == "Phi":
        for ii in range(numScalInds):
          ## Find the correct row of the matrix to use
          rowVal = scalInds[ii]
          if rowVal == -1:
            continue

          for jj in range(numScalInds):
            ## Find the correct column of the matrix to use
            colVal = scalInds[jj]
            if colVal == -1:
              continue

            ## Append the vectors for matrix construction (if needed)
            if (ind_Gnn == curLen_Gnn):
              rows_Gnn = np.append(rows_Gnn, np.zeros((numAppend,)))
              cols_Gnn = np.append(cols_Gnn, np.zeros((numAppend,)))
              Gnn_vals = np.append(Gnn_vals, np.zeros((numAppend,), dtype=dataType))
              Tnn_vals = np.append(Tnn_vals, np.zeros((numAppend,), dtype=dataType))
              if condFlag and simType == "TCS":
                Rnn_vals = np.append(Rnn_vals, np.zeros((numAppend,), dtype=np.float64))
              curLen_Gnn += numAppend

            rows_Gnn[ind_Gnn] = rowVal
            cols_Gnn[ind_Gnn] = colVal

            if eleType == "Tet":
              Gnn_vals[ind_Gnn] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, scalVals[ii] * scalVals[jj])
              if basisOrder == 1: # Gradient is constant over the element for the first-order case
                Tnn_vals[ind_Gnn] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.array(list(repeat(np.sum((gradVals[ii] @ ep) * gradVals[jj]), len(loc)))))
              else:
                Tnn_vals[ind_Gnn] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.sum((gradVals[ii] @ ep) * gradVals[jj], axis=1))
            elif eleType == "Wedge":
              Gnn_vals[ind_Gnn] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, scalVals[ii] * scalVals[jj])
              Tnn_vals[ind_Gnn] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.sum((gradVals[ii] @ ep) * gradVals[jj], axis=1))

            if simType == "TCS" and condFlag:
              if eleType == "Tet":
                if basisOrder == 1: # Gradient is constant over the element for the first-order case
                  Rnn_vals[ind_Gnn] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.array(list(repeat(np.sum((gradVals[ii] @ sigma) * gradVals[jj]), len(loc)))))
                else:
                  Rnn_vals[ind_Gnn] = q.gauss_quadrature_tetra_numba_I(coords, quadOrder, np.sum((gradVals[ii] @ sigma) * gradVals[jj], axis=1))
              elif eleType == "Wedge":
                Rnn_vals[ind_Gnn] = q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.sum((gradVals[ii] @ sigma) * gradVals[jj], axis=1))
            ind_Gnn += 1
            
            if len(ABC_loc) != 0 and (ii in np.unique(ABC_surfs)) and (jj in np.unique(ABC_surfs)):
              if ind_Bnn == curLen_Bnn: # Add length to the matrices if they're at their maximum
                rows_Bnn = np.append(rows_Bnn, np.zeros((numAppend,)))
                cols_Bnn = np.append(cols_Bnn, np.zeros((numAppend,)))
                Bnn_vals = np.append(Bnn_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Bnn += numAppend
              
              rows_Bnn[ind_Bnn] = rowVal
              cols_Bnn[ind_Bnn] = colVal

              for kk in range(len(ABC_loc)):
                if simType == "A":
                  Bnn_vals[ind_Bnn] += 1/np.sqrt(mu0*ep0) * q.gauss_quadrature_trisimp(coords[ABC_surfs[kk][0]], coords[ABC_surfs[kk][1]], coords[ABC_surfs[kk][2]],
                                                                                  ABC_nodeVals[kk][ii]*ABC_nodeVals[kk][jj])
                elif simType == "TCS":
                  Bnn_vals[ind_Bnn] += np.sqrt(ep0/mu0) * q.gauss_quadrature_trisimp(coords[ABC_surfs[kk][0]], coords[ABC_surfs[kk][1]], coords[ABC_surfs[kk][2]],
                                                                                  np.array(list(repeat(np.sum(np.cross(ABC_norm[kk], ABC_gradVals[kk][ii]) * np.cross(ABC_norm[kk], ABC_gradVals[kk][jj])), len(ABC_edgeVals[0][0])))))
              ind_Bnn += 1

  ## Create sparse matrices and save them to a dictionary
  matStruct = {}
  if simType == "E" or simType == "A" or simType == "TCS":
    rows = rows[:ind]
    cols = cols[:ind]
    K1_vals = K1_vals[:ind]
    K2_vals = K2_vals[:ind]

    matStruct["K1"] = spa.csr_array((K1_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)
    matStruct["K2"] = spa.csr_array((K2_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)

    if condFlag:
      Ree_vals = Ree_vals[0:ind]
      matStruct["Ree"] = spa.csr_array((Ree_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)
    else:
      matStruct["Ree"] = spa.csr_array((numVecUn, numVecUn), dtype=np.float64)

    if bc == "ABC":
      rows_Bee = rows_Bee[:ind_Bee]
      cols_Bee = cols_Bee[:ind_Bee]
      Bee_vals = Bee_vals[:ind_Bee]

      matStruct["Bee"] = spa.csr_array((Bee_vals, (rows_Bee, cols_Bee)), shape=(numVecUn, numVecUn), dtype=np.float64)
    else:
      matStruct["Bee"] = spa.csr_array((numVecUn, numVecUn), dtype=np.float64)

  if simType == "A" or simType == "TCS":
    rows_Ken = rows_Ken[:ind_Ken]
    cols_Ken = cols_Ken[:ind_Ken]
    Ken_vals = Ken_vals[:ind_Ken]
    Kne_vals = Kne_vals[:ind_Ken]
    Gnn_vals = Gnn_vals[:ind_Gnn]
    Tnn_vals = Tnn_vals[:ind_Gnn]

    matStruct["Ken"] = spa.csr_array((Ken_vals, (rows_Ken, cols_Ken)), shape=(numVecUn, numScalUn), dtype=dataType)
    if simType == "A":
      matStruct["Kne"] = -1/(mu0*ep0**2)*matStruct["Ken"].copy().T.asformat('csr')
    else:
      matStruct["Kne"] = matStruct["Ken"].copy().T.asformat('csr')

    if condFlag:
      Ren_vals = Ren_vals[:ind_Ken]
      matStruct["Ren"] = spa.csr_array((Ren_vals, (rows_Ken, cols_Ken)), shape=(numVecUn, numScalUn), dtype=np.float64)
    else:
      matStruct["Ren"] = spa.csr_array((numVecUn, numScalUn), dtype=np.float64)
    
    if simType == "TCS":
      if condFlag:
        Rnn_vals = Rnn_vals[:ind_Ken]
        matStruct["Rne"] = matStruct["Ren"].copy().T.asformat('csr')
        matStruct["Rnn"] = spa.csr_array((Rnn_vals, (cols_Gnn, rows_Gnn)), shape=(numScalUn, numScalUn), dtype=np.float64)
      else:
        matStruct["Rne"] = spa.csr_array((numScalUn, numVecUn), dtype=np.float64)
        matStruct["Rnn"] = spa.csr_array((numScalUn, numScalUn), dtype=np.float64)

    if bc == "ABC":
      rows_Bnn = rows_Bnn[:ind_Bnn]
      cols_Bnn = cols_Bnn[:ind_Bnn]
      Bnn_vals = Bnn_vals[:ind_Bnn]

      matStruct["Bnn"] = spa.csr_array((Bnn_vals, (rows_Bnn, cols_Bnn)), shape=(numScalUn, numScalUn), dtype=np.float64)

      if simType == "TCS":
        rows_Ben = rows_Ben[:ind_Ben]
        cols_Ben = cols_Ben[:ind_Ben]
        Ben_vals = Ben_vals[:ind_Ben]

        matStruct["Ben"] = spa.csr_array((Ben_vals, (rows_Ben, cols_Ben)), shape=(numVecUn, numScalUn), dtype=np.float64)
        matStruct["Bne"] = spa.csr_array((Ben_vals, (cols_Ben, rows_Ben)), shape=(numScalUn, numVecUn), dtype=np.float64)
    else:
      matStruct["Bnn"] = spa.csr_array((numScalUn, numScalUn), dtype=np.float64)
      matStruct["Ben"] = spa.csr_array((numVecUn, numScalUn), dtype=np.float64)
      matStruct["Bne"] = spa.csr_array((numScalUn, numVecUn), dtype=np.float64)

  if simType == "A" or simType == "TCS" or simType == "Phi":
    rows_Gnn = rows_Gnn[:ind_Gnn]
    cols_Gnn = cols_Gnn[:ind_Gnn]
    Gnn_vals = Gnn_vals[:ind_Gnn]
    Tnn_vals = Tnn_vals[:ind_Gnn]

    matStruct["Gnn"] = spa.csr_array((Gnn_vals, (rows_Gnn, cols_Gnn)), shape=(numScalUn, numScalUn), dtype=dataType)
    matStruct["Tnn"] = spa.csr_array((Tnn_vals, (rows_Gnn, cols_Gnn)), shape=(numScalUn, numScalUn), dtype=dataType)

  return matStruct
  
def basisSetupTet(coords, loc, simType, basisOrder, edgeNums, getDCurl=False, singFlag=-1, singCoeffs=[], singOrder=0):
  """
  Sets up basis functions in a tetrahedral element given a set of evaluation points

  Inputs:
    coords     - coordinates of the 4 nodes of the tetrahedron
    loc        - coordinates of each point to evaluate the basis functions at
    simType    - Type of simulation being ran (either "E", "A", or "TCS")
    basisOrder - Order of the standard basis functions
    edgeNums   - Node numbers forming each edge (in local numbering)
    getDCurl   - Flag indicating whether to return the second curl of the basis functions
    singFlag   - Flag indicating whether the element is touching a singular feature
    singCoeffs - Singular coefficients of nodes in the element
    singOrder  - Order of singular basis functions

  Outputs:
    vecBasis   - Vector basis functions evaluated at each point
    curlBasis  - Curl of the vector basis functions at each point
    dCurlBasis - Second curl of the vector basis functions (only returned if getDCurl==True)
    scalBasis  - Scalar basis functions evaluated at each point (only nonzero if simType != "E")
    gradBasis  - Gradient of the scalar basis functions evaluated at each point (only nonzero if simType != "E")
  """

  if singFlag == -1:
    match basisOrder:
      case 1:
        if simType == "E" or simType == "A" or simType == "TCS":
          vecBasis = [basis_vector_numba(coords, edgeNums[ii, 0], edgeNums[ii, 1], loc) for ii in range(6)]
          curlBasis = [vecBasis[ii][1] for ii in range(6)] # Curl of the vector basis functions
          vecBasis = [vecBasis[ii][0] for ii in range(6)] # Vector basis functions
        else:
          vecBasis, curlBasis = [[], []]
        if simType == "A" or simType == "TCS" or simType == "Phi": # Also need to evaluate scalar basis functions (and their gradient)
          scalBasis = [basis3Dscalar_numba(coords, ii, loc) for ii in range(4)]
          gradBasis = [scalBasis[ii][1] for ii in range(4)] # Gradient of scalar basis functions
          scalBasis = [scalBasis[ii][0] for ii in range(4)] # Scalar basis functions
        else:
          scalBasis, gradBasis = [[], []]
      case 2: 
        if simType == "E" or simType == "A" or simType == "TCS":
          vecBasis, curlBasis, dCurlBasis = basis3DVquad(coords, loc, getDCurl)
        else:
          vecBasis, curlBasis = [[], []]
        if simType == "A" or simType == "TCS" or simType == "Phi": # Also need to evaluate scalar basis functions (and their gradient)
          scalBasis, gradBasis = basis3DSquad_numba(coords, loc)
        else:
          scalBasis, gradBasis = [[], []]
  else: 
    singCornerCoeff = singCoeffs[singCoeffs != 0][0] # TODO: Adjust to allow different coefficients for different singular points
    singCoeff = singCoeffs[singCoeffs != 0][0]
    # vecBasis, curlBasis = vecBasisSingular(coords, singType, singCoeff, singCornerCoeff, loc, basisOrder)
    vecBasis, curlBasis = vecBasisSingular_orderN(coords, singCoeff, singCornerCoeff, loc, basisOrder, singOrder)
    scalBasis, gradBasis = [[], []]

  if getDCurl:
    return vecBasis, curlBasis, dCurlBasis, scalBasis, gradBasis
  else:
    return vecBasis, curlBasis, scalBasis, gradBasis

def basisSetupWedge(coords, globNodeNums, edgeNums, triLoc, vertLoc, vertDir, simType, basisOrder, basisOrder_long, singFlag=-1, singCoeffs=[], singOrder=0, singOrder_long=0):
  """
  Sets up basis functions in a tetrahedral element given a set of evaluation points

  Inputs:
    coords          - coordinates of the 4 nodes of the tetrahedron
    globNodeNums    - global numbering of the nodes forming the wedge (for handling cases where the signs of edges on the top face are flipped)
    edgeNums        - Node numbers forming each edge (in local numbering)
    triLoc          - coordinates of each point along the triangular face to evaluate the basis functions
    vertLoc         - coordinates of each point in the vertical direction to evaluate the basis functions
    simType         - Type of simulation being ran (either "E", "A", or "TCS")
    basisOrder      - Order of the standard basis functions in the transverse direction
    basisOrder_long - Order of the standard basis functions in the longitudinal/vertical direction
    singFlag        - Flag indicating whether the element is touching a singular feature
    singCoeffs      - Singular coefficients of nodes in the element
    singOrder       - Order of singular basis functions in the transverse direction
    singOrder_long  - Order of singular basis functions in the longitudinal direction

  Outputs:
    vecBasis        - Vector basis functions evaluated at each point
    curlBasis       - Curl of the vector basis functions at each point
    dCurlBasis      - Second curl of the vector basis functions (only returned if getDCurl==True)
    scalBasis       - Scalar basis functions evaluated at each point (only nonzero if simType != "E")
    gradBasis       - Gradient of the scalar basis functions evaluated at each point (only nonzero if simType != "E")
  """
  if singFlag == -1:
    match basisOrder:
      case 1:
        vecBasis, curlBasis = basis3DVWedge(coords, globNodeNums, triLoc, vertLoc, vertDir)
        if simType == "A" or simType == "TCS":
          scalBasis, gradBasis = basis3DSWedge(coords, triLoc, vertLoc, vertDir)
        else:
          scalBasis, gradBasis = [[], []]
      case 2:
        match basisOrder_long:
          case 1:
            vecBasis, curlBasis = basis3DVWedgeQuad(coords, globNodeNums, triLoc, vertLoc, vertDir)
          case 2:
            vecBasis, curlBasis = basis3DVWedge_2x2(coords, globNodeNums, triLoc, vertLoc, vertDir)
        if simType == "A" or simType == "TCS":
          # scalBasis, gradBasis = basis3DSWedgeQuad_new(coords, globNodeNums, edgeNums, triLoc, vertLoc, vertDir)
          scalBasis, gradBasis = basis3DSWedgeQuad(coords, globNodeNums, triLoc, vertLoc, vertDir)
        else:
          scalBasis, gradBasis = [[], []]

  return vecBasis, curlBasis, scalBasis, gradBasis

def basisSetupSingTet(coords, ep, muInv, singMats, singCoeffs, eleInd):
  """
  Sets up quantities needed to use precomputed matrices for singular tetrahedra

  Inputs:
    coords       - Coordinates of each node of the tetrahedron
    ep           - Permittivity tensor of the current element
    muInv        - Inverse of the permeability tensor of the current element
    singMats     - Dictionary containing precalculated integrals for each singular coefficient in the mesh
    singCoeffs   - Singular coefficients of the nodes of the current element
    eleInd       - Index of the current element
  Outputs:
    gradProducts - Products of the gradients of the barycentric coordinates in each direction
    eProducts    - Products of the cross products of gradients of the barycentric coordinates in each direction
    eleVol       - Volume of the element
    curTmn       - Precalculated integrals of basis functions for the current element
    curSmn       - Precalculated integrals of curls of basis functions for the current element
  """

  gradLoc = np.array(q.gauss_quadrature_tetra_numba_loc(coords, 2))
  gradVals = [basis3Dscalar_numba(coords, ii, gradLoc) for ii in range(4)]
  gradVals = np.array([gradVals[ii][1] for ii in range(3)])
  gradInds = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]])
  gradProducts = np.array([np.sum((gradVals[gradInds[ii,0]] @ ep) * gradVals[gradInds[ii,1]]) for ii in range(len(gradInds))])
  eVals = np.zeros((3,3), dtype=np.float64)
  eVals[0,:] = np.cross(gradVals[1,:], gradVals[2,:])
  eVals[1,:] = np.cross(gradVals[2,:], gradVals[0,:])
  eVals[2,:] = np.cross(gradVals[0,:], gradVals[1,:])
  eProducts = np.array([np.sum((eVals[gradInds[ii,0]] @ muInv)*eVals[gradInds[ii,1]]) for ii in range(len(gradInds))])
  eleVol = np.abs(np.linalg.det(np.array([[1, *coords[0]], [1, *coords[1]], [1, *coords[2]], [1, *coords[3]]])))

  curCoeffs = np.unique(singCoeffs[singCoeffs != 0])
  if len(curCoeffs) == 1:
      curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}"
  elif len(curCoeffs) == 2:
    curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}_0p{int(np.round(curCoeffs[1]*100))}"
  elif len(curCoeffs) > 2:
    print(f"Tet {eleInd} has singular nodes with more than two coefficients. \
            Only taking the first two.")
    curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}_0p{int(np.round(curCoeffs[1]*100))}"
  else:
    return gradProducts, eProducts, eleVol, [], []
  curTmn = singMats[curCoeffStr]["singTmn"]
  curSmn = singMats[curCoeffStr]["singSmn"]

  return gradProducts, eProducts, eleVol, curTmn, curSmn

def basisSetupSingWedge(coords, ep, muInv, singMats, singCoeffs, eleInd, globNodeNums):
  """
  Sets up quantities needed to use precomputed matrices for singular wedges

  Inputs:
    coords       - Coordinates of each node of the tetrahedron
    ep           - Permittivity tensor of the current element
    muInv        - Inverse of the permeability tensor of the current element
    singMats     - Dictionary containing precalculated integrals for each singular coefficient in the mesh
    singCoeffs   - Singular coefficients of the nodes of the current element
    eleInd       - Index of the current element
    globNodeNums - Global numbering of nodes in the wedge (used to handle edge cases where the top nodes are out of order)
  Outputs:
    gradProducts - Products of the gradients of the barycentric coordinates in each direction
    eProducts    - Products of the cross products of gradients of the barycentric coordinates in each direction
    eleVol       - Volume of the element
    curTmn       - Precalculated integrals of basis functions for the current element
    curSmn       - Precalculated integrals of curls of basis functions for the current element
  """

  tanDirs = []
  for testDir in [0, 1, 2]: # Find the vertical direction
    if np.isclose(coords[0,testDir], coords[1,testDir], atol=1e-8) and np.isclose(coords[0,testDir], coords[2,testDir], atol=1e-8):
      vertDir = testDir
    else:
      tanDirs.append(testDir)
  tanDirs = np.array(tanDirs, dtype=np.int32)

  triCoords = coords[:3]
  gradLoc = np.array(q.gauss_quadrature_trisimp_init(2, *triCoords))
  gradVals = [basis2Dscalar_numba(triCoords, ii, gradLoc) for ii in range(2)]
  gradVals = np.array([gradVals[ii][1] for ii in range(2)])
  vertGrad = np.zeros((1,3), dtype=np.float64)
  vertGrad[0,vertDir] = 1/(coords[3,vertDir] - coords[0,vertDir])
  gradVals = np.vstack([gradVals, vertGrad])

  gradInds = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]])
  gradProducts = np.array([np.sum((gradVals[gradInds[ii,0]] @ ep) * gradVals[gradInds[ii,1]]) for ii in range(len(gradInds))])
  eVals = np.zeros((3,3), dtype=np.float64)
  eVals[0,:] = np.cross(gradVals[1,:], gradVals[2,:])
  eVals[1,:] = np.cross(gradVals[2,:], gradVals[0,:])
  eVals[2,:] = np.cross(gradVals[0,:], gradVals[1,:])
  eProducts = np.array([np.sum((eVals[gradInds[ii,0]] @ muInv)*eVals[gradInds[ii,1]]) for ii in range(len(gradInds))])
  eleVol = np.abs((coords[3,vertDir] - coords[0,vertDir]) * np.linalg.det(np.array([[1, *coords[0,tanDirs]], [1, *coords[1,tanDirs]], [1, *coords[2,tanDirs]]])))

  curCoeffs = np.unique(singCoeffs[singCoeffs != 0])
  if len(curCoeffs) == 1:
      curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}"
  elif len(curCoeffs) == 2:
    curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}_0p{int(np.round(curCoeffs[1]*100))}"
  elif len(curCoeffs) > 2:
    print(f"Wedge {eleInd} has singular nodes with more than two coefficients. \
            Only taking the first two.")
    curCoeffStr = f"0p{int(np.round(curCoeffs[0]*100))}_0p{int(np.round(curCoeffs[1]*100))}"
  else:
    return gradProducts, eProducts, eleVol, [], []

  topNodeOrder = np.argsort(np.argsort(globNodeNums[3:]))
  if np.all(topNodeOrder == np.arange(3)):
    # If the top nodes are in order, use the standard precomputed matrices
    curTmn = singMats[curCoeffStr]["singTmn"]
    curSmn = singMats[curCoeffStr]["singSmn"]
  else:
    # If the top nodes are out of order, use precomputed matrices for that order
    curTmn = singMats[curCoeffStr][f"singTmn_{topNodeOrder[0]}{topNodeOrder[1]}{topNodeOrder[2]}"]
    curSmn = singMats[curCoeffStr][f"singSmn_{topNodeOrder[0]}{topNodeOrder[1]}{topNodeOrder[2]}"]

  return gradProducts, eProducts, eleVol, curTmn, curSmn

def getWedgeLocs(coords, quadOrder):
  """
  For a given gaussian quadrature order, returns the quadrature locations in a wedge element

  Inputs:
    coords    - Coordinates of each node of the tetrahedron
    quadOrder - Gaussian quadrature order
  Outputs:
    vertDir   - Unit vector of the vertical direction in the wedge
    triLoc    - Integration locations along the triangular face
    vertLoc   - Integration locations in the vertical direction
  """
  for testDir in [0, 1, 2]: # Find the vertical direction
    if np.isclose(coords[0,testDir], coords[1,testDir], atol=1e-8) and np.isclose(coords[0,testDir], coords[2,testDir], atol=1e-8):
      vertDir = testDir
  triLoc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, coords[0], coords[1], coords[2])) # Integration points on the triangular surface
  vertLoc = np.array(q.gauss_quadrature_1D_init(quadOrder, coords[0,vertDir], coords[3,vertDir])) # Integration points in the vertical direction

  return vertDir, triLoc, vertLoc

def getPortInts(portList, portEdgeList, portFaceList, nodeList, vecEdgeMap, vecFaceMap, scalNodeMap, scalEdgeMap, \
                portP, quadOrder, basisOrder, simType, numVecUn):
  '''
  Saves integrals of port basis functions for forming matrices and calculating voltages

  Inputs:
    portList     - List of global (i.e.: including PEC nodes) node indices forming triangles along the port
    portEdgeList - List of edge indices forming triangles along the port
    portFaceList - Face numbers associated with each port triangle. Only needed for basisOrder = 2
    nodeList     - Column 0 contains the node number (excluding PEC) corresponding to the global 
                   node number (including PEC) of that row index. Columns 1-3 provide the x,y,z coordinates
                   of the node.
    vecEdgeMap   - Provides the matrix indices for each edge number for vector basis functions
    vecFaceMap   - Provides the matrix indices for each face number for vector basis functions
    scalNodeMap  - Provides the matrix indices for each node number for scalar basis functions
    scalEdgeMap  - Provides the matrix indices for each edge number for scalar basis functions
    portP        - Contains info on the properties of the current port
    quadOrder    - The order of Gaussian Quadrature to use (for surface integrals).
    basisOrder   - Order of basis functions to use (currently 1 and 2 are supported).
    simType      - Indicates whether to use the field-based solver ("E"), potential-based solver ("A"),
                   or tree-cotree splitting ("TCS")
    numVecUn     - Total number of vector unknowns

  Outputs:
    vecInts         - Integrals of each vector basis function along the port surface
    scalInts        - Integrals of each scalar basis function along the port surface
    newVoltVecVals  - Values for calculating voltage across the surface using vector basis functions
    newVoltVecInds  - Indices for calculating voltage across the surface using vector basis functions
    newVoltScalVals - Values for calculating voltage across the surface using scalar basis functions
    newVoltScalInds - Indices for calculating voltage across the surface using scalar basis functions
    portLen         - Total length of the port surface
  '''

  match basisOrder: # Handle different basis function numbering for higher order cases
    case 1:
      numVecBasis = 3
      numScalBasis = 3
    case 2:
      numVecBasis = 8
      numScalBasis = 6


  edgeNums = np.array([[0,1], [0,2], [1,2]])
  # Initialize vectors for storage
  vecInts = np.zeros((len(portList), numVecBasis)) # Store integration results over all edges on the port
  voltVecVals = np.zeros(((numVecBasis)*len(portList),), dtype=np.float64) # Store values for calculating the voltage along the port
  voltVecInds = np.zeros(voltVecVals.shape, dtype=np.int32) # Store indices for calculating the voltage along the port
  if simType == "A" or simType == "TCS":
    scalInts = np.zeros((len(portList), numScalBasis)) # Store integration results over all nodes on the port
    voltScalVals = np.zeros(((numScalBasis)*len(portList),), dtype=np.float64) 
  else:
    scalInts = []
    voltScalVals = np.zeros((0,), dtype=np.float64) 
  voltScalInds = np.zeros(voltScalVals.shape, dtype=np.int32) # Store indices for calculating the voltage along the port
  vecInd, scalInd = [0, 0]

  ## Set up a rotated coordinate system normal to z to simplify 2D integration
  newY = np.cross(portP["direction"], portP["normal"])
  rotMat = np.array([portP["direction"], newY, portP["normal"]])
  xDir = np.array([1.0,0,0], dtype=np.float64) # The port is aligned in the x direction in the new system
  zDir = np.array([0.0,0,1.0], dtype=np.float64) # The port is normal to the x direction in the new system

  # Index through the triangles, evaluate basis functions and save integration results
  for triNum, tri in enumerate(portList):
    coords = nodeList[tri][:,1:] # Coordinates for the 3 nodes of the triangle
    coords = (rotMat @ coords.T).T # Rotate the coordinates to be normal to z and align the port direction with x
    loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, coords[0], coords[1], coords[2])) # Determine integration points in the triangle

    if triNum == 0: # At the first triangle, set up coordinate storage for finding the port length
      xMin = np.min(coords[:,0])
      xMax = np.max(coords[:,0])
    else: # Update the min and max values for the current triangle
      xMin = np.min([xMin, np.min(coords[:,0])])
      xMax = np.max([xMax, np.min(coords[:,0])])

    ## Save basis functions for the current triangle
    if basisOrder == 1:
      edgeValsTri = [basis_vector_numba(coords, edgeNums[ii,0], edgeNums[ii,1], loc)[0] for ii in range(3)]
    elif basisOrder == 2:
      edgeValsTri, _ = basis2DVquad(coords, loc)

    for edgeInd in range(numVecBasis):
      colVal = selectVecInds2D(basisOrder, edgeInd, vecEdgeMap, vecFaceMap, portEdgeList[triNum], portFaceList[triNum])
      if colVal == -1:
        continue

      vecVals = edgeValsTri[edgeInd]

      # Integrate the basis function over the triangle and store the result
      if np.all(portP["normal"] == [0, 0, 1]): # If not normal to z, rotate the coordinates to allow integration in the xy plane
        vecInts[triNum, edgeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], vecVals @ xDir)

      voltVecVals[vecInd] = -vecInts[triNum, edgeInd] / portP["width"]
      voltVecInds[vecInd] = colVal
      vecInd += 1

    if simType == "A" or simType == "TCS": # Repeat for nodes (if using A-phi or TCS)
      if basisOrder == 1:
        gradBasisTri = [basis2Dscalar_numba(coords, nodeInd, loc)[1] for nodeInd in range(3)]
      elif basisOrder == 2:
        _, gradBasisTri = basis2DSquad(coords, loc)

      nodeNums = nodeList[portList[triNum], 0]
      for nodeInd in range(numScalBasis):
        colVal = selectScalInds2D(nodeInd, scalNodeMap, scalEdgeMap, portList[triNum], portEdgeList[triNum], nodeNums, edgeNums)
        if colVal == -1:
          continue
        
        # Evaluate the scalar basis function at the integration points
        gradVals = gradBasisTri[nodeInd]
        if np.any(portP["normal"] != [0, 0, 1]): # If not normal to z, rotate the coordinates to allow integration in the xy plane
          ## TODO: Generalize this to allow arbitrary rotations
          if np.all(portP["normal"] == [1, 0, 0]):
            gradVals = np.array([gradVals[2], gradVals[0], gradVals[1]])
          elif np.all(portP["normal"] == [0, 1, 0]):
            gradVals = np.array([gradVals[1], gradVals[2], gradVals[0]])

        # Integrate the basis function over the triangle and store the result
        if basisOrder == 1: # Handle 1st order case seperately b/c the gradient is constant and defined at only a single point
          scalInts[triNum, nodeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.array(list(repeat(gradVals @ xDir,len(loc)))))
        else:
          scalInts[triNum, nodeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], gradVals @ xDir)

        voltScalVals[scalInd] = -scalInts[triNum, nodeInd] / portP["width"]
        voltScalInds[scalInd] = colVal
        scalInd += 1

  ## Condense the voltVec vectors by consolidating entries pointing to the same index
  voltVecVals = voltVecVals[:vecInd]
  voltVecInds = voltVecInds[:vecInd]
  voltVecSort = np.argsort(voltVecInds)
  voltVecVals = voltVecVals[voltVecSort]
  voltVecInds = voltVecInds[voltVecSort]
  newVoltVecInds = np.unique(voltVecInds[voltVecSort])
  newVoltVecVals = np.zeros((len(newVoltVecInds),), dtype=np.float64)
  ind = 0
  for ii in range(len(newVoltVecVals)):
    while ind != vecInd and newVoltVecInds[ii] == voltVecInds[ind]:
      newVoltVecVals[ii] += voltVecVals[ind]
      ind += 1

  if simType == "A" or simType == "TCS":
    voltScalVals = voltScalVals[:scalInd]
    voltScalInds = voltScalInds[:scalInd]
    voltScalSort = np.argsort(voltScalInds)
    voltScalVals = voltScalVals[voltScalSort]
    voltScalInds = voltScalInds[voltScalSort]
    newVoltScalInds = np.unique(voltScalInds[voltScalSort])
    newVoltScalVals = np.zeros((len(newVoltScalInds),), dtype=np.float64)
    ind = 0
    for ii in range(len(newVoltScalVals)):
      while ind != scalInd and newVoltScalInds[ii] == voltScalInds[ind]:
        newVoltScalVals[ii] += voltScalVals[ind]
        ind += 1
  else:
    newVoltScalInds, newVoltScalVals = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float64)]

  ## Find the length of the port
  portLen = np.abs(xMin - xMax)

  return vecInts, scalInts, newVoltVecVals, newVoltVecInds, newVoltScalVals, newVoltScalInds, portLen

def getSurfCurrentInts(volEleList, volEdgeList, volFaceList, faceVolList, volEleType, surfList, surfEdgeList, surfFaceList, \
                       nodeList, vecEdgeMap, vecFaceMap, surfP, volP, quadOrder, basisOrder, simType):
  '''
  Saves a vector for calculating the current along a surface in sparse format

  Inputs:
    volEleList   - List of global (including PEC nodes) node indices forming each volume element
    volEdgeList  - List of edge indices forming each volume element
    volFaceList  - Face numbers associated with each volume element
    faceVolList  - Volume element numbers associated with each face
    volEleType   - Element type for the volume
    surfList     - Node numbers associated with each surface triangle
    surfEdgeList - Edge numbers associated with each surface triangle
    surfFaceList - Face numbers associated with each surface triangle
    nodeList     - Column 0 contains the node number (excluding PEC) corresponding to the global 
                   node number (including PEC) of that row index. Columns 1-3 provide the x,y,z coordinates
                   of the node.
    vecEdgeMap   - Provides the matrix indices for each edge number for vector basis functions
    vecFaceMap   - Provides the matrix indices for each face number for vector basis functions
    surfP        - Contains info on the properties of the current surface
    volP         - Contains info about the material properties of each volume
    quadOrder    - The order of Gaussian Quadrature to use (for surface integrals).
    basisOrder   - Order of basis functions to use (currently 1 and 2 are supported).
    simType      - Indicates whether to use the field-based solver ("E"), potential-based solver ("A"),
                   or tree-cotree splitting ("TCS")

  Outputs:
    curVecVals   - Nonzero values for a vector for calculating the current across the surface
    curVecInds   - Indices for the values of curVecVals
  '''

  mu0 = 4e-7 * np.pi

  if volEleType == "Tet":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 6
        numScalBasis = 4
      case 2:
        numVecBasis = 20 
        numScalBasis = 10
    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
  elif volEleType == "Wedge":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 9
        numScalBasis = 6
      case 2:
        numVecBasis = 22
        numScalBasis = 15
    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]])

  # Initialize vectors for storage
  surfStartInds = [surfP[f"{surf}"]["startInd"] for surf in range(surfP["num"])]
  surfStartInds.append(len(surfList))
  curVecVals = [np.zeros((2*numVecBasis*(surfStartInds[ii+1]-surfStartInds[ii]),), dtype=np.float64) for ii in range(surfP["num"])]
  curVecInds = [np.zeros(curVecVals[0].shape, dtype=np.int32) for _ in range(surfP["num"])] # Store indices for calculating the current along the port
  newCurVecVals = [[] for _ in range(volP["num"])]
  newCurVecInds = [[] for _ in range(volP["num"])]

  volStartInds = [volP[f"vol{vol}"]["startInd"] for vol in range(volP["num"])]

  # Index through the triangles on each surface
  for surf in range(surfP["num"]):
    surfStartInd = surfStartInds[surf]
    surfEndInd = surfStartInds[surf+1]
    curInd = 0

    for triInd, triFace in enumerate(surfFaceList[surfStartInd:surfEndInd]):
      volEleNums = faceVolList[triFace]

      # Loop through the elements and save basis function integrals
      for locFaceNum, eleInd in enumerate(volEleNums):
        sign = 2*locFaceNum - 1 # Set the sign based on which tet is being looked at

        # Determine material parameters of the current volume
        shiftedStartInds = volStartInds - eleInd
        curVolNum = np.sum(shiftedStartInds <= 0) - 1
        muInv = volP[f"vol{curVolNum}"]["murInv"]/mu0

        # Evaluate basis functions at each integration point (set up along the triangle)
        coords = nodeList[volEleList[eleInd]][:,1:]
        triCoords = nodeList[surfList[surfStartInd+triInd]][:,1:]
        loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, triCoords[0], triCoords[1], triCoords[2])) # Determine integration points in the tetrahedron
        if volEleType == "Tet":
          _, curlVals, _, _ = basisSetupTet(coords, loc, simType, basisOrder, edgeNums)
        elif volEleType == "Wedge":
          triLoc = loc
          vertLoc = loc[0,2]
          vertDir = surfP[f"{surf}"]["normal"]
          _, curlVals, _, _ = basisSetupWedge(coords, volEleList[eleInd], volEdgeList[eleInd], triLoc, vertLoc, vertDir, simType, basisOrder)

        # Loop through the vector basis functions to get the contributions of each to the current
        for ii in range(numVecBasis):
          # Determine which column the entry should be placed in
          colVal = selectVecInds3D(basisOrder, ii, volEleType, vecEdgeMap, vecFaceMap, volEdgeList[eleInd], volFaceList[eleInd])
          if colVal == -1:
            continue

          # Integrate the basis function over the triangle and store the result
          curVecVals[surf][curInd] = sign*q.gauss_quadrature_trisimp(triCoords[0,:], triCoords[1,:], triCoords[2,:], \
                                                          np.cross(surfP[f"{surf}"]["normal"], muInv @ curlVals[ii]) @ surfP[f"{surf}"]["direction"]) / surfP[f"{surf}"]["width"]
          curVecInds[surf][curInd] = colVal
          curInd += 1

    ## Condense the curVec vectors by consolidating entries pointing to the same index
    curVecVals[surf] = curVecVals[surf][:curInd]
    curVecInds[surf] = curVecInds[surf][:curInd]
    curVecSort = np.argsort(curVecInds[surf])
    curVecVals[surf] = curVecVals[surf][curVecSort]
    curVecInds[surf] = curVecInds[surf][curVecSort]
    newCurVecInds[surf] = np.unique(curVecInds[surf][curVecSort])
    newCurVecVals[surf] = np.zeros((len(newCurVecInds[surf]),), dtype=np.float64)
    ind = 0
    for ii in range(len(newCurVecVals[surf])):
      while ind != curInd and newCurVecInds[surf][ii] == curVecInds[surf][ind]:
        newCurVecVals[surf][ii] += curVecVals[surf][ind]
        ind += 1

  return newCurVecVals, newCurVecInds
  
def assemblePortMats(portList, portEdgeList, portFaceList, nodeList, vecEdgeMap, vecFaceMap, scalNodeMap, scalEdgeMap, \
                     portP, vecInts, scalInts, basisOrder, simType, numVecUn, numScalUn):
  '''
  A function to create matrices and driving vectors for lumped ports

  Inputs:
    portList     - List of global (i.e.: including PEC nodes) node indices forming triangles along the port
    portEdgeList - List of edge indices forming triangles along the port
    portFaceList - Face numbers associated with each port triangle. Only needed for basisOrder = 2
    nodeList     - Column 0 contains the node number (excluding PEC) corresponding to the global 
                   node number (including PEC) of that row index. Columns 1-3 provide the x,y,z coordinates
                   of the node.
    vecEdgeMap   - Provides the matrix indices for each edge number for vector basis functions
    vecFaceMap   - Provides the matrix indices for each face number for vector basis functions
    scalNodeMap  - Provides the matrix indices for each node number for scalar basis functions
    scalEdgeMap  - Provides the matrix indices for each edge number for scalar basis functions
    portP        - Contains info on the properties of the current port
    vecInts      - Integrals of vector basis functions for each triangle
    scalInts     - Integrals of scalar basis functions for each triangle
    quadOrder    - The order of Gaussian Quadrature to use (for surface integrals).
    basisOrder   - Order of basis functions to use (currently 1 and 2 are supported).
    simType      - Indicates whether to use the field-based solver ("E"), potential-based solver ("A"),
                   or tree-cotree splitting ("TCS")
    numVecUn     - Total number of vector unknowns in the mesh
    numScalUn    - Total number of scalar unknowns in the mesh

  Outputs:
    f            - Source vector
    Ree          - Matrix representing edge-edge contribution to the resistance of each port
    Ren          - Matrix representing edge-node contribution to the resistance of each port 
                  (only used if simType == "A" or "TCS")
    Ren          - Matrix representing node-edge contribution to the resistance of each port 
                  (only used if simType == "TCS")
    Rnn          - Matrix representing node-node contribution to the resistnance of each port
                  (only used if simType == "TCS")
  '''
  
  numAppend = int(1e6) # number of zeros to append to each vector each time it needs to be increased
  rows = np.zeros((numAppend, ), dtype=np.int32)
  cols = np.zeros((numAppend, ), dtype=np.int32)
  Ree_vals = np.zeros((numAppend, ), dtype=np.float64)

  if simType == "A" or simType == "TCS": # Need additional nodal matrices if A-phi or TCS is used
    rows_Ren = np.zeros((numAppend, ), dtype=np.int32)
    cols_Ren = np.zeros((numAppend, ), dtype=np.int32)
    Ren_vals = np.zeros((numAppend, ), dtype=np.float64)
    # if simType == "TCS":
    rows_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    cols_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    Rnn_vals = np.zeros((numAppend, ), dtype=np.float64)
  
  ind = 0
  curLen = numAppend

  if simType == "A" or simType == "TCS":
    ind_Ren = 0
    curLen_Ren = numAppend

    # if simType == "TCS":
    ind_Rnn = 0
    curLen_Rnn = numAppend

  match basisOrder: # Handle different basis function numbering for higher order cases
    case 1:
      numVecBasis = 3
      numScalBasis = 3
    case 2:
      numVecBasis = 8
      numScalBasis = 6

  if simType == "E":
    f = np.zeros((numVecUn, ), dtype=np.float64) # Source vector
  elif simType == "A" or simType == "TCS":
    f = np.zeros((numVecUn+numScalUn, ), dtype=np.float64) # Source vector

  edgeNums = np.array([[0,1], [0,2], [1,2]])

  # Form the source vector
  if portP["Vsmag"] != 0:
    for triNum, _ in enumerate(portList):
      for edgeInd in range(numVecBasis):
        colVal = selectVecInds2D(basisOrder, edgeInd, vecEdgeMap, vecFaceMap, portEdgeList[triNum], portFaceList[triNum])
        if colVal == -1:
          continue
        
        if portP["type"].casefold() != "open".casefold():
          f[colVal] += portP["Vsmag"]*vecInts[triNum, edgeInd] / (portP["Z"]*portP["width"]) 
        else:
          f[colVal] += vecInts[triNum, edgeInd] / portP["width"]

      if simType == "TCS": 
        nodeNums = nodeList[portList[triNum], 0]
        for nodeInd in range(numScalBasis):
          colVal = selectScalInds2D(nodeInd, scalNodeMap, scalEdgeMap, portList[triNum], portEdgeList[triNum], nodeNums, edgeNums)
          if colVal == -1:
            continue

          if portP["type"].casefold() != "open".casefold():
            f[numVecUn+colVal] += portP["Vsmag"]*scalInts[triNum, nodeInd] / (portP["Z"]*portP["width"]) 
          else:
            f[numVecUn+colVal] += scalInts[triNum, nodeInd] / portP["width"]
            
  # If resistances are used, form those matrices by looping through the set of triangles
  if portP["type"].casefold() != "open".casefold():
    for tri_i in range(len(portList)):
      for ii in range(numVecBasis):
        rowVal = selectVecInds2D(basisOrder, ii, vecEdgeMap, vecFaceMap, portEdgeList[tri_i], portFaceList[tri_i])
        if rowVal == -1:
          continue
        
        for tri_j in range(len(portList)):
          for jj in range(numVecBasis):
            colVal = selectVecInds2D(basisOrder, jj, vecEdgeMap, vecFaceMap, portEdgeList[tri_j], portFaceList[tri_j])
            if colVal == -1:
              continue

            if ind == curLen:
              rows = np.append(rows, np.zeros((numAppend,)))
              cols = np.append(cols, np.zeros((numAppend,)))
              Ree_vals = np.append(Ree_vals, np.zeros((numAppend,), dtype=np.float64))
              curLen += numAppend
            
            rows[ind] = rowVal
            cols[ind] = colVal
            Ree_vals[ind] = vecInts[tri_i, ii] * vecInts[tri_j, jj] / (portP["Z"]*portP["width"]**2)
            ind += 1

          if simType == "A" or simType == "TCS": # Also loop through nodes for A-phi
            nodeNums_j = nodeList[portList[tri_j], 0]
            for jj in range(numScalBasis):
              colVal = selectScalInds2D(jj, scalNodeMap, scalEdgeMap, portList[tri_j], portEdgeList[tri_j], nodeNums_j, edgeNums)
              if colVal == -1:
                continue

              if ind_Ren == curLen_Ren:
                rows_Ren = np.append(rows_Ren, np.zeros((numAppend,)))
                cols_Ren = np.append(cols_Ren, np.zeros((numAppend,)))
                Ren_vals = np.append(Ren_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Ren += numAppend
            
              rows_Ren[ind_Ren] = rowVal
              cols_Ren[ind_Ren] = colVal

              Ren_vals[ind_Ren] = vecInts[tri_i, ii] * scalInts[tri_j, jj] / (portP["Z"]*portP["width"]**2)
              ind_Ren += 1

      if simType == "TCS": # For TCS, a node-node matrix is also needed
        nodeNums_i = nodeList[portList[tri_i], 0]
        for ii in range(numScalBasis):
          rowVal = selectScalInds2D(ii, scalNodeMap, scalEdgeMap, portList[tri_i], portEdgeList[tri_i], nodeNums_i, edgeNums)
          if rowVal == -1:
            continue

          for tri_j in range(len(portList)):
            nodeNums_j = nodeList[portList[tri_j], 0]
            for jj in range(numScalBasis):
              colVal = selectScalInds2D(jj, scalNodeMap, scalEdgeMap, portList[tri_j], portEdgeList[tri_j], nodeNums_j, edgeNums)
              if colVal == -1:
                continue

              if ind_Rnn == curLen_Rnn:
                rows_Rnn = np.append(rows_Rnn, np.zeros((numAppend,)))
                cols_Rnn = np.append(cols_Rnn, np.zeros((numAppend,)))
                Rnn_vals = np.append(Rnn_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Rnn += numAppend
            
              rows_Rnn[ind_Rnn] = rowVal
              cols_Rnn[ind_Rnn] = colVal

              Rnn_vals[ind_Rnn] = scalInts[tri_i, ii] * scalInts[tri_j, jj] / (portP["Z"]*portP["width"]**2)
              ind_Rnn += 1
          
  rows = rows[0:ind]
  cols = cols[0:ind]
  Ree_vals = Ree_vals[0:ind]

  Ree = spa.csr_array((Ree_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)

  if simType == "E":
    return f, Ree
  elif simType == "A" or simType == "TCS":
    rows_Ren = rows_Ren[0:ind_Ren]
    cols_Ren = cols_Ren[0:ind_Ren]
    Ren_vals = Ren_vals[0:ind_Ren]
    Ren = spa.csr_array((Ren_vals, (rows_Ren, cols_Ren)), shape=(numVecUn, numScalUn), dtype=np.float64)
    
    if simType == "A":
      return f, Ree, Ren
    elif simType == "TCS":

      rows_Rnn = rows_Rnn[0:ind_Rnn]
      cols_Rnn = cols_Rnn[0:ind_Rnn]
      Rnn_vals = Rnn_vals[0:ind_Rnn]

      Rne = spa.csr_array((Ren_vals, (cols_Ren, rows_Ren)), shape=(numScalUn, numVecUn), dtype=np.float64)
      Rnn = spa.csr_array((Rnn_vals, (rows_Rnn, cols_Rnn)), shape=(numScalUn, numScalUn), dtype=np.float64)

      return f, Ree, Ren, Rne, Rnn
    

def assemblePortMats_vols(volEleList, volEdgeList, volSingEdgeList, volFaceList, faceVolList, volEleType, portList, portEdgeList, portFaceList, nodeList, \
                          singEleLists, basisMaps, portP, basisOrder, quadOrder, simType):
  '''
  A function to create matrices and driving vectors for lumped ports
  This version of the function takes information from tets neighboring the port

  Inputs:
    volEleList   - List of global (including PEC nodes) node indices forming each volume element
    volEdgeList  - List of edge indices forming each volume element
    volSingEdgeList - List of singular edge indices forming each volume element
    volFaceList  - Face numbers associated with each volume element
    faceVolList  - Volume element numbers associated with each face
    volEleType   - Element type for the volume
    portList     - List of global (including PEC nodes) node indices forming triangles along the port
    portEdgeList - List of edge indices forming triangles along the port
    portFaceList - Face numbers associated with each port triangle. Only needed for basisOrder = 2
    nodeList     - Column 0 contains the node number (excluding PEC) corresponding to the global 
                   node number (including PEC) of that row index. Columns 1-3 provide the x,y,z coordinates
                   of the node.
    singEleLists - Dictionary containing information about singular lines and basis functions
    basisMaps    - Contains information on how basis functions map to geometric entities
    portP        - Contains info on the properties of the each port
    quadOrder    - The order of Gaussian Quadrature to use (for surface integrals).
    basisOrder   - Order of basis functions to use (currently 1 and 2 are supported).
    simType      - Indicates whether to use the field-based solver ("E"), potential-based solver ("A"),
                   or tree-cotree splitting ("TCS")

  Outputs:
    f            - Source vector
    Ree          - Matrix representing edge-edge contribution to the resistance of each port
    Ren          - Matrix representing edge-node contribution to the resistance of each port 
                  (only used if simType == "A" or "TCS")
    Ren          - Matrix representing node-edge contribution to the resistance of each port 
                  (only used if simType == "TCS")
    Rnn          - Matrix representing node-node contribution to the resistnance of each port
                  (only used if simType == "TCS")
  '''

  vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap = [basisMaps["vecEdgeMap"], basisMaps["vecSingEdgeMap"], basisMaps["vecFaceMap"], basisMaps["vecVolMap"]]
  scalNodeMap, scalEdgeMap = [basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"]]

  if volEleType == "Tet":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 6
        numScalBasis = 4
        numSingCornerBasis = 18#15 # Number of basis functions for a singular corner element
        numSingLineBasis = 18#16 # Number of basis functions for a singular line element
        numSingPointBasis = 12 # Number of basis functions for a singular point element
      case 2:
        numVecBasis = 20 
        numScalBasis = 10
        numSingCornerBasis = 65#56 # Number of basis functions for a singular corner element
        numSingLineBasis = 61#56 # Number of basis functions for a singular line element
        numSingPointBasis = 40 # Number of basis functions for a singular point element
    
    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
  elif volEleType == "Wedge":
    match basisOrder: # Handle different basis function numbering for higher order cases
      case 1:
        numVecBasis = 9
        numScalBasis = 6
      case 2:
        numVecBasis = 22
        numScalBasis = 12 # 15
    numSingCornerBasis = 0 # Number of basis functions for a singular corner element
    numSingLineBasis = 0 # Number of basis functions for a singular line element
    numSingPointBasis = 0 # Number of basis functions for a singular point element
    # Record the local node numbers forming each local edge
    edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]])

  numVecUn, numScalUn = getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap)

  ## Retrieve mappings for singular elements
  singCornerEleMap, singLineEleMap, singPointEleMap \
      = [singEleLists["singCornerEleMap"], singEleLists["singLineEleMap"], singEleLists["singPointEleMap"]]
  if len(singEleLists["singPointTets"]) != 0:
    singCornerCoeffs, singLineCoeffs, singPointCoeffs \
      = [singEleLists["singCornerCoeffs"], singEleLists["singLineCoeffs"], singEleLists["singPointCoeffs"]]
  
  numAppend = int(1e6) # number of zeros to append to each vector each time it needs to be increased
  rows = np.zeros((numAppend, ), dtype=np.int32)
  cols = np.zeros((numAppend, ), dtype=np.int32)
  Ree_vals = np.zeros((numAppend, ), dtype=np.float64)

  if simType == "A" or simType == "TCS": # Need additional nodal matrices if A-phi or TCS is used
    rows_Ren = np.zeros((numAppend, ), dtype=np.int32)
    cols_Ren = np.zeros((numAppend, ), dtype=np.int32)
    Ren_vals = np.zeros((numAppend, ), dtype=np.float64)
    # if simType == "TCS":
    rows_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    cols_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    Rnn_vals = np.zeros((numAppend, ), dtype=np.float64)
  
  ind = 0
  curLen = numAppend

  if simType == "A" or simType == "TCS":
    ind_Ren = 0
    curLen_Ren = numAppend

    # if simType == "TCS":
    ind_Rnn = 0
    curLen_Rnn = numAppend

  if simType == "E":
    f = np.zeros((numVecUn, ), dtype=np.float64) # Source vector
  elif simType == "A" or simType == "TCS":
    f = np.zeros((numVecUn+numScalUn, ), dtype=np.float64) # Source vector

  # Initialize vectors for storage
  voltVecVals = np.zeros((4*(numVecBasis)*len(portList),), dtype=np.float64) # Store values for calculating the voltage along the port
  voltVecInds = np.zeros(voltVecVals.shape, dtype=np.int32) # Store indices for calculating the voltage along the port
  if simType == "A" or simType == "TCS":
    voltScalVals = np.zeros((4*(numScalBasis)*len(portList),), dtype=np.float64) 
  else:
    scalInts = []
    voltScalVals = np.zeros((0,), dtype=np.float64) 
  voltScalInds = np.zeros(voltScalVals.shape, dtype=np.int32) # Store indices for calculating the voltage along the port
  vecInd, scalInd = [0, 0]

  # Loop through the set of triangles on the current port
  for triInd, triFace in enumerate(portFaceList):
    triNodes = portList[triInd]
    triCoords = nodeList[triNodes][:,1:]
    triEdges = portEdgeList[triInd]
    triFace = portFaceList[triInd]
    volEleNums = faceVolList[triFace]

    # Loop through tets touching the port and add integral values
    for locFaceNum, eleInd in enumerate(volEleNums):
      if eleInd == -1: # -1 marks cases where there's only 1 neighboring tet (typically due to being on PEC)
        continue
      if locFaceNum == 1:
        continue # Ignore the second face as it's tangentially zero

      volEleNodes = volEleList[eleInd]
      nodeNums = nodeList[volEleNodes, 0]
      volEleCoords = nodeList[volEleNodes, 1:]
      volEleEdges = volEdgeList[eleInd]
      volEleFaces = volFaceList[eleInd]

      # if locFaceNum == 1: 
      #   # If this is the second tet, eliminate the degrees of freedom associated 
      #   # with the shared face to avoid counting them twice
      #   locTriNodes = np.array([np.where(triNodes[ind] == volEleNodes)[0][0] for ind in range(len(triNodes))])
      #   nodeNums[locTriNodes] = -1
      #   locEdgeNums = np.array([[locTriNodes[0], locTriNodes[1]], [locTriNodes[0], locTriNodes[2]], [locTriNodes[1], locTriNodes[2]]])
      #   locTriEdges = np.array([np.where(np.all(locEdgeNums[ind] == edgeNums, axis=1))[0][0] for ind in range(len(triEdges))])
      #   volEleEdges[locTriEdges] = -1
      #   if triFace != -1:
      #     locTriFace = np.where(volEleFaces == triFace)[0]
      #     volEleFaces[locTriFace] = -1

      # Check whether the current element is singular or not and get indices for the rows and columns
      vecInds, scalInds, singType = selectInds3D(basisOrder, volEleType, scalNodeMap, scalEdgeMap, vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, \
                                                  volEleList[eleInd], volEdgeList[eleInd], volSingEdgeList[eleInd], volFaceList[eleInd], eleInd, nodeNums, edgeNums, \
                                                  numVecBasis, numScalBasis, numSingCornerBasis, numSingLineBasis, numSingPointBasis, singCornerEleMap[eleInd], singLineEleMap[eleInd], singPointEleMap[eleInd], simType)
      numVecInds, numScalInds = [len(vecInds), len(scalInds)]

      # Set up integration points
      if singType != "None":
        # For singular elements, use high order quadrature
        loc = np.array(q.gauss_quadrature_trisimp_init(9, triCoords[0], triCoords[1], triCoords[2]))
        if singType == "Corner":
          singCoeffs = singEleLists["singCornerCoeffs"][singCornerEleMap[eleInd]]
        elif singType == "Line":
          singCoeffs = singEleLists["singLineCoeffs"][singLineEleMap[eleInd]]
        elif singType == "Point":
          singCoeffs = singEleLists["singPointCoeffs"][singPointEleMap[eleInd]]
      else:
        loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, triCoords[0], triCoords[1], triCoords[2]))
        singCoeffs = []

      # Evaluate basis functions at each integration point
      if volEleType == "Tet":
        vecVals, curlVals, scalVals, gradVals = basisSetupTet(volEleCoords, loc, simType, basisOrder, edgeNums, singType=singType, singCoeffs=singCoeffs)
        
      elif volEleType == "Wedge":
        vertLoc = np.array([0.0], dtype=np.float64) # TODO: Make sure this actually works
        triLoc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, triCoords[0], triCoords[1], triCoords[2]))
        vertDir = portP["normal"]
        vecVals, curlVals, scalVals, gradVals = basisSetupWedge(volEleCoords, volEleList[eleInd], volEdgeList[eleInd], triLoc, vertLoc, vertDir, simType, basisOrder)

      # Record integrals of each basis function along the port surface
      vecInts = np.zeros((numVecInds,), dtype=np.float64)
      for ii in range(numVecInds):
        vecInts[ii] = q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], vecVals[ii] @ portP["direction"])
        if np.isclose(vecInts[ii], 0.0, atol=1e-10):
          vecInts[ii] = 0.0 # If very close to zero, set to exactly zero to avoid reduced sparsity
      
      if simType == "A":
        scalInts = np.zeros((numScalInds,), dtype=np.float64)
        for ii in range(numScalInds):
          if basisOrder == 1: # Handle 1st order case seperately b/c the gradient is constant and defined at only a single point
            scalInts[ii] = q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], np.array(list(repeat(gradVals[ii] @ portP["direction"], len(loc)))))
          else:
            scalInts[ii] = q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], gradVals[ii] @ portP["direction"])
          if np.isclose(scalInts[ii], 0.0, atol=1e-10):
            scalInts[ii] = 0.0 # If very close to zero, set to exactly zero to avoid reduced sparsity

      # Form source vector and vectors for measuring voltage
      for ii in range(numVecInds):
        rowVal = vecInds[ii]
        if rowVal == -1:
          continue
        
        voltVecVals[vecInd] = -vecInts[ii] / portP["width"]
        voltVecInds[vecInd] = rowVal
        vecInd += 1
        
        if portP["Vsmag"] != 0.0:
          if portP["type"].lower() != "open":
            f[rowVal] += portP["Vsmag"]*vecInts[ii] / (portP["Z"]*portP["width"]) 
          else:
            f[rowVal] += vecInts[ii] / portP["width"]

      for ii in range(numScalInds):
        rowVal = scalInds[ii]
        if rowVal == -1:
          continue
        
        voltScalVals[scalInd] = -scalInts[ii] / portP["width"]
        voltScalInds[scalInd] = rowVal
        scalInd += 1
        
        if simType == "TCS" and portP["Vsmag"] != 0.0: # A scalar source term only exists in tree-cotree
          if portP["type"].lower() != "open":
            f[rowVal+numVecUn] += portP["Vsmag"]*scalInts[ii] / (portP["Z"]*portP["width"]) 
          else:
            f[rowVal+numVecUn] += scalInts[ii] / portP["width"]
          
      # If the port has impedance, add values to those matrices
      if portP["type"].casefold() != "open".casefold():
        for ii in range(numVecBasis):
          rowVal = vecInds[ii]
          if colVal == -1:
            continue

          for jj in range(numVecBasis):
            colVal = vecInds[jj]
            if colVal == -1:
              continue

            if ind == curLen:
              rows = np.append(rows, np.zeros((numAppend,)))
              cols = np.append(cols, np.zeros((numAppend,)))
              Ree_vals = np.append(Ree_vals, np.zeros((numAppend,), dtype=np.float64))
              curLen += numAppend
            
            rows[ind] = rowVal
            cols[ind] = colVal
            Ree_vals[ind] = vecInts[ii] * vecInts[jj] / (portP["Z"]*portP["width"]**2)
            ind += 1

          if simType == "A" or simType == "TCS": # Also loop through nodes for A-phi
            for jj in range(numScalBasis):
              colVal = scalInds[jj]
              if colVal == -1:
                continue

              if ind_Ren == curLen_Ren:
                rows_Ren = np.append(rows_Ren, np.zeros((numAppend,)))
                cols_Ren = np.append(cols_Ren, np.zeros((numAppend,)))
                Ren_vals = np.append(Ren_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Ren += numAppend
            
              rows_Ren[ind_Ren] = rowVal
              cols_Ren[ind_Ren] = colVal

              Ren_vals[ind_Ren] = vecInts[ii] * scalInts[jj] / (portP["Z"]*portP["width"]**2)
              ind_Ren += 1

      if simType == "TCS": # For TCS, a node-node matrix is also needed
        for ii in range(numScalBasis):
          rowVal = scalInds[ii]
          if rowVal == -1:
            continue

          for tri_j in range(len(portList)):
            for jj in range(numScalBasis):
              colVal = scalInds[jj]
              if colVal == -1:
                continue

              if ind_Rnn == curLen_Rnn:
                rows_Rnn = np.append(rows_Rnn, np.zeros((numAppend,)))
                cols_Rnn = np.append(cols_Rnn, np.zeros((numAppend,)))
                Rnn_vals = np.append(Rnn_vals, np.zeros((numAppend,), dtype=np.float64))
                curLen_Rnn += numAppend
            
              rows_Rnn[ind_Rnn] = rowVal
              cols_Rnn[ind_Rnn] = colVal

              Rnn_vals[ind_Rnn] = scalInts[ii] * scalInts[jj] / (portP["Z"]*portP["width"]**2)
              ind_Rnn += 1

  ## Condense the voltVec vectors by consolidating entries pointing to the same index
  voltVecVals = voltVecVals[:vecInd]
  voltVecInds = voltVecInds[:vecInd]
  voltVecSort = np.argsort(voltVecInds)
  voltVecVals = voltVecVals[voltVecSort]
  voltVecInds = voltVecInds[voltVecSort]
  newVoltVecInds = np.unique(voltVecInds[voltVecSort])
  newVoltVecVals = np.zeros((len(newVoltVecInds),), dtype=np.float64)
  ind = 0
  for ii in range(len(newVoltVecVals)):
    while ind != vecInd and newVoltVecInds[ii] == voltVecInds[ind]:
      newVoltVecVals[ii] += voltVecVals[ind]
      ind += 1
  nzInds = np.where(newVoltVecVals != 0.0)[0] # Eliminate the zero indices
  newVoltVecVals = newVoltVecVals[nzInds]
  newVoltVecInds = newVoltVecInds[nzInds]

  if simType == "A" or simType == "TCS":
    voltScalVals = voltScalVals[:scalInd]
    voltScalInds = voltScalInds[:scalInd]
    voltScalSort = np.argsort(voltScalInds)
    voltScalVals = voltScalVals[voltScalSort]
    voltScalInds = voltScalInds[voltScalSort]
    newVoltScalInds = np.unique(voltScalInds[voltScalSort])
    newVoltScalVals = np.zeros((len(newVoltScalInds),), dtype=np.float64)
    ind = 0
    for ii in range(len(newVoltScalVals)):
      while ind != scalInd and newVoltScalInds[ii] == voltScalInds[ind]:
        newVoltScalVals[ii] += voltScalVals[ind]
        ind += 1
    nzInds = np.where(newVoltScalVals != 0.0)[0] # Eliminate the zero indices
    newVoltScalVals = newVoltScalVals[nzInds]
    newVoltScalInds = newVoltScalInds[nzInds]
  else:
    newVoltScalInds, newVoltScalVals = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float64)]
          
  rows = rows[0:ind]
  cols = cols[0:ind]
  Ree_vals = Ree_vals[0:ind]

  Ree = spa.csr_array((Ree_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)

  if simType == "E":
    matArr = [f, Ree]
  elif simType == "A" or simType == "TCS":
    rows_Ren = rows_Ren[0:ind_Ren]
    cols_Ren = cols_Ren[0:ind_Ren]
    Ren_vals = Ren_vals[0:ind_Ren]
    Ren = spa.csr_array((Ren_vals, (rows_Ren, cols_Ren)), shape=(numVecUn, numScalUn), dtype=np.float64)
    
    if simType == "A":
      matArr = [f, Ree, Ren]
    elif simType == "TCS":

      rows_Rnn = rows_Rnn[0:ind_Rnn]
      cols_Rnn = cols_Rnn[0:ind_Rnn]
      Rnn_vals = Rnn_vals[0:ind_Rnn]

      Rne = spa.csr_array((Ren_vals, (cols_Ren, rows_Ren)), shape=(numScalUn, numVecUn), dtype=np.float64)
      Rnn = spa.csr_array((Rnn_vals, (rows_Rnn, cols_Rnn)), shape=(numScalUn, numScalUn), dtype=np.float64)

      matArr = [f, Ree, Ren, Rne, Rnn]

  return newVoltVecVals, newVoltVecInds, newVoltScalVals, newVoltScalInds, matArr
    
def assembleImpSurfMats(triList, triEdgeList, triFaceList, nodeList, vecEdgeMap, vecFaceMap, scalNodeMap, scalEdgeMap, surfP, \
                        basisOrder, quadOrder, simType, numVecUn, numScalUn):
  '''
  A function to create matrices for an surface with an impedance boundary condition

  Inputs:
    triList      - List of global (i.e.: including PEC nodes) node indices forming triangles along the surface
    triEdgeList  - List of edge indices forming triangles along the surface
    triFaceList  - Face numbers associated with each surface triangle. Only needed for basisOrder = 2
    nodeList     - Column 0 contains the node number (excluding PEC) corresponding to the global 
                   node number (including PEC) of that row index. Columns 1-3 provide the x,y,z coordinates
                   of the node.
    vecEdgeMap   - Provides the matrix indices for each edge number for vector basis functions
    vecFaceMap   - Provides the matrix indices for each face number for vector basis functions
    scalNodeMap  - Provides the matrix indices for each node number for scalar basis functions
    scalEdgeMap  - Provides the matrix indices for each edge number for scalar basis functions
    surfP        - Dictionary containing parameters of the impedance surface
    basisOrder   - Order of basis functions to use (currently 1 and 2 are supported).
    quadOrder    - The order of Gaussian Quadrature to use (for surface integrals).
    simType      - Indicates whether to use the field-based solver ("E"), potential-based solver ("A"),
                   or tree-cotree splitting ("TCS")
    numVecUn     - Total number of vector unknowns in the mesh
    numScalUn    - Total number of scalar unknowns in the mesh

  Outputs:
    Ree          - Matrix representing edge-edge contribution to the resistance of the surface
    Ren          - Matrix representing edge-node contribution to the resistance of the surface
                   (only used if simType == "A" or "TCS")
    Ren          - Matrix representing node-edge contribution to the resistance of the surface
                   (only used if simType == "TCS")
    Rnn          - Matrix representing node-node contribution to the resistnance of the surface
                   (only used if simType == "TCS")
  '''

  numAppend = int(1e6) # number of zeros to append to each vector each time it needs to be increased
  rows = np.zeros((numAppend, ), dtype=np.int32)
  cols = np.zeros((numAppend, ), dtype=np.int32)
  Ree_vals = np.zeros((numAppend, ), dtype=np.float64)

  if simType == "A" or simType == "TCS": # Need additional nodal matrices if A-phi or TCS is used
    rows_Ren = np.zeros((numAppend, ), dtype=np.int32)
    cols_Ren = np.zeros((numAppend, ), dtype=np.int32)
    Ren_vals = np.zeros((numAppend, ), dtype=np.float64)
    # if simType == "TCS":
    rows_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    cols_Rnn = np.zeros((numAppend, ), dtype=np.int32)
    Rnn_vals = np.zeros((numAppend, ), dtype=np.float64)
  
  ind = 0
  curLen = numAppend

  if simType == "A" or simType == "TCS":
    ind_Ren = 0
    curLen_Ren = numAppend

    # if simType == "TCS":
    ind_Rnn = 0
    curLen_Rnn = numAppend

  match basisOrder: # Handle different basis function numbering for higher order cases
    case 1:
      numVecBasis = 3
      numScalBasis = 3
    case 2:
      numVecBasis = 8
      numScalBasis = 6

  edgeNums = np.array([[0,1], [0,2], [1,2]])
  zVec = np.array([0, 0, 1.0], dtype=np.float64)
  
  for triInd, tri in enumerate(triList):
    nodeNums = nodeList[tri,0] # Node numbers (with PEC marked) for the triangle
    coords = nodeList[tri,1:] # Coordinates for the 3 nodes of the triangle
    coords = rotateCoords2D(coords) # Rotate the coordinates to be normal to z
    loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, coords[0], coords[1], coords[2])) # Determine integration points in the triangle

    ## Save basis functions for the current triangle
    if basisOrder == 1:
      edgeBasis = [basis_vector_numba(coords, edgeNums[ii,0], edgeNums[ii,1], loc)[0] for ii in range(3)]
      if simType == "A" or simType == "TCS":
        gradBasis = [basis2Dscalar_numba(coords, nodeInd, loc)[1] for nodeInd in range(3)]
    elif basisOrder == 2:
      edgeBasis, _ = basis2DVquad(coords, loc)
      if simType == "A" or simType == "TCS":
        _, gradBasis = basis2DSquad(coords, loc)

    for ii in range(numVecBasis):
      rowVal = selectVecInds2D(basisOrder, ii, vecEdgeMap, vecFaceMap, triEdgeList[triInd], triFaceList[triInd])
      if rowVal == -1:
        continue
      
      for jj in range(numVecBasis):
        colVal = selectVecInds2D(basisOrder, jj, vecEdgeMap, vecFaceMap, triEdgeList[triInd], triFaceList[triInd])
        if colVal == -1:
          continue

        if ind == curLen:
          rows = np.append(rows, np.zeros((numAppend,)))
          cols = np.append(cols, np.zeros((numAppend,)))
          Ree_vals = np.append(Ree_vals, np.zeros((numAppend,), dtype=np.float64))
          curLen += numAppend
        
        rows[ind] = rowVal
        cols[ind] = colVal
        Ree_vals[ind] = (1/surfP["Z"]) * q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.sum(np.cross(zVec, edgeBasis[ii]) * np.cross(zVec, edgeBasis[jj]), axis=1))
        ind += 1

      if simType == "A" or simType == "TCS": # Also loop through nodes for A-phi
        for jj in range(numScalBasis):
          colVal = selectScalInds2D(jj, scalNodeMap, scalEdgeMap, triList[triInd], triEdgeList[triInd], nodeNums, edgeNums)
          if colVal == -1:
            continue

          if ind_Ren == curLen_Ren:
            rows_Ren = np.append(rows_Ren, np.zeros((numAppend,)))
            cols_Ren = np.append(cols_Ren, np.zeros((numAppend,)))
            Ren_vals = np.append(Ren_vals, np.zeros((numAppend,), dtype=np.float64))
            curLen_Ren += numAppend
        
          rows_Ren[ind_Ren] = rowVal
          cols_Ren[ind_Ren] = colVal

          Ren_vals[ind_Ren] = (1/surfP["Z"]) * q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.sum(np.cross(zVec, edgeBasis[ii]) * np.cross(zVec, gradBasis[jj]), axis=1))
          ind_Ren += 1

    if simType == "TCS": # For TCS, a node-node matrix is also needed
      for ii in range(numScalBasis):
        rowVal = selectScalInds2D(ii, scalNodeMap, scalEdgeMap, triList[triInd], triEdgeList[triInd], nodeNums, edgeNums)
        if rowVal == -1:
          continue

        for jj in range(numScalBasis):
          colVal = selectScalInds2D(jj, scalNodeMap, scalEdgeMap, triList[triInd], triEdgeList[triInd], nodeNums, edgeNums)
          if colVal == -1:
            continue

          if ind_Rnn == curLen_Rnn:
            rows_Rnn = np.append(rows_Rnn, np.zeros((numAppend,)))
            cols_Rnn = np.append(cols_Rnn, np.zeros((numAppend,)))
            Rnn_vals = np.append(Rnn_vals, np.zeros((numAppend,), dtype=np.float64))
            curLen_Rnn += numAppend
        
          rows_Rnn[ind_Rnn] = rowVal
          cols_Rnn[ind_Rnn] = colVal

          if basisOrder == 1:
            Rnn_vals[ind_Rnn] = (1/surfP["Z"]) * q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.array(list(repeat(np.sum(np.cross(zVec, gradBasis[ii]) * np.cross(zVec, gradBasis[jj])), len(edgeBasis[0])))))
          else:
            Rnn_vals[ind_Rnn] = (1/surfP["Z"]) * q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.sum(np.cross(zVec, gradBasis[ii]) * np.cross(zVec, gradBasis[jj]), axis=1))
          ind_Rnn += 1
          
  rows = rows[0:ind]
  cols = cols[0:ind]
  Ree_vals = Ree_vals[0:ind]

  Ree = spa.csr_array((Ree_vals, (rows, cols)), shape=(numVecUn, numVecUn), dtype=np.float64)

  if simType == "E":
    return [Ree] # Put it in a list to ensure consistency between methods
  elif simType == "A" or simType == "TCS":
    rows_Ren = rows_Ren[0:ind_Ren]
    cols_Ren = cols_Ren[0:ind_Ren]
    Ren_vals = Ren_vals[0:ind_Ren]
    Ren = spa.csr_array((Ren_vals, (rows_Ren, cols_Ren)), shape=(numVecUn, numScalUn), dtype=np.float64)
    
    if simType == "A":
      return Ree, Ren
    elif simType == "TCS":

      rows_Rnn = rows_Rnn[0:ind_Rnn]
      cols_Rnn = cols_Rnn[0:ind_Rnn]
      Rnn_vals = Rnn_vals[0:ind_Rnn]

      Rne = spa.csr_array((Ren_vals, (cols_Ren, rows_Ren)), shape=(numScalUn, numVecUn), dtype=np.float64)
      Rnn = spa.csr_array((Rnn_vals, (rows_Rnn, cols_Rnn)), shape=(numScalUn, numScalUn), dtype=np.float64)

      return Ree, Ren, Rne, Rnn
    
def rotateCoords2D(coords, normVec=np.zeros((3,), dtype=np.float64), returnMat=False):
  """
  Rotates the coordinates of a triangle to be normal to the z-axis

  Inputs:
    coords    - The x, y, and z coordinates of the three nodes of the triangle
    normVec   - The unit vector normal to the triangle
    returnMat - A flag which indicates whether to return the rotation matrix

  Outputs:
    newCoords - The rotated coordinates
    rotMat    - The rotation matrix (only returned if requested)
  """
  newX = coords[0,:] - coords[1,:] # This edge will become the x-axis
  newX /= np.linalg.norm(newX)
  if np.all(normVec == 0): # If the normal isn't provided, find it
    altEdge = coords[0,:] - coords[2,:]
    normVec = np.cross(newX, altEdge)
    normVec /= np.linalg.norm(normVec)
  newY = np.cross(normVec, newX) # This edge will become the y-axis
  rotMat = np.array([newX, newY, normVec])
  
  if returnMat:
    return (rotMat @ coords.T).T, rotMat
  else:
    return (rotMat @ coords.T).T
  
def selectInds3D(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, \
                 scalNodeMap, scalEdgeMap, vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, \
                 eleNodes, eleEdges, eleSingEdges, eleFaces, eleInd, nodeNums, edgeNums, \
                 numVecBasis, numScalBasis, N_tet, singFlag, singCoeffs, simType):
  """
  Function to unify row / column selection between singular and non-singular elements. 
  
  See selectVecInds3D and selectVecInds3D_Singular_orderN for details on other arguments
  """

  if singFlag != -1:
    vecInds = selectVecInds3D_Singular_orderN(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, singCoeffs, \
                                                               vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, eleNodes, eleEdges, \
                                                               eleSingEdges, eleFaces, eleInd, N_tet)
  else:
    if simType == "E" or simType == "A" or simType == "TCS":
      # vecInds = [selectVecInds3D(basisOrder, ii, eleType, vecEdgeMap, vecFaceMap, eleEdges, eleFaces) for ii in range(numVecBasis)]
      vecInds = selectVecInds3D_orderN(basisOrder, basisOrder_long, eleType, vecEdgeMap, vecFaceMap, vecVolMap, eleEdges, eleFaces, eleInd, N_tet)
    if simType == "A" or simType == "TCS" or simType == "Phi":
      scalInds = [selectScalInds3D(basisOrder, ii, scalNodeMap, scalEdgeMap, eleType, eleNodes, eleEdges, nodeNums, edgeNums) for ii in range(numScalBasis)]
  if simType == "E":
    scalInds = []
  elif simType == "Phi":
    vecInds = []
    vecBasisSigns = []

  return vecInds, scalInds
  
@jit
def selectVecInds3D(basisOrder, locInd, eleType, vecEdgeMap, vecFaceMap, eleEdges, eleFaces):
  """
  A function for determining the correct row or column to be adding a matrix entry to
  given the index for the current element and a local vector basis function

  Inputs:
    basisOrder     - Order of the basis functions. Currently 1 and 2 are supported.
    locInd         - Local index of the current edge basis function (0-5 for order=1 tets,
                     0-19 for order=2 tets, 0-7 for order=1 wedges, 0-18 for order=2 wedges)
    eleType        - Type of elements being used (either "Tet" or "Wedge")
    vecEdgeMap     - Provides the matrix indices for each edge number
    vecFaceMap     - Provides the matrix indices for each face number
    eleEdges       - List of edge numbers associated with the current element
    eleFaces       - List of face numbers associated with the current element (only for order=2)
  Outputs:
    globInd        - Global index to use for the row / column of the matrix. Set to -1 if the basis function shouldn't be used
  """

  if basisOrder == 1: # 1st order edges
    if (eleEdges[locInd] <= -1): # If the edge is on PEC, do not use
      globInd = -1
    else:
      globInd = vecEdgeMap[eleEdges[locInd], 0]
  elif basisOrder == 2:
    if eleType == "Tet":
      if locInd < 12: # 2nd order edges
        locEdgeNum = int(np.floor(locInd/2)) # Local edge number
        if (eleEdges[locEdgeNum] <= -1): # If the edge is on PEC, do not use
          globInd = -1
        else:
          globInd = vecEdgeMap[eleEdges[locEdgeNum], int(np.mod(locInd,2))]
      elif locInd >= 12 and locInd < 20: # Faces
        locFaceNum = int(np.floor((locInd-12)/2))
        if eleFaces[locFaceNum] == -1: # Ignore faces that lie on PEC
          globInd = -1
        else:
          globInd = vecFaceMap[eleFaces[locFaceNum],np.mod(locInd,2)] # Also have 2 allowed per face
    elif eleType == "Wedge":
      if locInd < 6: # 2nd order edges for the bottom triangle
        locEdgeNum = int(np.floor(locInd/2))
        if (eleEdges[locEdgeNum] <= -1): # If the edge is on PEC, do not use
          globInd = -1
        else:
          globInd = vecEdgeMap[eleEdges[locEdgeNum],np.mod(locInd,2)] # Set the values so 2 are allowed per edge
      elif locInd >= 6 and locInd < 8: # 2nd order faces for the bottom triangle
        locFaceNum = 0
        if eleFaces[locFaceNum] <= -1: # Ignore faces that lie on PEC
          globInd = -1
        else:
          globInd = vecFaceMap[eleFaces[locFaceNum],np.mod(locInd,2)] # Also have 2 allowed per face
      elif locInd >= 8 and locInd < 14: # 2nd order edges for the top triangle
        locEdgeNum = int(np.floor(locInd/2)) - 1
        if (eleEdges[locEdgeNum] <= -1): # If the edge is on PEC, do not use
          globInd = -1
        else:
          globInd = vecEdgeMap[eleEdges[locEdgeNum],np.mod(locInd,2)] # Set the values so 2 are allowed per edge
      elif locInd >= 14 and locInd < 16: # 2nd order faces for the top triangle
        locFaceNum = 1
        if eleFaces[locFaceNum] <= -1: # Ignore faces that lie on PEC
          globInd = -1
        else:
          globInd = vecFaceMap[eleFaces[locFaceNum],np.mod(locInd,2)] # Also have 2 allowed per face
      elif locInd >= 16 and locInd < 19: # 2nd order edges for the longitudinal direction
        locEdgeNum = locInd - 10
        if (eleEdges[locEdgeNum] <= -1): # If the edge is on PEC, do not use
          globInd = -1
        else:
          globInd = vecEdgeMap[eleEdges[locEdgeNum],0]
      elif locInd >= 19 and locInd < 22: # 2nd order faces for the longitudinal direction
        locFaceNum = locInd - 17
        if eleFaces[locFaceNum] == -1: # Ignore faces that lie on PEC
          globInd = -1
        else:
          globInd = vecFaceMap[eleFaces[locFaceNum],0]


  return globInd

@jit
def selectVecInds3D_orderN(basisOrder, basisOrder_long, eleType, vecEdgeMap, vecFaceMap, vecVolMap, eleEdges, eleFaces, eleInd, N_tet):
  """
  A function for determining the correct rows and columns to add matrix 
  entries for the basis functions in the current element.
  This version is generalized to work with arbitrary order basis functions.

  Inputs:
    basisOrder      - Order of the basis functions
    basisOrder_long - Order of the basis functions in the longitudinal direction (for wedges)
    eleType         - Type of elements being used (either "Tet" or "Wedge")
    vecEdgeMap      - Provides the matrix indices for each edge number
    vecFaceMap      - Provides the matrix indices for each face number
    vecVolMap       - Provides the matrix indices for each element index
    eleEdges        - List of edge numbers associated with the current element
    eleFaces        - List of face numbers associated with the current element 
  Outputs:
    globInd         - Global index to use for the row / column of the matrix. Set to -1 if the basis function shouldn't be used
  """

  ## Enumerate each type of basis function
  numEdgeBasis = basisOrder # Number of vector basis functions on each triangular edge
  numLongEdgeBasis = basisOrder_long # Number of vector basis functions on each longitudinal / vertical edge
  numFaceBasis = basisOrder*(basisOrder-1) # Number of vector basis functions on each triangular face
  numQuadFaceBasis = basisOrder*(basisOrder_long-1) # Number of transverse vector basis functions on each quadrilateral face
  numQuadFaceBasis_long = basisOrder_long*(basisOrder-1)  # Number of longitudinal vector basis functions on each quadrilateral face

  if eleType == "Tet":
    numNodes, numEdges, numFaces = [4, 6, 4] # Number of nodes, edges, and faces in a tet
    numVolBasis = int(basisOrder*(basisOrder-1)*(basisOrder-2)/2)
    numTotBasis = int(basisOrder*(basisOrder+2)*(basisOrder+3)/2) # Total number of vector basis functions
  elif eleType == "Wedge":
    numNodes, numEdges, numFaces = [6, 9, 5] # Number of nodes, edges, and faces in a wedge
    numVolBasis = basisOrder*(basisOrder-1)*(basisOrder_long-1) # Number of transverse vector basis functions in each wedge volume
    numVolBasis_long = int(basisOrder_long*(basisOrder-1)*(basisOrder-2)/2) # Number of longitudinal vector basis functions in each wedge volume
    numTotBasis = basisOrder*(basisOrder+2)*(basisOrder_long+1) \
                  + int((basisOrder+1)*(basisOrder+2)*basisOrder_long/2) # Total number of vector basis functions

  globInds = -np.ones((numTotBasis,), dtype=np.int32)
  curInd = 0

  if eleType == "Tet":
    ## Add edge-associated functions
    for edgeInd in eleEdges:
      if edgeInd >= 0: # Make sure edge isn't on PEC
        globInds[curInd:curInd+numEdgeBasis] = vecEdgeMap[edgeInd, :numEdgeBasis]
      curInd += numEdgeBasis

    ## Add face-associated functions
    for faceInd in eleFaces:
      if faceInd >= 0: # Make sure face isn't on PEC
        globInds[curInd:curInd+numFaceBasis] = vecFaceMap[faceInd, :numFaceBasis]
      curInd += numFaceBasis

    ## Add volume-associated functions
    globInds[curInd:curInd+numVolBasis] = vecVolMap[eleInd, :numVolBasis]
    curInd += numVolBasis

  elif eleType == "Wedge":
    ## Loop through the bottom and top faces
    for ii in range(2):
      curTriEdges = eleEdges[3*ii:3*(ii+1)] # Triangular edges forming the current face
      # Add triangular edge-associated functions
      for edgeInd in curTriEdges:
        if edgeInd >= 0: # Make sure edge isn't on PEC
          globInds[curInd:curInd+numEdgeBasis] = vecEdgeMap[edgeInd, :numEdgeBasis]
        curInd += numEdgeBasis

      # Add triangular face-associated functions
      faceInd = eleFaces[ii]
      if faceInd >= 0: # Make sure face isn't on PEC
        globInds[curInd:curInd+numFaceBasis] = vecFaceMap[faceInd, :numFaceBasis]
      curInd += numFaceBasis

    ## Loop through transverse functions on quadrilateral faces / in the volume
    for ii in range(basisOrder_long-1):
      # Add transverse quadrilateral face-associated functions
      for faceInd in eleFaces[2:]:
        if faceInd >= 0: # Make sure the face isn't on PEC
          basisRange = np.arange(ii*numQuadFaceBasis, (ii+1)*numQuadFaceBasis)
          globInds[curInd:curInd+numQuadFaceBasis] = vecFaceMap[faceInd, basisRange]
        curInd += numQuadFaceBasis

      # Add transverse volume-associated functions
      basisRange = np.arange(ii*numVolBasis, (ii+1)*numVolBasis)
      globInds[curInd:curInd+numVolBasis] = vecVolMap[eleInd+N_tet, basisRange]
      curInd += numVolBasis

    ## Add longitudinal functions
    # Add longitudinal functions on longitudinal edges
    for edgeInd in eleEdges[6:]:
      if edgeInd >= 0: # Make sure edge isn't on PEC
        globInds[curInd:curInd+numLongEdgeBasis] = vecEdgeMap[edgeInd, :numLongEdgeBasis]
      curInd += numLongEdgeBasis

    # Add longitudinal functions on quadrilateral faces
    for faceInd in eleFaces[2:]:
      if faceInd >= 0: # Make sure the face isn't on PEC
        basisRange = (basisOrder_long-1)*numQuadFaceBasis + np.arange(numQuadFaceBasis_long)
        globInds[curInd:curInd+numQuadFaceBasis_long] = vecFaceMap[faceInd, basisRange]
      curInd += numQuadFaceBasis_long

    # Add longitudinal functions in the volume
    basisRange = (basisOrder_long-1)*numVolBasis + np.arange(numVolBasis_long)
    globInds[curInd:curInd+numVolBasis_long] = vecVolMap[eleInd+N_tet, basisRange]
    curInd += numVolBasis_long


  return globInds

def selectVecInds3D_Singular_orderN(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, singCoeffs, \
                                    vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, \
                                    eleNodes, eleEdges, eleSingEdges, eleFaces, eleInd, N_tet):
  """
  A function for determining the correct row or column to be adding a matrix entry to
  given the index for the current element and a local vector basis function for a singular element
  This version is generalized to work with arbitrary order basis functions (with order 
  seperately tunable for the nonsingular and singular functions)

  Inputs:
    basisOrder      - Order of the nonsingular basis functions
    singOrder       - Order of the singular basis functions
    basisOrder_long - Order of the nonsingular basis functions in the longitudinal direction (for wedges/prisms)
    singOrder_long  - Order of the singular basis functions in the longitudinal direction (for wedges/prisms)
    eleType         - Type of elements being used (currently "Tet" or "Wedge")
    singCoeffs      - Coefficients for the singular nodes in the tet
    vecEdgeMap      - Provides the matrix indices for each edge number
    vecSingEdgeMap  - Provides the matrix indices for each singular edge number
    vecFaceMap      - Provides the matrix indices for each face number
    vecVolMap       - Provides matrix indices for each element volume
    eleNodes        - List of node numbers associated with the current element
    eleEdges        - List of edge numbers associated with the current element
    eleSingEdges    - List of singular edge numbers associated with the current element
    eleFaces        - List of face numbers associated with the current element (only for order=2)
    eleInd          - Index of the current element
    N_tet           - Number of tetrahedra in the mesh
  Outputs:
    globInds        - Global indices to use for the rows / columns of the matrix for each basis function in the tet.
                      Set to -1 if the basis function shouldn't be used
  """

  if eleType == "Tet":
    ## Start by numbering the different types of functions
    numNodes, numEdges, numFaces = [4, 6, 4] # Number of nodes, edges, and faces in a tet
    numTriEdges, numTriFaces = [numEdges, numFaces] # Number of triangular edges and faces
    numStdBasis = int(basisOrder*(basisOrder+2)*(basisOrder+3)/2) # Total number of nonsingular functions
    singEdgeNums = np.array([[0,1,2], [0,3,4], [1,3,5], [2,4,5]], dtype=np.int32) # Edges touching each singular node
    singFaceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32) # Faces touching each singular node
    singFaceEdgeNums = np.array([[0,1], [0,2], [1,2], [0,3], [1,3], [2,3]], dtype=np.int32) # Faces touching each singular edge

    numVecEdgeStdBasis = basisOrder
    numVecFaceStdBasis = basisOrder*(basisOrder-1)
    numVecVolStdBasis = int(basisOrder*(basisOrder-1)*(basisOrder-2)/2)
  
    numVecEdgeSingBasis = singOrder # Number of singular functions on each edge
    numVecFaceSingBasis = int(singOrder*(singOrder-1)/2) # Number of singular functions on each face
    numVecFaceRotBasis = int((singOrder+1)*singOrder/2) # Number of rotational functions on each face
    numVecVolSingBasis = int((singOrder-2)*(singOrder-1)*singOrder/6) # Singular functions
    numVecVolRotBasis = int((singOrder-1)*singOrder*(singOrder+1)/6) # Rotational singular functions (face-derived)
    numVecVolTotBasis = numVecVolSingBasis + numVecVolRotBasis # Number of basis functions in the volume
  
    numVecFaceSingLineBasis = numVecFaceRotBasis # Number of singular line functions on each face
    numVecVolSingLineGradBasis = numVecVolRotBasis # Gradient edge-singular functions (face-derived)
    numVecVolSingLineRotBasis = singOrder + int((singOrder-1)*singOrder) + int((singOrder-2)*(singOrder-1)*singOrder/6) # Rotational edge-singular functions
    numVecVolSingLineTotBasis = numVecVolSingLineGradBasis + numVecVolSingLineRotBasis

    numSingPointBasis = 3*numVecEdgeSingBasis + 3*(numVecFaceSingBasis + numVecFaceRotBasis) \
                            + numVecVolSingBasis + 2*numVecVolRotBasis
    numSingLineBasis = 2*numVecFaceSingLineBasis + numVecVolSingLineTotBasis

  elif eleType == "Wedge":
    numNodes, numEdges, numFaces = [6, 9, 5] # Number of nodes, edges, and faces in a wedge
    numTriEdges, numTriFaces = [6, 2] # Number of triangular edges and faces
    numStdBasis = basisOrder*(basisOrder+2)*(basisOrder_long+1) \
                  + int((basisOrder+1)*(basisOrder+2)*basisOrder_long/2) # Total number of nonsingular functions
    singEdgeNums = np.array([[0,1,6], [0,2,7], [1,2,8], [3,4,6], [3,5,7], [4,5,8]], dtype=np.int32) # Edges touching each singular node
    singFaceNums = np.array([[0,2,3], [0,2,4], [0,3,4], [1,2,3], [1,2,4], [1,3,4]], dtype=np.int32) # Faces touching each singular node
    singFaceEdgeNums = np.array([[0,2], [0,3], [0,4], [1,2], [1,3], [1,4]], dtype=np.int32) # Faces touching each singular edge

    numVecEdgeStdBasis = basisOrder
    numVecLongEdgeStdBasis = basisOrder_long
    numVecFaceStdBasis = basisOrder*(basisOrder-1)
    numVecQuadFaceStdBasis = basisOrder*(basisOrder_long-1) \
                              + (basisOrder-1)*basisOrder_long # Number of standard (nonsingular) functions on each quadrilateral face
    numVecVolStdBasis = basisOrder*(basisOrder-1)*(basisOrder_long-1) \
                        + int((basisOrder-1)*(basisOrder-2)*basisOrder_long/2) # Number of standard (nonsingular) functions in the volume
  
    numVecEdgeSingBasis = singOrder # Number of singular functions on each triangular edge
    numVecLongEdgeSingBasis = singOrder_long # Number of singular functions on each longitudinal edge
    numVecFaceSingBasis = int(singOrder*(singOrder-1)/2) # Number of singular functions on each triangular face
    numVecFaceRotBasis = int((singOrder+1)*singOrder/2) # Number of rotational functions on each triangular face
    numVecQuadFaceSingBasis = singOrder*(singOrder_long-1) # Number of singular functions built from triangular edges on each quadrilateral face
    numVecQuadFaceSingBasis_long = (singOrder-1)*singOrder_long # Number of singular functions built from longitudinal edges on each quadrilateral face
    numVecQuadFaceRotBasis = singOrder*singOrder_long # Number of rotational functions on each quadrilateral face
    numVecVolSingBasis = int(singOrder*(singOrder-1)*(singOrder_long-1)/2) # Transverse singular functions
    numVecVolSingBasis_long =int((singOrder-1)*(singOrder-2)*singOrder_long/2) # Longitudinal singular functions
    numVecVolRotBasis = int((singOrder+1)*singOrder*(singOrder_long-1)/2) # Transverse rotational singular functions (face-derived)
    numVecVolRotBasis_long = int(singOrder*(singOrder-1)*singOrder_long/2) # Longitudinal rotational singular functions (face-derived)
    numVecVolTotBasis = numVecVolSingBasis+numVecVolRotBasis \
                        + numVecVolSingBasis_long+numVecVolRotBasis_long # Number of basis functions in the volume (including longitudinal functions)
  
    numVecFaceSingLineBasis = numVecFaceRotBasis # Number of singular line functions on each triangular face
    numVecQuadFaceSingLineBasis = numVecQuadFaceRotBasis # Number of singular line functions on each quadrilateral face
    numVecVolSingLineGradBasis = numVecVolRotBasis # Gradient edge-singular functions (triangular face-derived)
    numVecVolSingLineGradBasis_long = 0 # numVecVolRotBasis_long # Gradient edge-singular functions (quadrilateral face-derived)
                                        # NOTE: In the current formulation, these are fully dependent with combinations of edge-singular 
                                        # and node-singular face functions, so they are eliminated
    numVecVolSingLineRotBasis = 0 # Remove these functions 
                                  # TODO: Add these back once I have a better idea of what they would look like
    numVecVolSingLineRotBasis_long = 0 # Remove these functions 
                                       # TODO: Same as above
    numVecVolSingLineTotBasis = numVecVolSingLineGradBasis + numVecVolSingLineGradBasis_long \
                                  + numVecVolSingLineRotBasis + numVecVolSingLineRotBasis_long

    numSingPointBasis = 2*numVecEdgeSingBasis + numVecLongEdgeSingBasis + numVecFaceSingBasis \
                        + 2*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long) + numVecVolSingBasis + numVecVolSingBasis_long \
                        + numVecFaceRotBasis + 2*numVecQuadFaceRotBasis + numVecVolRotBasis + numVecVolRotBasis_long
    numSingLineBasis = numVecFaceSingLineBasis + numVecQuadFaceSingLineBasis + numVecVolSingLineTotBasis

  singNodeInds = np.where(singCoeffs != 0)[0] # Indices of the singular nodes in the element
  singEdgeInds = np.where(eleSingEdges != -1)[0] # Indices of the singular edges in the element
  
  numSingBasis = numNodes*numSingPointBasis + numTriEdges*numSingLineBasis
  globInds = -np.ones((numStdBasis+numSingBasis,), dtype=np.int32)

  ## Nonsingular functions
  # globInds[:numStdBasis] = \
  #   [selectVecInds3D(basisOrder, locInd, eleType, vecEdgeMap, vecFaceMap, eleEdges, eleFaces) for locInd in range(numStdBasis)]
  globInds[:numStdBasis] = \
    selectVecInds3D_orderN(basisOrder, basisOrder_long, eleType, vecEdgeMap, vecFaceMap, vecVolMap, eleEdges, eleFaces, eleInd, N_tet)
  
  ## Singular functions
  curInd = numStdBasis # Track the index of globInds being added to
  # Keep count of how many times each entity is visited to get the correct basis # assignment
  edgeCnt = np.zeros((numEdges,), dtype=np.int32) 
  faceCnt = np.zeros((numFaces,), dtype=np.int32)

  # Add functions for each singular point
  nodeSort = sortSingNodes(eleNodes, eleType) # Use the sorted nodes to correct for wedges with top nodes out of order
  singNodeSort = np.argsort(nodeSort[singNodeInds])
  for ii, nodeInd in enumerate(singNodeInds[singNodeSort]):
    curInd = numStdBasis + nodeInd*numSingPointBasis # Start index for the current singular node
    edgeSet = singEdgeNums[nodeInd]
    faceSet = singFaceNums[nodeInd]

    # Add gradient edge funcs
    for jj, edgeInd in enumerate(eleEdges[edgeSet]):
      if edgeSet[jj] < 6: # Tet edges or triangular edges of a wedge
        prevBasisCnt = numVecEdgeStdBasis
        curBasisCnt = numVecEdgeSingBasis
      else: # Longitudinal edges of a wedge
        prevBasisCnt = numVecLongEdgeStdBasis
        curBasisCnt = numVecLongEdgeSingBasis
      basisRange = prevBasisCnt + edgeCnt[edgeSet[jj]]*curBasisCnt \
                  + np.arange(curBasisCnt)
       
      # Make sure the edge isn't on PEC
      if edgeInd >= 0:
        globInds[curInd:(curInd+curBasisCnt)] = vecEdgeMap[edgeInd,basisRange]
      curInd += curBasisCnt

      if eleType == "Wedge" and edgeSet[jj] < 6 and singOrder_long > 1:
        # Also add quadrilateral face-associated functions built from the triangular edge
        quadFaceInd = np.mod(edgeSet[jj], 3)
        prevBasisCnt = numVecQuadFaceStdBasis + faceCnt[quadFaceInd+2]*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis)
        curBasisCnt = numVecQuadFaceSingBasis
        basisRange = prevBasisCnt + np.arange(curBasisCnt)

        # Make sure the quadrilateral face isn't on PEC
        if eleFaces[quadFaceInd+2] != -1:
          globInds[curInd:(curInd+curBasisCnt)] = vecFaceMap[eleFaces[quadFaceInd+2],basisRange]
        curInd += curBasisCnt

    edgeCnt[edgeSet] += 1 # Update the set of visited edges

    # Add gradient face funcs
    for jj, faceInd in enumerate(eleFaces[faceSet]): 
      if eleType == "Wedge" and faceSet[jj] >= 2: # Quadrilateral face
        prevBasisCnt = numVecQuadFaceStdBasis + numVecQuadFaceSingBasis \
            + faceCnt[faceSet[jj]]*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) # Number of functions already added
        curBasisCnt = numVecQuadFaceSingBasis_long # Number of functions to be added
      else: # Triangular face
        prevBasisCnt = numVecFaceStdBasis + faceCnt[faceSet[jj]]*(numVecFaceSingBasis+numVecFaceRotBasis) # Number of functions already added
        curBasisCnt = numVecFaceSingBasis # Number of functions to be added
      basisRange = prevBasisCnt + np.arange(curBasisCnt)

      # Make sure the face isn't on PEC
      if faceInd >= 0:
        globInds[curInd:(curInd+curBasisCnt)] = vecFaceMap[faceInd,basisRange]
      curInd += curBasisCnt

      if eleType == "Wedge" and faceSet[jj] < 2 and singOrder_long > 1:
        # Also add volume-associated functions built from the triangular face
        prevBasisCnt = numVecVolStdBasis + ii*numVecVolTotBasis
        curBasisCnt = numVecVolSingBasis
        basisRange = prevBasisCnt + np.arange(curBasisCnt)

        globInds[curInd:(curInd+curBasisCnt)] = vecVolMap[eleInd+N_tet,basisRange] # Element is always wedge, so always use eleInd + N_tet
        curInd += curBasisCnt

    # Add rotational face funcs
    for jj, faceInd in enumerate(eleFaces[faceSet]): 
      if eleType == "Wedge" and faceSet[jj] >= 2: # Quadrilateral face
        prevBasisCnt = numVecQuadFaceStdBasis + numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long \
                        + faceCnt[faceSet[jj]]*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) # Number of functions already added
        curBasisCnt = numVecQuadFaceRotBasis # Number of functions to be added
      else: # Triangular face
        prevBasisCnt = numVecFaceStdBasis + numVecFaceSingBasis \
                        + faceCnt[faceSet[jj]]*(numVecFaceSingBasis+numVecFaceRotBasis) # Number of functions already added
        curBasisCnt = numVecFaceRotBasis # Number of functions to be added
      basisRange = prevBasisCnt + np.arange(curBasisCnt)

      # Make sure the face isn't on PEC
      if faceInd >= 0:
        globInds[curInd:(curInd+curBasisCnt)] = vecFaceMap[faceInd,basisRange]
      curInd += curBasisCnt

      if eleType == "Wedge" and faceSet[jj] < 2 and singOrder_long > 1:
        # Also add volume-associated functions built from the triangular face
        prevBasisCnt = numVecVolStdBasis + ii*numVecVolTotBasis + numVecVolSingBasis
        curBasisCnt = numVecVolRotBasis
        basisRange = prevBasisCnt + np.arange(curBasisCnt)

        globInds[curInd:(curInd+curBasisCnt)] = vecVolMap[eleInd+N_tet,basisRange] # Element is always wedge, so always use eleInd + N_tet
        curInd += curBasisCnt

    faceCnt[faceSet] += 1 # Update the set of visited faces

    # Add gradient and rotational volume funcs
    if eleType == "Tet":
      prevBasisCnt = numVecVolStdBasis + ii*numVecVolTotBasis
      curBasisCnt = numVecVolTotBasis
      basisRange = prevBasisCnt + np.arange(curBasisCnt)
      globInds[curInd:curInd+curBasisCnt] = vecVolMap[eleInd, basisRange]
    elif eleType == "Wedge":
      # For wedges, the transverse functions are already added, so only add the longitudinal ones
      prevBasisCnt = numVecVolStdBasis + ii*(numVecVolTotBasis) + numVecVolSingBasis+numVecVolRotBasis
      curBasisCnt = numVecVolSingBasis_long+numVecVolRotBasis_long
      basisRange = prevBasisCnt + np.arange(curBasisCnt)
      globInds[curInd:curInd+curBasisCnt] = vecVolMap[eleInd+N_tet, basisRange]
    curInd += curBasisCnt
    
  # Add functions for each singular edge
  faceCnt = np.zeros((numFaces,), dtype=np.int32)
  for ii, edgeInd in enumerate(singEdgeInds):
    curInd = numStdBasis + numSingPointBasis*numNodes + edgeInd*numSingLineBasis # Start index for the current edge
    # Add gradient face funcs
    faceSet = singFaceEdgeNums[edgeInd]
    for jj, faceInd in enumerate(eleFaces[faceSet]): 
      if eleSingEdges[edgeInd] != -1:
        # Make sure the face is connected to a singular line
        if eleType == "Wedge" and faceSet[jj] >= 2: # Quadrilateral face
          prevBasisCnt = numVecQuadFaceStdBasis + 4*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) \
                          + faceCnt[faceSet[jj]]*numVecQuadFaceSingLineBasis # Number of functions already added
          curBasisCnt = numVecQuadFaceSingLineBasis # Number of functions to be added
        else: # Triangular face
          prevBasisCnt = numVecFaceStdBasis + 3*(numVecFaceSingBasis+numVecFaceRotBasis) \
                          + faceCnt[faceSet[jj]]*numVecFaceSingLineBasis # Number of functions already added
          curBasisCnt = numVecFaceSingLineBasis # Number of functions to be added
        basisRange = prevBasisCnt + np.arange(curBasisCnt)
        # basisRange = numVecFaceStdBasis + 3*(numVecFaceSingBasis+numVecFaceRotBasis) \
        #             + faceCnt[faceSet[jj]]*numVecFaceSingLineBasis + np.arange(numVecFaceSingLineBasis)
        faceCnt[faceSet[jj]] += 1

        # Make sure the face isn't on PEC
        if faceInd >= 0:
          globInds[curInd:curInd+curBasisCnt] = vecFaceMap[faceInd,basisRange]
      curInd += curBasisCnt

      if eleType == "Wedge" and faceSet[jj] < 2 and singOrder_long > 1:
        # Also add volume-associated functions built from the triangular face
        prevBasisCnt = numVecVolStdBasis + 6*numVecVolTotBasis + ii*numVecVolSingLineTotBasis
        curBasisCnt = numVecVolSingLineGradBasis
        basisRange = prevBasisCnt + np.arange(curBasisCnt)

        globInds[curInd:(curInd+curBasisCnt)] = vecVolMap[eleInd+N_tet,basisRange] # Element is always wedge, so always use eleInd + N_tet
        curInd += curBasisCnt


    # Add edge-singular volume funcs
    if eleType == "Tet":
      prevBasisCnt = numVecVolStdBasis + numNodes*numVecVolTotBasis + ii*numVecVolSingLineTotBasis
      curBasisCnt = numVecVolSingLineTotBasis
      basisRange = prevBasisCnt + np.arange(curBasisCnt)
      globInds[curInd:curInd+curBasisCnt] = vecVolMap[eleInd, basisRange]
    elif eleType == "Wedge":
      # For wedges, the transverse functions are already added, so only add the longitudinal ones
      prevBasisCnt = numVecVolStdBasis + numNodes*(numVecVolTotBasis) + ii*numVecVolSingLineTotBasis + numVecVolSingLineGradBasis
      curBasisCnt = numVecVolSingLineGradBasis_long + numVecVolSingLineRotBasis + numVecVolSingLineRotBasis_long
      basisRange = prevBasisCnt + np.arange(curBasisCnt)
      globInds[curInd:curInd+curBasisCnt] = vecVolMap[eleInd+N_tet, basisRange]
    curInd += curBasisCnt
  
  return globInds

def sortSingNodes(eleNodes, eleType):
  """
  Function for sorting the singular nodes. Needed to handle cases 
  where the top nodes of a wedge are out of order. This makes sure that
  the order basis functions are added is consistent with the order in neighboring elements 
  """
  if eleType == "Tet":
    nodeSort = np.arange(4)
  elif eleType == "Wedge":
    nodeSort = np.hstack([np.arange(3), np.argsort(eleNodes[3:])+3])
  return nodeSort

@jit
def selectVecInds2D(basisOrder, locInd, vecEdgeMap, vecFaceMap, surfEdges, faceNum=[]):
  """
  A function for determining the correct row or column to be adding a matrix entry for the lumped ports
  given the index for the current element and a local edge basis function

  Inputs:
    basisOrder   - Order of the basis functions. Currently 1 and 2 are supported.
    locInd       - Local index of the current edge basis function (0-3 for order=1, 0-7 for order=2)
    vecEdgeMap   - Provides the matrix indices for each edge number
    vecFaceMap   - Provides the matrix indices for each face number
    surfEdges    - List of edge numbers associated with the current triangle
    faceNum      - Face number of the current triangle (only for order=2)
  Outputs:
    globInd      - Global index to use for the row / column of the matrix. Set to -1 if the basis function shouldn't be used
  """

  if basisOrder == 1: # 1st order edges
    if surfEdges[locInd] <= -1: # If the edge is on PEC, do not use
      globInd = -1
    else:
      globInd = vecEdgeMap[surfEdges[locInd],0]
  elif basisOrder == 2:
    if locInd < 6: # 2nd order edges
      locEdgeNum = int(np.floor(locInd/2)) # Local edge number
      if (surfEdges[locEdgeNum] <= -1): # If the edge is on PEC, do not use
        globInd = -1
      else:
        globInd = vecEdgeMap[surfEdges[locEdgeNum], int(np.mod(locInd,2))] # Set the values so 2 are allowed per edge
    elif locInd >= 6 and locInd < 8: # Faces
      if faceNum == -1: # Ignore faces that lie on PEC
        globInd = -1
      else:
        globInd = vecFaceMap[faceNum, int(np.mod(locInd,2))] # Also have 2 allowed per face

  return int(globInd)

@jit
def selectScalInds3D(basisOrder, locInd, scalNodeMap, scalEdgeMap, eleType, eleNodes, eleEdges, nodeNums, locEdgeNums=[]):
  """
  A function for determining the correct row or column to be adding a matrix entry to
  given the index for the current element and a local node basis function

  Inputs:
    basisOrder   - Order of the basis functions. Currently 1 and 2 are supported.
    locInd      - Local index of the current node basis function (0-3 for order=1, 0-9 for order=2)
    N_n         - Total number of nodes in the mesh (excluding GND PEC, and counting each non-GND PEC once)
    eleType     - The type of element beting considered (currently "Tet" and "Wedge" are supported)
    eleEdges    - List of edge numbers associated with the edges
    nodeNums    - Global node numbers of each of the local nodes 
    locEdgeNums - Local node numbers associated with each local edge number

  Outputs:
    globInd     - Global index to use for the row / column of the matrix. Set to -1 if the basis function shouldn't be used.
  """
  if eleType == "Tet":
    if locInd < 4: # Vertices (1st order)
      globInd = scalNodeMap[eleNodes[locInd]]
    elif locInd >= 4 and locInd < 10: # Edge midpoints (2nd order)
      curEdgeNum = eleEdges[locInd-4] # Edge the function is on the midpoint of
      if curEdgeNum <= -1: # Check if the midpoint is on PEC
        curPECNode = nodeNums[locEdgeNums[locInd-4, 0]]
        if curPECNode == -1: # If the node is on GND PEC, do not use
          globInd = -1
        else:
          globInd = scalNodeMap[eleNodes[locEdgeNums[locInd-4, 0]]]
      else:
        globInd = scalEdgeMap[curEdgeNum] # Numbering uses the midpoint edges
  elif eleType == "Wedge":
    if basisOrder == 1: # Vertices (1st order)
      globInd = scalNodeMap[eleNodes[locInd]]
    elif basisOrder == 2:
      if locInd < 3: # Bottom triangle vertices
        globInd = scalNodeMap[eleNodes[locInd]]
      elif locInd >= 3 and locInd < 6: # Bottom triangle edge midpoints
        curEdgeNum = eleEdges[locInd-3] # Edge the function is on the midpoint of
        if curEdgeNum <= -1: # Check if the midpoint is on PEC
          curPECNode = nodeNums[locEdgeNums[locInd-3, 0]]
          if curPECNode == -1: # If the node is on GND PEC, do not use
            globInd = -1
          else:
            globInd = scalNodeMap[eleNodes[locEdgeNums[locInd-3, 0]]]
        else:
          globInd = scalEdgeMap[curEdgeNum] # Numbering uses the midpoint edges
      elif locInd >= 6 and locInd < 9: # Top triangle vertices
        globInd = scalNodeMap[eleNodes[locInd-3]]
      elif locInd >= 9 and locInd < 12: # Top triangle edge midpoints
        curEdgeNum = eleEdges[locInd-6] # Edge the function is on the midpoint of
        if curEdgeNum <= -1: # Check if the midpoint is on PEC
          curPECNode = nodeNums[locEdgeNums[locInd-6, 0]]
          if curPECNode == -1: # If the node is on GND PEC, do not use
            globInd = -1
          else:
            globInd = scalNodeMap[eleNodes[locEdgeNums[locInd-6, 0]]]
        else:
          globInd = scalEdgeMap[curEdgeNum] # Numbering uses the midpoint edges
      elif locInd >= 12 and locInd < 15: # Side edge midpoints
        curEdgeNum = eleEdges[locInd-6] # Edge the function is on the midpoint of
        if curEdgeNum <= -1: # Check if the midpoint is on PEC
          curPECNode = nodeNums[locEdgeNums[locInd-6, 0]]
          if curPECNode == -1: # If the node is on GND PEC, do not use
            globInd = -1
          else:
            globInd = scalNodeMap[eleNodes[locEdgeNums[locInd-6, 0]]]
        else:
          globInd = scalEdgeMap[curEdgeNum] # Numbering uses the midpoint edges

  return globInd

@jit
def selectScalInds2D(locInd, scalNodeMap, scalEdgeMap, triNodes, triEdges, nodeNums, locEdgeNums):
  """
  A function for determining the correct row or column to be adding a matrix entry for the lumped ports
  given the index for the current element and a local node basis function

  Inputs:
    locInd       - Local index of the current node basis function (0-2 for order=1, 0-5 for order=2)
    N_n          - Total number of nodes in the mesh (excluding GND PEC, and counting each non-GND PEC once)
    triNodes     - Node numbers (not including GND) of each of the triangle nodes
    triEdges     - List of edge numbers associated with the triangle
    nodeNums     - Global node numbers (including GND) of each of the triangle nodes nodes 
    locEdgeNums  - Local node numbers associated with each local edge number
  Outputs:
    globInd      - Global index to use for the row / column of the matrix. Set to -1 if the basis function shouldn't be used
  """

  if locInd < 3: # Vertices (1st order)
    globInd = scalNodeMap[triNodes[locInd]]
  elif locInd >= 3 and locInd < 6: # Edge midpoints (2nd order)
    curEdgeNum = triEdges[locInd-3] # Edge the function is on the midpoint of
    if curEdgeNum <= -1: # Check if the midpoint is on PEC
      curPECNode = nodeNums[locEdgeNums[locInd-3, 0]]
      if curPECNode == -1: # If the node is on GND PEC, do not use
        globInd = -1
      else:
        globInd = scalNodeMap[triNodes[locEdgeNums[locInd-3, 0]]]
    else:
      globInd = scalEdgeMap[curEdgeNum] # Numbering uses the midpoint edges

  return int(globInd)

def assignVoltageFields(nodeList, edgeList, voltList, voltEdgeList, voltFaceList, voltP, basisMaps, basisOrder, quadOrder, numUnknowns, simType):
  """
  Assigns values to the fields (or potentials) based on the voltages on each port
  """

  vecEdgeMap, vecFaceMap, vecVolMap = [basisMaps["vecEdgeMap"], basisMaps["vecFaceMap"], basisMaps["vecVolMap"]]
  scalNodeMap, scalEdgeMap = [basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"]]
  if basisOrder == 1:
    numScalBasis = 3
  elif basisOrder == 2:
    numScalBasis = 6
  numVecUn, numScalUn = getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap)

  edgeNums = np.array([[0,1], [0,2], [1,2]], dtype=np.int32)
  
  knownInds = [] # Indices where the field is known
  knownVals = [] # Values of the unknowns associated with the boundary condition
  for surfNum in range(voltP["num"]):
    startInd = voltP[f"{surfNum}"]["startInd"]
    if surfNum == voltP["num"] - 1:
      endInd = len(voltList)
    else:
      endInd = voltP[f"{surfNum+1}"]["startInd"]
    surfEdges = np.unique(voltEdgeList[startInd:endInd])
    surfEdges = surfEdges[surfEdges > -1] # Ignore PEC entries
    surfNodes = np.unique(voltList[startInd:endInd])

    if simType == "E":
      # Record the value of each interpolation point coordinate in the direction of the port
      if basisOrder == 1:
        # vecIntCoords = (nodeList[edgeList[surfEdges, 0], 1:] @ voltP[f"{surfNum}"]["direction"] \
        #                 + nodeList[edgeList[surfEdges, 1], 1:] @ voltP[f"{surfNum}"]["direction"]) / 2
        vecDir = nodeList[edgeList[surfEdges, 0], 1:] - nodeList[edgeList[surfEdges, 1], 1:]
        vecDir = (vecDir @ voltP[f"{surfNum}"]["direction"]) / np.linalg.norm(vecDir)
        
      # Calculate voltages at each point based on the coordinates
      maxCoord = np.max(nodeList[edgeList[surfEdges], 1:] @ voltP[f"{surfNum}"]["direction"])
      minCoord = np.min(nodeList[edgeList[surfEdges], 1:] @ voltP[f"{surfNum}"]["direction"])
      # vecVoltages = voltP[f"{surfNum}"]["V"]*(vecIntCoords - minCoord)/(maxCoord-minCoord)
      knownVals.append(voltP[f"{surfNum}"]["V"]*vecDir/(maxCoord-minCoord))
      knownInds.append(surfEdges)

    if simType == "A" or simType == "Phi":
      # Record the integral of each gradient function along the line
      gradInts = np.zeros((numScalUn,), dtype=np.float64)
      for triInd in range(len(voltList)):
        triNodes = voltList[triInd]
        triEdges = voltEdgeList[triInd]
        nodeNums = nodeList[triNodes, 0]
        coords = nodeList[triNodes, 1:]

        scalInds = [selectScalInds2D(ii, scalNodeMap, scalEdgeMap, triNodes, triEdges, nodeNums, edgeNums) for ii in range(numScalBasis)]
        loc = q.gauss_quadrature_trisimp_init(quadOrder, *coords)
        if basisOrder == 1:
          gradBasis = np.array([basis2Dscalar_numba(coords, ii, loc)[1] for ii in range(3)])
        elif basisOrder == 2:
          _, gradBasis = basis2DSquad(coords, loc)

        for ii, ind in enumerate(scalInds):
          if ind == -1:
            continue

          if basisOrder == 1:
            if ind == 6076:
              direction = voltP[f"{surfNum}"]["direction"]
            gradInts[ind] += q.gauss_quadrature_trisimp(*coords, np.array(list(repeat(gradBasis[ii] @ voltP[f"{surfNum}"]["direction"], len(loc))))) / voltP[f"{surfNum}"]["width"]
          else:
            gradInts[ind] += q.gauss_quadrature_trisimp(*coords, gradBasis[ii] @ voltP[f"{surfNum}"]["direction"]) / voltP[f"{surfNum}"]["width"]
      
      nzInds = np.where(gradInts != 0)[0]
      totInt = np.sum(gradInts[nzInds])
      knownInds.append(nzInds)
      knownVals.append(gradInts[nzInds]/totInt)

  return np.array(knownVals, dtype=np.float64), np.array(knownInds, dtype=np.int32)

def getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap):
  """
  Determines the total number of unknowns in the problem
  """
  if vecVolMap.size != 0:
    numVecUn = np.max([np.max(vecEdgeMap), np.max(vecFaceMap), np.max(vecVolMap)])+1
  elif vecFaceMap.size != 0:
    numVecUn = np.max([np.max(vecEdgeMap), np.max(vecFaceMap)])+1
  elif vecEdgeMap.size != 0:
    numVecUn = np.max(vecEdgeMap)+1
  else:
    numVecUn = 0

  if scalEdgeMap.size != 0:
    numScalUn = np.max([np.max(scalEdgeMap), np.max(scalNodeMap)])+1
  elif scalNodeMap.size != 0:
    numScalUn = np.max(scalNodeMap)+1
  else:
    numScalUn = 0

  return numVecUn, numScalUn

def splitTet(coords, numRecur):
  """
  Recursively splits a tetrahedron into smaller tetrahedrons

  Inputs:
    coords - coordinates of the vertices of the original tetrahedron
    numRecur - Number of recursive passes to perform

  Outputs:
    newCoords - coordinates of the vertices of new tetrahedra
    tetList   - Node connectivity of the new tetrahedra
  """
  oldTetList = np.array([[0, 1, 2, 3]], dtype=np.int32) # Node numbers for the tet
  newCoords = np.copy(coords)
  curInd = 4
  for ii in range(numRecur): # Recursively split into new tets
    newTetList = np.zeros((4*len(oldTetList),4), dtype=np.int32)
    for jj, oldEleNodes in enumerate(oldTetList): # Split each tet into 4 new tets
      oldEleCenter = np.average(newCoords[oldEleNodes], axis=0).reshape(1,3)
      newCoords = np.append(newCoords, oldEleCenter, axis=0)
      newInd = curInd
      curInd += 1
      newTets = np.array([np.hstack([oldEleNodes[:3], newInd]), \
                          np.hstack([oldEleNodes[:2], newInd, oldEleNodes[3]]), \
                          np.hstack([oldEleNodes[0], newInd, oldEleNodes[2:]]), \
                          np.hstack([newInd, oldEleNodes[1:]])])
      newTetList[4*jj:4*(jj+1)] = np.sort(newTets, axis=1)
    oldTetList = newTetList

  return newCoords, newTetList

def splitTri(coords, numRecur):
  """
  Recursively splits a triangle into smaller triangles

  Inputs:
    coords    - coordinates of the vertices of the original triangle
    numRecur  - Number of recursive passes to perform

  Outputs:
    newCoords - coordinates of the vertices of new triangles
    triList   - Node connectivity of the new triangles
  """
  oldTriList = np.array([[0, 1, 2]], dtype=np.int32) # Node numbers for the tri
  newCoords = np.copy(coords)
  curInd = 3
  for ii in range(numRecur): # Recursively split into new tris
    newTriList = np.zeros((3*len(oldTriList),3), dtype=np.int32)
    for jj, oldEleNodes in enumerate(oldTriList): # Split each tri into 4 new tris
      oldEleCenter = np.average(newCoords[oldEleNodes], axis=0).reshape(1,3)
      newCoords = np.append(newCoords, oldEleCenter, axis=0)
      newInd = curInd
      curInd += 1
      newTris = np.array([np.hstack([oldEleNodes[:2], newInd]), \
                          np.hstack([oldEleNodes[0], newInd, oldEleNodes[2]]), \
                          np.hstack([newInd, oldEleNodes[1:]])])
      newTriList[3*jj:3*(jj+1)] = np.sort(newTris, axis=1)
    oldTriList = newTriList

  return newCoords, newTriList