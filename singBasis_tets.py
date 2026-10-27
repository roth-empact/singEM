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
from numba import jit
from itertools import repeat
from basis_funcs import basis3Dscalar_numba, basis_vector_numba
from HigherOrder3D import basis3DVquad
from scipy.special import factorial

def vecBasisSingular_orderN(coords, coeff, cornerCoeff, points, bOrder, sOrder):
  """
  Forms singular basis functions for an arbitrary order tet

  Inputs:
    coords       - Coordinates of each node of the tet
    singType    - Type of singular tet ("Point", "Line", or "Corner")
    coeff       - Singular coefficient
    cornerCoeff - Singular coefficient for the corner nodes (same as coeff if this isn't a corner)
    points      - Points to evaluate the basis function at
    bOrder      - Order of nonsingular basis functions
    sOrder      - Order of singular functions
  
  """

  ## Start by numbering the different types of functions
  numStdBasis = int(bOrder*(bOrder+2)*(bOrder+3)/2) # Total number of nonsingular functions
  numVecEdgeStdBasis = bOrder
  numVecFaceStdBasis = bOrder*(bOrder-1)
  numVecVolStdBasis = int(bOrder*(bOrder-1)*(bOrder-2)/2)

  numVecEdgeSingBasis = sOrder # Number of singular functions on each edge
  numVecFaceSingBasis = int(sOrder*(sOrder-1)/2) # Number of singular functions on each face
  numVecFaceRotBasis = int((sOrder+1)*sOrder/2) # Number of rotational functions on each face
  numVecVolSingBasis = int((sOrder-2)*(sOrder-1)*sOrder/6) # Singular functions
  numVecVolRotBasis = int((sOrder-1)*sOrder*(sOrder+1)/6) # Rotational singular functions (face-derived)

  numVecFaceSingLineBasis = numVecFaceRotBasis # Number of singular line functions on each face
  numVecVolSingLineGradBasis = numVecVolRotBasis # Gradient edge-singular functions (face-derived)
  numVecVolSingLineRotBasis = sOrder + int((sOrder-1)*sOrder) + int((sOrder-2)*(sOrder-1)*sOrder/6) # Rotational edge-singular functions
  numVecVolSingLineBasis = numVecVolSingLineGradBasis + numVecVolSingLineRotBasis

  edgeNums = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], dtype=np.int32)
  faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)

  numSingPointBasis = 3*(numVecEdgeSingBasis + numVecFaceSingBasis + numVecFaceRotBasis) \
                        + numVecVolSingBasis + 2*numVecVolRotBasis
  numSingEdgeBasis = 2*numVecFaceSingLineBasis + numVecVolSingLineBasis
  numNodes, numEdges = [4,6]
  numSingBasis = numNodes*numSingPointBasis + numEdges*numSingEdgeBasis

  fvals = np.zeros((numStdBasis+numSingBasis, len(points), 3), dtype=np.float64)
  cvals = np.zeros((numStdBasis+numSingBasis, len(points), 3), dtype=np.float64)

  ## Get baricentric coordinates
  xi = [basis3Dscalar_numba(coords, ii, points) for ii in range(4)]
  gXi = np.array([xi[ii][1] for ii in range(4)]) # Gradients of baricentric coordinates
  xi = np.array([xi[ii][0] for ii in range(4)]) # baricentric coordinates

  if bOrder == 1:
    for ii, edge in enumerate(edgeNums): 
      fvals[ii,:,:], cvals[ii,:,:] = basis_vector_numba(coords, edge[0], edge[1], points)
    curNumBasis = 6 # Current number of basis functions
  elif bOrder == 2:
    fvals[:20,:,:], cvals[:20,:,:], _ = basis3DVquad(coords, points)
    curNumBasis = 20 # Current number of basis functions

  ## Get lagrange interpolants
  edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, volLagrange, gradVolLagrange, \
     faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, volLagrange_faceFuncs, gradVolLagrange_faceFuncs \
      = sylvester_lagrange(xi, gXi, points, sOrder, edgeNums, faceNums)

  ## Set up singular functions for the each point
  for singNode in range(4):
    basisRange = curNumBasis + np.arange(numSingPointBasis)
    fvals[basisRange,:,:], cvals[basisRange,:,:] \
      = singFuncs_node_N(coords, singNode, xi, gXi, points, cornerCoeff, edgeNums, faceNums, \
                      edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, \
                      volLagrange, gradVolLagrange, faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, \
                          volLagrange_faceFuncs, gradVolLagrange_faceFuncs, numSingPointBasis, sOrder)
    curNumBasis += numSingPointBasis

  ## Set up singular functions for each edge
  for singEdge in range(6):
    basisRange = curNumBasis + np.arange(numSingEdgeBasis)
    fvals[basisRange,:,:], cvals[basisRange,:,:] \
        = singFuncs_edge_N(coords, singEdge, xi, gXi, points, coeff, edgeNums, faceNums, \
                          edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, \
                          volLagrange, gradVolLagrange, faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, \
                          volLagrange_faceFuncs, gradVolLagrange_faceFuncs, numSingEdgeBasis, sOrder)
    curNumBasis += numSingEdgeBasis


  return fvals, cvals

def singFuncs_node_N(coords, singNode, xi, gXi, points, coeff, edgeNums, faceNums, \
                     edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, \
                     volLagrange, gradVolLagrange, faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, \
                     volLagrange_faceFuncs, gradVolLagrange_faceFuncs, numBasis, bOrder):
  """
  Function for creating nth order singular basis functions about a particular singular node
  
  Inputs:
    coords           - Coordinates of the 4 nodes of the tet
    singNode         - Local node index to be treated as the singular node
    xi               - Barycentric coordinates
    gXi              - Gradient of the barycentric coordinates
    points           - Points to evaluate each function at
    coeff            - Singularity coefficient (nu)
    edgeNums         - Local node indices forming each edge
    faceNums         - Local node indices forming each face
    edgeLagrange     - Interpolating functions on each edge
    gradEdgeLagrange - Gradients of edge interpolating functions
    faceLagrange     - Interpolating functions on each face
    gradFaceLagrange - Gradients of face interpolating functions
    volLagrange      - Interpolating functions in the volume
    gradVolLagrange  - Gradients of volume interpolating functions
    numBasis         - Number of basis functions to add
    bOrder           - Order of the basis functions
  
  Outputs:
    fvals         - Singular basis function values in terms of barycentric coordinates
    cvals        - Curl of the singular basis functions in terms of barycentric coordinates
  """

  # Select the (non-PEC) edges and faces touching the singular node
  if singNode == 0:
    singEdges = np.array([0, 1, 2], dtype=np.int32)
    singFaces = np.array([0, 1, 2], dtype=np.int32)
  elif singNode == 1:
    singEdges = np.array([0, 3, 4], dtype=np.int32)
    singFaces = np.array([0, 1, 3], dtype=np.int32)
  elif singNode == 2:
    singEdges = np.array([1, 3, 5], dtype=np.int32)
    singFaces = np.array([0, 2, 3], dtype=np.int32)
  elif singNode == 3:
    singEdges = np.array([2, 4, 5], dtype=np.int32)
    singFaces = np.array([1, 2, 3], dtype=np.int32)

  numPts = len(points)
  fvals = np.zeros((numBasis, numPts, 3), dtype=np.float64)
  cvals = np.zeros((numBasis, numPts, 3), dtype=np.float64)
  ind = 0

  # Gradient edge functions
  edgePairs = [[singEdges[ii], edgeNums[singEdges[ii]]] for ii in range(len(singEdges))]
  for edgeInd, edge in edgePairs: 
    # Find the nonsingular node in the edge
    for node in edge:
      if node != singNode:
        nonSingNode = node
        break

    # Build the basis function
    scalBaseFunc = (1-(1-xi[singNode])**(coeff-1))*xi[nonSingNode]
    gradBaseFunc = np.outer((coeff-1)*(1-xi[singNode])**(coeff-2)*xi[nonSingNode], gXi[singNode]) \
                           + np.outer(1-(1-xi[singNode])**(coeff-1), gXi[nonSingNode])
    for jj in range(len(edgeLagrange[edgeInd])):
      fvals[ind,:,:] = scalBaseFunc.reshape(numPts,1)*gradEdgeLagrange[edgeInd,jj] \
                            + gradBaseFunc*edgeLagrange[edgeInd,jj].reshape(numPts,1)
      ind += 1

  # Gradient face functions
  facePairs = [[singFaces[ii], faceNums[singFaces[ii]]] for ii in range(len(singFaces))]
  for faceInd, face in facePairs:
    faceEdges = np.array([[face[0], face[1]],
                          [face[0], face[2]],
                          [face[1], face[2]]])

    # Find the first edge that contains the singular node, and build the function using that
    for ii, curEdge in enumerate(faceEdges):
      if singNode in curEdge:
        edge = curEdge
        faceEdgeInd = ii # Index of the edge with respect to the face
        break
    # Find the nonsingular node in the edge
    for node in edge:
      if node != singNode:
        nonSingNode = node
        break

    # Form the basis function
    scalBaseFunc = (1-(1-xi[singNode])**(coeff-1))*xi[nonSingNode]
    gradBaseFunc = np.outer((coeff-1)*(1-xi[singNode])**(coeff-2)*xi[nonSingNode], gXi[singNode]) \
                    + np.outer(1-(1-xi[singNode])**(coeff-1), gXi[nonSingNode])

    # Only use the basis function assocaited with the current edge at each interpolation point on the face
    numFaceLagrange = int(len(faceLagrange[faceInd])/3) # Number of interpolation points on the face
    faceLagrangeInds = np.arange(faceEdgeInd*numFaceLagrange, (faceEdgeInd+1)*numFaceLagrange) 
    for jj in faceLagrangeInds:
      fvals[ind,:,:] = scalBaseFunc.reshape(numPts,1)*gradFaceLagrange[faceInd,jj] \
                          + gradBaseFunc*faceLagrange[faceInd,jj].reshape(numPts,1)
      ind += 1

  ## Rotational face functions
  for faceInd, face in facePairs:
    # Find the nonsingular edge in the face
    locSingNode = (face == singNode)
    edge = face[np.where(~locSingNode)[0]] # The two nonsingular nodes on the face form the edge 
    edgeInd = np.where(np.all(edge == edgeNums, axis=1))[0][0] # Edge index in the tet
    locSingNode = np.where(locSingNode)[0][0]

    faceEdges = np.array([[face[0], face[1]],
                          [face[0], face[2]],
                          [face[1], face[2]]])
    faceEdgeInd = np.where(np.all(edge == faceEdges, axis=1))[0][0] # Edge index on the face

    # Build the basis function
    edgeBasis, curlBasis = basis_vector_numba(coords, edge[0], edge[1], points)
    vecBaseFunc = (1-coeff)*((1-xi[singNode])**(coeff) - 1) .reshape(numPts, 1)* edgeBasis
    curlBaseFunc = (1-coeff)*((1-xi[singNode])**(coeff) - 1).reshape(numPts, 1) * curlBasis \
                            - (coeff*(1-coeff)*(1-xi[singNode])**(coeff-1)).reshape(numPts, 1) * np.cross(gXi[singNode], edgeBasis)

    # Add interpolants along the face built for a face-based function
    for jj in range(len(faceLagrange_faceFuncs[faceInd])):
      fvals[ind,:,:] = vecBaseFunc*faceLagrange_faceFuncs[faceInd][jj].reshape(numPts,1)
      cvals[ind,:,:] = curlBaseFunc*faceLagrange_faceFuncs[faceInd][jj].reshape(numPts,1) \
                          + np.cross(gradFaceLagrange_faceFuncs[faceInd][jj], vecBaseFunc)
      ind += 1

  ## Gradient volume functions
  # All of these functions are dependent, so only add them for a single edge
  edgeInd = singEdges[0]
  edge = edgeNums[edgeInd]
  # Find the nonsingular node in the edge
  for node in edge:
    if node != singNode:
      nonSingNode = node
      break

  # Form the basis function
  scalBaseFunc = (1-(1-xi[singNode])**(coeff-1))*xi[nonSingNode]
  gradBaseFunc = np.outer((coeff-1)*(1-xi[singNode])**(coeff-2)*xi[nonSingNode], gXi[singNode]) \
                    + np.outer(1-(1-xi[singNode])**(coeff-1), gXi[nonSingNode])
  numVolLagrange = int(len(volLagrange)/6) # Number of interpolation points in the volume
  volLagrangeInds = np.arange(edgeInd*numVolLagrange, (edgeInd+1)*numVolLagrange) 
  for jj in volLagrangeInds:
    fvals[ind,:,:] = scalBaseFunc.reshape(numPts,1)*gradVolLagrange[jj] \
                          + gradBaseFunc*volLagrange[jj].reshape(len(points),1)
    ind += 1

  ## Rotational volume functions
  # Record the numbers of the nonsingular edges in each 
  for faceInd, face in facePairs[:2]: # Only use two faces to avoid adding dependent functions
    # Find the nonsingular edge in the face
    locSingNode = (face == singNode)
    edge = face[np.where(~locSingNode)[0]] # The two nonsingular nodes on the face form the edge 
    edgeInd = np.where(np.all(edge == edgeNums, axis=1))[0][0] # Edge index in the tet

    # Build the basis function
    edgeBasis, curlBasis = basis_vector_numba(coords, edge[0], edge[1], points)
    vecBaseFunc = (1-coeff)*((1-xi[singNode])**(coeff) - 1).reshape(len(points), 1)* edgeBasis
    curlBaseFunc = (1-coeff)*((1-xi[singNode])**(coeff) - 1).reshape(len(points), 1) * curlBasis \
                            - (coeff*(1-coeff)*(1-xi[singNode])**(coeff-1)).reshape(len(points), 1) * np.cross(gXi[singNode], edgeBasis)

    # Add interpolants in the volume built for a face-based function
    numVolLagrange_faceFuncs = int(len(volLagrange_faceFuncs)/4) # Number of interpolation points in the volume
    volLagrangeInds = np.arange(faceInd*numVolLagrange_faceFuncs, (faceInd+1)*numVolLagrange_faceFuncs) 
    for jj in volLagrangeInds:
      fvals[ind,:,:] = vecBaseFunc*volLagrange_faceFuncs[jj].reshape(numPts,1)
      cvals[ind,:,:] = curlBaseFunc*volLagrange_faceFuncs[jj].reshape(numPts,1) \
                            + np.cross(gradVolLagrange_faceFuncs[jj], vecBaseFunc)
      ind += 1

  return fvals, cvals

def singFuncs_edge_N(coords, singEdge, xi, gXi, points, coeff, edgeNums, faceNums, \
                     edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, \
                     volLagrange, gradVolLagrange, faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, \
                     volLagrange_faceFuncs, gradVolLagrange_faceFuncs, numBasis, bOrder):
  """
  Function for creating nth order singular basis functions about a particular singular edge
  
  Inputs:
    coords           - Coordinates of the 4 nodes of the tet
    singEdge         - Local edge index to be treated as the singular node
    xi               - Barycentric coordinates
    gXi              - Gradient of the barycentric coordinates
    points              - Points to evaluate each function at
    coeff            - Singularity coefficient (nu)
    edgeNums         - Local node indices forming each edge
    faceNums         - Local node indices forming each face
    edgeLagrange     - Interpolating functions on each edge
    gradEdgeLagrange - Gradients of edge interpolating functions
    faceLagrange     - Interpolating functions on each face
    gradFaceLagrange - Gradients of face interpolating functions
    volLagrange      - Interpolating functions in the volume
    gradVolLagrange  - Gradients of volume interpolating functions
    numBasis         - Number of basis functions to add
    bOrder           - Order of the basis functions
  
  Outputs:
    fvals         - Singular basis function values in terms of barycentric coordinates
    cvals        - Curl of the singular basis functions in terms of barycentric coordinates
  """

  singNodes = edgeNums[singEdge] # Nodes touching the singular edge
  edgeFaces = np.array([[0,1], [0,2], [1,2], [0,3], [1,3], [2,3]], dtype=np.int32) # Faces touching each edge
  singFaces = edgeFaces[singEdge] # Faces touching the singular edge

  numPts = len(points)
  fvals = np.zeros((numBasis, numPts, 3), dtype=np.float64)
  cvals = np.zeros((numBasis, numPts, 3), dtype=np.float64)
  ind = 0

  ## Gradient face functions
  facePairs = [[singFaces[ii], faceNums[singFaces[ii]]] for ii in range(len(singFaces))]
  for faceInd, face in facePairs:
    # Find the index of the singular edge with respect to the face
    faceEdges = np.array([[face[0], face[1]],
                          [face[0], face[2]],
                          [face[1], face[2]]])
    for ii, curEdge in enumerate(faceEdges):
      if np.all(singNodes == curEdge):
        faceEdgeInd = ii # Index of the edge with respect to the face
        faceOppNode = 2 - faceEdgeInd # Index of the opposite node with respect to the face
        break

    # Form the basis function
    scalBaseFunc = (1-(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-1))*xi[face[0]]*xi[face[1]]*xi[face[2]]
    gradBaseFunc = (1-(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-1)).reshape(numPts,1)\
                        *(np.outer(xi[face[0]]*xi[face[1]], gXi[face[2]]) + np.outer(xi[face[0]]*xi[face[2]], gXi[face[1]]) \
                          + np.outer(xi[face[1]]*xi[face[2]], gXi[face[0]])) \
                    + np.outer((coeff-1)*(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-2)*xi[face[0]]*xi[face[1]]*xi[face[2]], \
                               (gXi[singNodes[0]] + gXi[singNodes[1]])) # Face 012

    for jj in range(len(faceLagrange_faceFuncs[faceInd])):
      # if validFaceInds_faceFuncs[jj][faceOppNode] == 0: # Checks that the function is only interpolating from this edge
      fvals[ind,:,:] = scalBaseFunc.reshape(numPts,1)*gradFaceLagrange_faceFuncs[faceInd,jj] \
                        + gradBaseFunc*faceLagrange_faceFuncs[faceInd,jj].reshape(numPts,1)
      ind += 1

  ## Gradient volume funcs
  # All of these functions are dependent, so only add them for a single face
  faceInd = singFaces[0]
  face = faceNums[faceInd]

  # Form the basis function
  scalBaseFunc = (1-(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-1))*xi[face[0]]*xi[face[1]]*xi[face[2]]
  gradBaseFunc = (1-(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-1)).reshape(numPts,1)\
                        *(np.outer(xi[face[0]]*xi[face[1]], gXi[face[2]]) + np.outer(xi[face[0]]*xi[face[2]], gXi[face[1]]) \
                          + np.outer(xi[face[1]]*xi[face[2]], gXi[face[0]])) \
                    + np.outer((coeff-1)*(1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-2)*xi[face[0]]*xi[face[1]]*xi[face[2]], \
                               (gXi[singNodes[0]] + gXi[singNodes[1]])) # Face 012

  ## Add interpolants in the volume built for a face-based function
  numVolLagrange_faceFuncs = int(len(volLagrange_faceFuncs)/4) # Number of interpolation points in the volume
  volLagrangeInds = np.arange(faceInd*numVolLagrange_faceFuncs, (faceInd+1)*numVolLagrange_faceFuncs) 
  for jj in volLagrangeInds: # Only add the set of independent functions
    fvals[ind,:,:] = scalBaseFunc.reshape(numPts,1)*gradVolLagrange_faceFuncs[jj] \
                        + gradBaseFunc*volLagrange_faceFuncs[jj].reshape(numPts,1)
    ind += 1
  
  ## Rotational volume functions
  # Identify the nonsingular edge and nodes
  nonSingEdge = 5 - singEdge  # This edge always has the two nodes not in the singular edge
  nonSingNodes = edgeNums[nonSingEdge]

  # Form the basis function
  edgeBasis, curlBasis = basis_vector_numba(coords, nonSingNodes[0], nonSingNodes[1], points)
  scalFunc = ((1-coeff)*((1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff) - 1) \
                    *xi[singNodes[0]]*xi[singNodes[1]]).reshape(len(points), 1)
  gradFunc = (1-coeff)*(((1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff) - 1).reshape(len(points), 1) \
                          *(xi[singNodes[0]].reshape(len(points), 1)*gXi[singNodes[1]] + xi[singNodes[1]].reshape(len(points), 1)*gXi[singNodes[0]]) \
                        - coeff*((1-xi[singNodes[0]]-xi[singNodes[1]])**(coeff-1)*xi[singNodes[0]]*xi[singNodes[1]]).reshape(len(points), 1) \
                                  *(gXi[singNodes[0]] + gXi[singNodes[1]]))
  vecBaseFunc = scalFunc*edgeBasis
  curlBaseFunc = scalFunc*curlBasis + np.cross(gradFunc, edgeBasis)
  
  # Add interpolants in the volume
  numVolLagrange = int(len(volLagrange)/6) # Number of interpolation points in the volume
  volLagrangeInds = np.arange(nonSingEdge*numVolLagrange, (nonSingEdge+1)*numVolLagrange) 
  for jj in volLagrangeInds:
    fvals[ind,:,:] = vecBaseFunc*volLagrange[jj].reshape(numPts,1)
    cvals[ind,:,:] = curlBaseFunc*volLagrange[jj].reshape(numPts,1) \
                          + np.cross(gradVolLagrange[jj], vecBaseFunc)
    ind += 1

  # Add interpolants along each face touching the edge
  nonSingFaces = edgeFaces[nonSingEdge]
  facePairs = [[nonSingFaces[ii], faceNums[nonSingFaces[ii]]] for ii in range(len(singFaces))]
  for faceInd, face in facePairs:
    faceEdges = np.array([[face[0], face[1]],
                          [face[0], face[2]],
                          [face[1], face[2]]])
    faceEdgeInd = np.where(np.all(nonSingNodes == faceEdges, axis=1))[0][0] # Edge index on the face

    # Add interpolants along the face
    # Only use the basis function associated with the current edge at each interpolation point on the face
    numFaceLagrange = int(len(faceLagrange[faceInd])/3) # Number of interpolation points on the face
    faceLagrangeInds = np.arange(faceEdgeInd*numFaceLagrange, (faceEdgeInd+1)*numFaceLagrange) 
    for jj in faceLagrangeInds:
      fvals[ind,:,:] = vecBaseFunc*faceLagrange[faceInd][jj].reshape(numPts,1)
      cvals[ind,:,:] = curlBaseFunc*faceLagrange[faceInd][jj].reshape(numPts,1) \
                            + np.cross(gradFaceLagrange[faceInd][jj], vecBaseFunc)
      ind += 1


  # Add interpolants along the edge
  for jj in range(len(edgeLagrange[nonSingEdge])):
    fvals[ind,:,:] = vecBaseFunc*edgeLagrange[nonSingEdge][jj].reshape(numPts,1)
    cvals[ind,:,:] = curlBaseFunc*edgeLagrange[nonSingEdge][jj].reshape(numPts,1) \
                          + np.cross(gradEdgeLagrange[nonSingEdge][jj], vecBaseFunc)
    ind += 1

  return fvals, cvals



def sylvester_lagrange(xi, gXi, loc, bOrder, edgeNums, faceNums):
  """
  Constructs Sylvester-Lagrange interpolating polynomials at a specified set of points in a tet 
  for a specified order of basis functions

  Inputs:
    xi       - Barycentric coordinates of the tet
    gXi      - Gradient of the barycentric coordinates
    loc      - Positions to evaluate basis functions at
    bOrder   - Order of basis functions 
    edgeNums - Nodes forming each edge
    faceNums - Nodes forming each face

  Outputs:
    lagrange - Set of interpolating polynomials for the given order for this edge
  """

  numEdgePts = bOrder
  numFacePts = 3*int((bOrder-1)*bOrder/2)
  numVolPts = int((bOrder-2)*(bOrder-1)*bOrder)
  numIntPts = len(loc) # Number of integration points

  edgeLagrange = np.zeros((6, numEdgePts, numIntPts), dtype=np.float64) # Interpolating points along each edge
  gradEdgeLagrange = np.zeros((6, numEdgePts, numIntPts, 3), dtype=np.float64) # Gradients of interpolating points along each edge
  faceLagrange = np.zeros((4, numFacePts, numIntPts), dtype=np.float64) # Interpolating points along faces
  gradFaceLagrange = np.zeros((4, numFacePts, numIntPts, 3), dtype=np.float64) # Gradients of interpolating points along each face
  volLagrange = np.zeros((numVolPts, numIntPts), dtype=np.float64) # Interpolating points in the volume
  gradVolLagrange = np.zeros((numVolPts, numIntPts, 3), dtype=np.float64) # Gradients of interpolating points in the volume

  # Numbering for lagrange points built from face functions
  numFacePts_faceFuncs = bOrder + int((bOrder-1)*bOrder/2)
  numVolPts_faceFuncs = 2*int((bOrder-1)*bOrder) + 4*int((bOrder-2)*(bOrder-1)*bOrder/6)
  faceLagrange_faceFuncs = np.zeros((6, numFacePts_faceFuncs, numIntPts), dtype=np.float64)
  gradFaceLagrange_faceFuncs = np.zeros((6, numFacePts_faceFuncs, numIntPts, 3), dtype=np.float64)
  volLagrange_faceFuncs = np.zeros((numVolPts_faceFuncs, numIntPts), dtype=np.float64) 
  gradVolLagrange_faceFuncs = np.zeros((numVolPts_faceFuncs, numIntPts, 3), dtype=np.float64) 

  # Record nodes not forming each edge
  nonEdgeInds = []
  for ii in range(len(edgeNums)):
    nonEdgeInds.append([])
    for jj in range(4):
      if jj not in edgeNums[ii]:
        nonEdgeInds[ii].append(jj)
    nonEdgeInds[ii] = np.array(nonEdgeInds[ii])

  # Record nodes not forming each face
  nonFaceInds = []
  for ii in range(len(faceNums)):
    for jj in range(4):
      if jj not in faceNums[ii]:
        nonFaceInds.append(jj)

  # Start by forming edge-associated functions
  if bOrder > 1:
    for edgeInd in range(len(edgeNums)):
      edgeNodes = edgeNums[edgeInd]
      validEdgeInds = [[bOrder-jj-1, jj] for jj in range(bOrder)] # Possible values for i and j for the current order
      shiftedFlags = np.zeros((4,), dtype=np.bool_) # Marks which functions are shifted
      shiftedFlags[edgeNodes] = True
      ind = 0
      for ii, jj in validEdgeInds:
        gridInds = np.zeros((4,), dtype=np.int32)
        gridInds[edgeNodes] = [ii, jj]
        edgeLagrange[edgeInd,ind,:], gradEdgeLagrange[edgeInd,ind,:,:] \
          = calcLagrange(gridInds[0], gridInds[1], gridInds[2], gridInds[3], xi, gXi, shiftedFlags, bOrder)
        ind += 1
  else:
    edgeLagrange = np.ones((6, 1, numIntPts), dtype=np.float64) # There's only a single function for each edge in first order

  # Form the face associated functions
  if bOrder > 1:
    validFaceInds = []
    for kk in range(bOrder-1):
      validFaceInds.extend([[bOrder-jj-2-kk, jj, kk+1] for jj in range(bOrder-kk-1)])

    for faceInd, face in enumerate(faceNums):
      ind = 0
      faceEdges = np.array([[face[0], face[1]], [face[0], face[2]], [face[1], face[2]]], dtype=np.int32)
      nonEdgeNodes = np.array([face[2], face[1], face[0]], dtype=np.int32) # Nodes on the face that aren't on each edge
      for edgeInd, edge in enumerate(faceEdges):
        shiftedFlags = np.zeros((4,), dtype=np.bool_) # Marks which functions are shifted
        shiftedFlags[edge] = True
        for ii, jj, kk in validFaceInds:
          gridInds = np.zeros((4,), dtype=np.int32)
          gridInds[edge] = [ii, jj]
          gridInds[nonEdgeNodes[edgeInd]] = kk
          faceLagrange[faceInd,ind,:], gradFaceLagrange[faceInd,ind,:,:] \
            = calcLagrange(gridInds[0], gridInds[1], gridInds[2], gridInds[3], xi, gXi, shiftedFlags, bOrder)
          ind += 1

  # Also add face associated functions for basis functions on the face
  validFaceInds_faceFuncs = []
  for kk in range(bOrder):
    validFaceInds_faceFuncs.extend([[bOrder-jj-1-kk, jj, kk] for jj in range(bOrder-kk)])

  for faceInd, face in enumerate(faceNums):
    shiftedFlags = np.zeros((4,), dtype=np.bool_) # Marks which functions are shifted
    shiftedFlags[face] = True
    ind = 0
    for ii, jj, kk in validFaceInds_faceFuncs:
      gridInds = np.zeros((4,), dtype=np.int32)
      gridInds[face] = [ii, jj, kk]
      faceLagrange_faceFuncs[faceInd,ind,:], gradFaceLagrange_faceFuncs[faceInd,ind,:,:] \
        = calcLagrange(gridInds[0], gridInds[1], gridInds[2], gridInds[3], xi, gXi, shiftedFlags, bOrder)
      ind += 1

  # Form the volume associated functions
  if bOrder > 2:
    # Get possible values for i, j, k, and l for the current order
    validVolInds = []
    for kk in range(bOrder-2):
      for ll in range(bOrder-2-kk):
        validVolInds.extend([[bOrder-jj-3-ll-kk, jj, kk+1, ll+1] for jj in range(bOrder-kk-ll-2)])

    ind = 0
    for edgeInd, edge in enumerate(edgeNums):
      shiftedFlags = np.zeros((4,), dtype=np.bool_) # Marks which functions are shifted
      shiftedFlags[edge] = True

      for ii, jj, kk, ll in validVolInds:
        gridInds = np.zeros((4,), dtype=np.int32)
        gridInds[edge] = [ii, jj]
        gridInds[nonEdgeInds[edgeInd]] = [kk, ll]
        volLagrange[ind,:], gradVolLagrange[ind,:,:] \
          = calcLagrange(gridInds[0], gridInds[1], gridInds[2], gridInds[3], xi, gXi, shiftedFlags, bOrder)
        ind += 1

  # Also add volume associated functions for basis functions defined on the face
  if bOrder > 1:
    validVolInds_faceFuncs = []
    for kk in range(bOrder-1):
      for ll in range(bOrder-1-kk):
        validVolInds_faceFuncs.extend([[bOrder-jj-2-ll-kk, jj, kk, ll+1] for jj in range(bOrder-kk-ll-1)])

    ind = 0
    for faceInd, face in enumerate(faceNums):
      shiftedFlags = np.zeros((4,), dtype=np.bool_) # Marks which functions are shifted
      shiftedFlags[face] = True

      # Add interpolation polynomials for face-associated functions
      for ii, jj, kk, ll in validVolInds_faceFuncs:
        gridInds = np.zeros((4,), dtype=np.int32)
        gridInds[face] = [ii, jj, kk]
        gridInds[nonFaceInds[faceInd]] = ll
        volLagrange_faceFuncs[ind,:], gradVolLagrange_faceFuncs[ind,:,:] \
          = calcLagrange(gridInds[0], gridInds[1], gridInds[2], gridInds[3], xi, gXi, shiftedFlags, bOrder)
        ind += 1

  return edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, volLagrange, gradVolLagrange, \
         faceLagrange_faceFuncs, gradFaceLagrange_faceFuncs, volLagrange_faceFuncs, gradVolLagrange_faceFuncs

def calcLagrange(ii, jj, kk, ll, xi, gXi, shiftedFlags, bOrder):
  """
  Function for calculating the lagrange interpolation polynomials associated with a specific grid point
  at a set of points in a tet

  Inputs:
    ii, jj, kk, ll - Grid indices for the point in the tet
    xi             - Barycentric coordinates at each point
    gXi            - Gradient of each barycentric coordinate
    shiftedFlags   - Marks which of the polynomials are shifted 
    bOrder         - Bases 
  """
  numPts = len(xi[0,:])
  gridShift = bOrder + np.sum(shiftedFlags) - 1

  lagrange = 1
  gridInds = np.array([ii, jj, kk, ll], dtype=np.int32)
  for node, ind in enumerate(gridInds):
    if shiftedFlags[node]:
      for mm in range(ind):
        lagrange *= (gridShift)*xi[node] - (mm+1)
    else:
      for mm in range(ind):
        lagrange *= (gridShift)*xi[node] - mm
  lagrange /= factorial(ii)*factorial(jj)*factorial(kk)*factorial(ll)
  lagrange *= (gridShift)/(gridShift - np.sum(gridInds[~shiftedFlags])) # Normalization factor

  # Form the gradient by product rule summation
  if not np.any(lagrange == 0):
    # Simplified division rules can be used when the interpolation function is nonzero at all points
    gradLagrange = np.zeros((numPts, 3), dtype=np.float64)
    for node, ind in enumerate(gridInds):
      if shiftedFlags[node]:
        for mm in range(ind):
          gradLagrange += np.outer(lagrange*(gridShift) / ((gridShift)*xi[node] - (mm+1)), gXi[node])
      else:
        for mm in range(ind):
          gradLagrange += np.outer(lagrange*(gridShift) / ((gridShift)*xi[node] - mm), gXi[node])
    if np.any(np.isnan(gradLagrange)):
      print(f"Hit NaN")
  else:
    # Otherwise, explicitly take gradients of all the terms
    gradLagrangeTerms = [np.zeros((numPts, 3), dtype=np.float64) for _ in range(4)] # Terms with derivatives with respect to each interpolation function
    for node, ind in enumerate(gridInds):
      if shiftedFlags[node]:
        for mm in range(ind):
          curGradTerm = 1
          for nn in range(mm):
            curGradTerm *= (gridShift)*xi[node] - (nn+1)
          for nn in range(mm+1,ind):
            curGradTerm *= (gridShift)*xi[node] - (nn+1)
          gradLagrangeTerms[node] += np.outer(curGradTerm*(gridShift), gXi[node])
      else:
        for mm in range(ind):
          curGradTerm = 1
          for nn in range(mm):
            curGradTerm *= (gridShift)*xi[node] - nn
          for nn in range(mm+1,ind):
            curGradTerm *= (gridShift)*xi[node] - nn
          gradLagrangeTerms[node] += np.outer(curGradTerm*(gridShift), gXi[node])

    for node, ind in enumerate(gridInds):
      for mm in range(ind):
        for node_1 in range(4):
          if node_1 != node:
            if shiftedFlags[node]:
              gradLagrangeTerms[node_1] *= ((gridShift)*xi[node] - (mm+1)).reshape(numPts,1)
            else:
              gradLagrangeTerms[node_1] *= ((gridShift)*xi[node] - mm).reshape(numPts,1)

    gradLagrange = np.sum(gradLagrangeTerms, axis=0) / (factorial(ii)*factorial(jj)*factorial(kk)*factorial(ll))
    gradLagrange *= (gridShift)/(gridShift - np.sum(gridInds[~shiftedFlags])) # Normalization factor

  return lagrange, gradLagrange

if __name__ == "__main__":
  ## Test on an artificial tet
  tetCoords = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]], dtype=np.float64)
  edgeNums = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], dtype=np.int32)
  faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)

  sOrder = 3
  qOrder = 2*sOrder

  import gauss_quadrature_tetra as q
  # points = q.gauss_quadrature_tetra_numba_loc(tetCoords, qOrder)
  # xi = [basis3Dscalar_numba(tetCoords, ii, points) for ii in range(4)]
  # gXi = np.array([xi[ii][1] for ii in range(4)]) # Gradients of baricentric coordinates
  # xi = np.array([xi[ii][0] for ii in range(4)]) # baricentric coordinates

  # edgeLagrange, gradEdgeLagrange, faceLagrange, gradFaceLagrange, volLagrange, gradVolLagrange = sylvester_lagrange(xi, gXi, points, sOrder, edgeNums, faceNums)

  # # Compare against the symbolic function
  # import singBasis_numeric_general as sb
  # from sympy.vector import CoordSys3D, Del, Vector

  # coords = CoordSys3D('coords')
  # delop = Del()
  # xi = [coords.x, coords.y, coords.z, 1-coords.x-coords.y-coords.z] # Barycentric coordinates
  # gXi = [coords.i, coords.j, coords.k, -coords.i-coords.j-coords.k] # Gradients of barycentric coordinates
  # edgeLagrange_sym, faceLagrange_sym, volLagrange_sym = sb.sylvester_lagrange(xi, sOrder, edgeNums, faceNums)

  # gradEdgeLagrange_sym = np.zeros(gradEdgeLagrange.shape, dtype=np.float64)
  # for ii in range(len(edgeLagrange_sym)):
  #   for jj in range(len(edgeLagrange_sym[ii])):
  #     grad = delop(edgeLagrange_sym[ii][jj]).doit()
  #     gradSub = [grad.subs({coords.x: points[kk,0], coords.y: points[kk,1], coords.z: points[kk,2]}).evalf() for kk in range(len(points))]

  #     for kk in range(len(points)):
  #       for dir in range(3):
  #         if gXi[dir] in gradSub[kk].components:
  #           gradEdgeLagrange_sym[ii,jj,kk,dir] = gradSub[kk].components[gXi[dir]]

  # gradEdgeLagrange_err = (gradEdgeLagrange_sym - gradEdgeLagrange)
  # print(f"Error in the lagrange interpolants on the edges: {np.linalg.norm(gradEdgeLagrange_err)/np.linalg.norm(gradEdgeLagrange_sym)}")

  # gradFaceLagrange_sym = np.zeros(gradFaceLagrange.shape, dtype=np.float64)
  # for ii in range(len(faceLagrange_sym)):
  #   for jj in range(len(faceLagrange_sym[ii])):
  #     grad = delop(faceLagrange_sym[ii][jj]).doit()
  #     gradSub = [grad.subs({coords.x: points[kk,0], coords.y: points[kk,1], coords.z: points[kk,2]}).evalf() for kk in range(len(points))]

  #     for kk in range(len(points)):
  #       for dir in range(3):
  #         if gXi[dir] in gradSub[kk].components:
  #           gradFaceLagrange_sym[ii,jj,kk,dir] = gradSub[kk].components[gXi[dir]]

  # gradFaceLagrange_err = (gradFaceLagrange_sym - gradFaceLagrange)
  # print(f"Error in the lagrange interpolants on the faces: {np.linalg.norm(gradFaceLagrange_err)/np.linalg.norm(gradFaceLagrange_sym)}")

  # gradVolLagrange_sym = np.zeros(gradVolLagrange.shape, dtype=np.float64)
  # for ii in range(len(volLagrange_sym)):
  #   grad = delop(volLagrange_sym[ii]).doit()
  #   gradSub = [grad.subs({coords.x: points[kk,0], coords.y: points[kk,1], coords.z: points[kk,2]}).evalf() for kk in range(len(points))]

  #   for kk in range(len(points)):
  #     for dir in range(3):
  #       if gXi[dir] in gradSub[kk].components:
  #         gradVolLagrange_sym[ii,kk,dir] = gradSub[kk].components[gXi[dir]]

  # gradVolLagrange_err = (gradVolLagrange_sym - gradVolLagrange)
  # print(f"Error in the lagrange interpolants in the volume: {np.linalg.norm(gradVolLagrange_err)/np.linalg.norm(gradVolLagrange_sym)}")

  # ## Test integration over a face of the tet
  # triNodes = np.array([0, 1, 2], dtype=np.int32)
  # triCoords = tetCoords[triNodes]

  # # Add some offset to avoid being directly on the face
  # normDir = np.cross(triCoords[0]-triCoords[1], triCoords[0]-triCoords[2]) 
  # normDir /= np.linalg.norm(normDir)
  # tetCenter = np.sum(tetCoords/4, axis=0)
  # dist = np.linalg.norm(tetCenter - tetCoords[3]) # Distance from the center to the opposite point 
  # if np.linalg.norm(tetCenter+1e-9*normDir - tetCoords[3]) < dist:
  #   triCoords += 1e-9*normDir
  # else:
  #   triCoords -= 1e-9*normDir

  # loc = q.gauss_quadrature_trisimp_init(9, *triCoords)
  
  bOrder, sOrder = [2, 2]
  coeff = 2/3
  loc = q.gauss_quadrature_tetra_numba_loc(tetCoords, 8)
  fvals_point, cvals_point = vecBasisSingular_orderN(tetCoords, coeff, coeff, loc, bOrder, sOrder)
  print("Done")