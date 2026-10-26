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
import itertools

@jit
def basis3Dscalar_numba(nodes, node_eval, points):
  '''
  Evaluates a 3D scalar basis function and its gradient at points inside a tetrahedrom
    Inputs:
    nodes     - coordinates of each node (vertex) of the tetrahedron
    node_eval - the node index to evaluate the basis functions of (0, 1, 2, or 3)
    points    - coordinates of each of the evaluation points

    Outputs:
    fvales    - evaluations of the basis function at each point
    gradVals  - evaluations of the gradient at each point
  '''

  n1 = nodes[0]
  n2 = nodes[1]
  n3 = nodes[2]
  n4 = nodes[3]

  tetra = np.array([[1, n1[0], n1[1], n1[2]],
                    [1, n2[0], n2[1], n2[2]],
                    [1, n3[0], n3[1], n3[2]],
                    [1, n4[0], n4[1], n4[2]]], np.float64)

  v = np.linalg.det(tetra) # calculation of volumn
  gradvals = np.empty(3)

  for i in range(3):
    grad_matrix = np.copy(tetra)
    grad_matrix[node_eval] = [0, 0, 0, 0]
    grad_matrix[node_eval, i+1] = 1
    gradvals[i] = np.linalg.det(grad_matrix) / v

  fvals = np.empty(len(points))

  for i in range(len(points)):
      row = points[i]
      tetra_copy = np.copy(tetra)
      tetra_copy[node_eval] = [1, row[0], row[1], row[2]]
      val = np.linalg.det(tetra_copy) / v
      fvals[i] = val

  return fvals, gradvals

@jit()
def basis_vector_numba(nodes, nEval1, nEval2, points):
  '''
  Evaluates a 3D vector basis function and its curl at points inside a tetrahedron
    Inputs:
    nodes     - coordinates of each node (vertex) of the tetrahedron
    nEval1    - the first node index of the edge (0, 1, 2, or 3)
    nEval1    - the second node index of the edge (0, 1, 2, or 3)
    points    - coordinates of each of the evaluation points

    Outputs:
    vectN     - evaluations of the basis function at each point
    curlN     - evaluations of the gradient at each point
  '''
  grad1 = np.zeros(3)
  grad2 = np.zeros(3)
  scal1 = np.zeros((len(points),), dtype = np.float64)
  scal2 = np.zeros((len(points),), dtype = np.float64)
  if len(nodes) == 3:
    scal1, grad1 = basis2Dscalar_numba(nodes, nEval1, points)
    scal2, grad2 = basis2Dscalar_numba(nodes, nEval2, points)
  elif len(nodes) == 4:
    scal1, grad1 = basis3Dscalar_numba(nodes, nEval1, points)
    scal2, grad2 = basis3Dscalar_numba(nodes, nEval2, points)
  else:
    raise Exception("The number of nodes should be set to ")

  length = np.linalg.norm(nodes[nEval2] - nodes[nEval1])

  vectN = np.zeros((scal1.shape[0], 3))

  for i in range(scal1.shape[0]):
    for j in range(3):
      # vectN[i, j] = (scal1[i] * grad2[j] - scal2[i] * grad1[j])
      vectN[i, j] = (scal2[i] * grad1[j] - scal1[i] * grad2[j])

  # vectN *= length
  curlN = np.zeros((vectN.shape[0], 3))
  for i in range(vectN.shape[0]):
    # curlN[i] = 2 * length * np.cross(grad1, grad2)
    curlN[i] = 2 * np.cross(grad2, grad1)

  return vectN, curlN

@jit
def basis2Dscalar_numba(nodes, node_eval, points):
  '''
  basis2Dscalar_Optimized calculates the values of the linear interpolant and
  the gradient of the linear interpolant for a triangle
    Inputs:
    nodes     - coordinates of each node (vertex) of the triangle
    node_eval - the first node index of the edge (0, 1, 2)
    points    - coordinates of each of the evaluation points

    Outputs:
    N         - evaluations of the basis function at each point
    grad_N    - evaluations of the gradient at each point
  '''
  length = len(points) #number of points
  #Values of the basis function at each point
  N = np.zeros((length,), dtype = np.float64)

  #gradient of the basis function at each point
  grad_N = np.zeros((length, 3), dtype = np.float64)

  #Extract node positions from nodes list
  if np.isclose(nodes[0,2], nodes[1,2], atol=1e-9) and np.isclose(nodes[0,2], nodes[2,2], atol=1e-9): # If the surface is perpendicular to z, use x and y
    x1 = nodes[0, 0]
    x2 = nodes[1, 0]
    x3 = nodes[2, 0]
    y1 = nodes[0, 1]
    y2 = nodes[1, 1]
    y3 = nodes[2, 1]

  #Create area_mat as a 3x3 NumPy array
  area_mat = np.array([ [1.0, 1.0, 1.0], [x1, x2, x3], [y1, y2, y3] ], np.float64)

  #Calculate the signed area
  area = np.linalg.det(area_mat)
  
  Nx_mat = np.copy(area_mat)
  Nx_mat[0, node_eval] = 0.0
  Nx_mat[1, node_eval] = 1.0
  Nx_mat[2, node_eval] = 0.0

  grad_Nx = np.linalg.det(Nx_mat) / area

  Ny_mat = np.copy(area_mat)
  Ny_mat[0, node_eval] = 0.0
  Ny_mat[1, node_eval] = 0.0
  Ny_mat[2, node_eval] = 1.0

  grad_Ny = np.linalg.det(Ny_mat) / area

  for m in range(len(points[:, 1])):
    loc_mat = np.copy(area_mat)
    loc_mat[1, node_eval] = points[m, 0]
    loc_mat[2, node_eval] = points[m, 1]
    N[m] = np.linalg.det(loc_mat) / area
    grad_N[m, 0] = grad_Nx
    grad_N[m, 1] = grad_Ny

  grad_N = grad_N[0] # The gradient is constant for first-order basis functions, so the extra points can be removed
  return N, grad_N

@jit
def basis2DSquad(nodes, points):
    # nodes is a list of tri vertices
    # points is a list of points to evaluate

  fvals = np.zeros((6, len(points)), dtype = np.float64) # Basis function values
  gvals = np.zeros((6, len(points), 3), dtype = np.float64) # Gradient values

  L1vals, L1grad = basis2Dscalar_numba(nodes, 0, points)
  L2vals, L2grad = basis2Dscalar_numba(nodes, 1, points)
  L3vals, L3grad = basis2Dscalar_numba(nodes, 2, points)
    
  for ii in range(len(points)):
    # EVALUATE N1, N2, N3
    fvals[0,ii] = L1vals[ii]*(2*L1vals[ii] - 1)
    fvals[1,ii] = L2vals[ii]*(2*L2vals[ii] - 1)
    fvals[2,ii] = L3vals[ii]*(2*L3vals[ii] - 1)
    gvals[0,ii,:] = L1grad*(4*L1vals[ii] - 1)
    gvals[1,ii,:] = L2grad*(4*L2vals[ii] - 1)
    gvals[2,ii,:] = L3grad*(4*L3vals[ii] - 1)

    # EVALUATE N4 = 4L1L2, N5 = 4L1L3, N6 = 4L2L3
    fvals[3,ii] = 4*L1vals[ii]*L2vals[ii]
    fvals[4,ii] = 4*L1vals[ii]*L3vals[ii]
    fvals[5,ii] = 4*L2vals[ii]*L3vals[ii]
    gvals[3,ii,:] = 4*(L1grad*L2vals[ii] + L1vals[ii]*L2grad)
    gvals[4,ii,:] = 4*(L1grad*L3vals[ii] + L1vals[ii]*L3grad)
    gvals[5,ii,:] = 4*(L2grad*L3vals[ii] + L2vals[ii]*L3grad)

  return fvals, gvals

@jit
def basis2DVquad(nodes, points):
  # nodes is a list of tri vertices
  # points is a list of points to evaluate

  fvals = np.zeros((8, len(points), 3), dtype = np.float64) # Basis function values
  cvals = np.zeros((8, len(points), 3), dtype = np.float64) # Curl values

  W1vals, W1curl = basis_vector_numba(nodes, 0, 1, points) # Wn here are actually ln * Wn
  W2vals, W2curl = basis_vector_numba(nodes, 0, 2, points)
  W3vals, W3curl = basis_vector_numba(nodes, 1, 2, points)
  L1vals, L1grad = basis2Dscalar_numba(nodes, 0, points)
  L2vals, L2grad = basis2Dscalar_numba(nodes, 1, points)
  L3vals, L3grad = basis2Dscalar_numba(nodes, 2, points)
  W1curl = W1curl[0,:] # The curl is constant over the element, so only one point is needed
  W2curl = W2curl[0,:]
  W3curl = W3curl[0,:]

  for ii in range(len(points)):    
    # EVALUATE EDGE BASES
    fvals[0,ii,:] = W1vals[ii] * (3*L1vals[ii] - 1)
    fvals[1,ii,:] = W1vals[ii] * (3*L2vals[ii] - 1)
    fvals[2,ii,:] = W2vals[ii] * (3*L1vals[ii] - 1)
    fvals[3,ii,:] = W2vals[ii] * (3*L3vals[ii] - 1)
    fvals[4,ii,:] = W3vals[ii] * (3*L2vals[ii] - 1)
    fvals[5,ii,:] = W3vals[ii] * (3*L3vals[ii] - 1)

    cvals[0,ii,:] = W1curl * (3*L1vals[ii] - 1) + 3*np.cross(L1grad, W1vals[ii])
    cvals[1,ii,:] = W1curl * (3*L2vals[ii] - 1) + 3*np.cross(L2grad, W1vals[ii])
    cvals[2,ii,:] = W2curl * (3*L1vals[ii] - 1) + 3*np.cross(L1grad, W2vals[ii])
    cvals[3,ii,:] = W2curl * (3*L3vals[ii] - 1) + 3*np.cross(L3grad, W2vals[ii])
    cvals[4,ii,:] = W3curl * (3*L2vals[ii] - 1) + 3*np.cross(L2grad, W3vals[ii])
    cvals[5,ii,:] = W3curl * (3*L3vals[ii] - 1) + 3*np.cross(L3grad, W3vals[ii])

    # EVALUATE INTERIOR BASES: one can be discarded
    fvals[6,ii,:] = 4.5 * W1vals[ii] * L3vals[ii]
    # fvals[6,ii,:] = 4.5 * W2vals[ii] * L2vals[ii]
    fvals[7,ii,:] = 4.5 * W3vals[ii] * L1vals[ii]

    cvals[6,ii,:] = 4.5*(L3vals[ii]*W1curl + np.cross(L3grad, W1vals[ii]))
    # cvals[6,ii,:] = 4.5*(L2vals[ii]*W2curl + np.cross(L2grad, W2vals[ii]))
    cvals[7,ii,:] = 4.5*(L1vals[ii]*W3curl + np.cross(L1grad, W3vals[ii]))

  return fvals, cvals

def basis3DVWedge(nodes, globNodeNums, triPoints, vertPoints, vertDir):
  """
  Creates the vector basis functions for each edge of a "Wedge" (i.e.: triangular prism) element
  Inputs:
    nodes        - Coordinates of the 6 nodes of the wedge
    globNodeNums - Global node numbers of the wedge nodes (for determining edge directions)
    triPoints    - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints   - Integration points along the vertical direction
    vertDir      - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals        - Values of each vector basis function at each integration point
    cvals        - Curl of each vector basis function at each integration point
  """

  fvals = np.zeros((9, len(triPoints)*len(vertPoints), 3)) # Basis functions values at each point
  cvals = np.zeros((9, len(triPoints)*len(vertPoints), 3)) # Curl values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)
  vertGradMag = np.sum(np.abs(vertGrad))

  ## Get vector and scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  W1vals, W1curl = basis_vector_numba(triNodes, 0, 1, triPoints)
  W2vals, W2curl = basis_vector_numba(triNodes, 0, 2, triPoints)
  W3vals, W3curl = basis_vector_numba(triNodes, 1, 2, triPoints)
  L1vals, L1grad = basis2Dscalar_numba(triNodes, 0, triPoints)
  L2vals, L2grad = basis2Dscalar_numba(triNodes, 1, triPoints)
  L3vals, L3grad = basis2Dscalar_numba(triNodes, 2, triPoints)
  W1curl = W1curl[0,:] # The curl is constant over the element, so only one point is needed
  W2curl = W2curl[0,:]
  W3curl = W3curl[0,:]

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the edge elements for the triangle edges
    fvals[0,startInd:endInd,:] = (1-vertCoord[ii])*W1vals # Bottom face
    fvals[1,startInd:endInd,:] = (1-vertCoord[ii])*W2vals
    fvals[2,startInd:endInd,:] = (1-vertCoord[ii])*W3vals
    fvals[3,startInd:endInd,:] = vertCoord[ii]*W1vals # Top face
    fvals[4,startInd:endInd,:] = vertCoord[ii]*W2vals
    fvals[5,startInd:endInd,:] = vertCoord[ii]*W3vals

    ## Form the edge elements for the vertical edges
    fvals[6,startInd:endInd,:] = np.outer(L1vals, vertGrad) #/vertGradMag 
    fvals[7,startInd:endInd,:] = np.outer(L2vals, vertGrad) #/vertGradMag
    fvals[8,startInd:endInd,:] = np.outer(L3vals, vertGrad) #/vertGradMag
    
    ## Calculate the curls
    cvals[0,startInd:endInd,:] = (1-vertCoord[ii])*W1curl - np.cross(vertGrad, W1vals) # Bottom face
    cvals[1,startInd:endInd,:] = (1-vertCoord[ii])*W2curl - np.cross(vertGrad, W2vals)
    cvals[2,startInd:endInd,:] = (1-vertCoord[ii])*W3curl - np.cross(vertGrad, W3vals)
    cvals[3,startInd:endInd,:] = vertCoord[ii]*W1curl + np.cross(vertGrad, W1vals) # Top face
    cvals[4,startInd:endInd,:] = vertCoord[ii]*W2curl + np.cross(vertGrad, W2vals)
    cvals[5,startInd:endInd,:] = vertCoord[ii]*W3curl + np.cross(vertGrad, W3vals)

  cvals[6,:,:] = np.cross(L1grad, vertGrad) # / vertGradMag # Sides
  cvals[7,:,:] = np.cross(L2grad, vertGrad) #/ vertGradMag
  cvals[8,:,:] = np.cross(L3grad, vertGrad) #/ vertGradMag

  ## Adjust the signs of the top face to account for the nodes not being ordered by number
  if globNodeNums[3] > globNodeNums[4]:
    fvals[3,:,:] = -fvals[3,:,:]
    cvals[3,:,:] = -cvals[3,:,:]
  if globNodeNums[3] > globNodeNums[5]:
    fvals[4,:,:] = -fvals[4,:,:]
    cvals[4,:,:] = -cvals[4,:,:]
  if globNodeNums[4] > globNodeNums[5]:
    fvals[5,:,:] = -fvals[5,:,:]
    cvals[5,:,:] = -cvals[5,:,:]

  return fvals, cvals

def basis3DVWedge(nodes, globNodeNums, triPoints, vertPoints, vertDir):
  """
  Creates the vector basis functions for each edge of a "Wedge" (i.e.: triangular prism) element
  Inputs:
    nodes        - Coordinates of the 6 nodes of the wedge
    globNodeNums - Global node numbers of the wedge nodes (for determining edge directions)
    triPoints    - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints   - Integration points along the vertical direction
    vertDir      - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals        - Values of each vector basis function at each integration point
    cvals        - Curl of each vector basis function at each integration point
  """

  fvals = np.zeros((9, len(triPoints)*len(vertPoints), 3)) # Basis functions values at each point
  cvals = np.zeros((9, len(triPoints)*len(vertPoints), 3)) # Curl values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)
  vertGradMag = np.sum(np.abs(vertGrad))

  ## Get vector and scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  W1vals, W1curl = basis_vector_numba(triNodes, 0, 1, triPoints)
  W2vals, W2curl = basis_vector_numba(triNodes, 0, 2, triPoints)
  W3vals, W3curl = basis_vector_numba(triNodes, 1, 2, triPoints)
  L1vals, L1grad = basis2Dscalar_numba(triNodes, 0, triPoints)
  L2vals, L2grad = basis2Dscalar_numba(triNodes, 1, triPoints)
  L3vals, L3grad = basis2Dscalar_numba(triNodes, 2, triPoints)
  W1curl = W1curl[0,:] # The curl is constant over the element, so only one point is needed
  W2curl = W2curl[0,:]
  W3curl = W3curl[0,:]

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the edge elements for the triangle edges
    fvals[0,startInd:endInd,:] = (1-vertCoord[ii])*W1vals # Bottom face
    fvals[1,startInd:endInd,:] = (1-vertCoord[ii])*W2vals
    fvals[2,startInd:endInd,:] = (1-vertCoord[ii])*W3vals
    fvals[3,startInd:endInd,:] = vertCoord[ii]*W1vals # Top face
    fvals[4,startInd:endInd,:] = vertCoord[ii]*W2vals
    fvals[5,startInd:endInd,:] = vertCoord[ii]*W3vals

    ## Form the edge elements for the vertical edges
    fvals[6,startInd:endInd,:] = np.outer(L1vals, vertGrad) #/vertGradMag 
    fvals[7,startInd:endInd,:] = np.outer(L2vals, vertGrad) #/vertGradMag
    fvals[8,startInd:endInd,:] = np.outer(L3vals, vertGrad) #/vertGradMag
    
    ## Calculate the curls
    cvals[0,startInd:endInd,:] = (1-vertCoord[ii])*W1curl - np.cross(vertGrad, W1vals) # Bottom face
    cvals[1,startInd:endInd,:] = (1-vertCoord[ii])*W2curl - np.cross(vertGrad, W2vals)
    cvals[2,startInd:endInd,:] = (1-vertCoord[ii])*W3curl - np.cross(vertGrad, W3vals)
    cvals[3,startInd:endInd,:] = vertCoord[ii]*W1curl + np.cross(vertGrad, W1vals) # Top face
    cvals[4,startInd:endInd,:] = vertCoord[ii]*W2curl + np.cross(vertGrad, W2vals)
    cvals[5,startInd:endInd,:] = vertCoord[ii]*W3curl + np.cross(vertGrad, W3vals)

  cvals[6,:,:] = np.cross(L1grad, vertGrad) # / vertGradMag # Sides
  cvals[7,:,:] = np.cross(L2grad, vertGrad) #/ vertGradMag
  cvals[8,:,:] = np.cross(L3grad, vertGrad) #/ vertGradMag

  ## Adjust the signs of the top face to account for the nodes not being ordered by number
  if globNodeNums[3] > globNodeNums[4]:
    fvals[3,:,:] = -fvals[3,:,:]
    cvals[3,:,:] = -cvals[3,:,:]
  if globNodeNums[3] > globNodeNums[5]:
    fvals[4,:,:] = -fvals[4,:,:]
    cvals[4,:,:] = -cvals[4,:,:]
  if globNodeNums[4] > globNodeNums[5]:
    fvals[5,:,:] = -fvals[5,:,:]
    cvals[5,:,:] = -cvals[5,:,:]

  return fvals, cvals

def basis3DSWedge(nodes, triPoints, vertPoints, vertDir):
  """
  Creates the scalar basis functions for each node of a "Wedge" (i.e.: triangular prism) element
  Inputs:
    nodes      - Coordinates of the 6 nodes of the wedge
    triPoints  - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints - Integration points along the vertical direction
    vertDir    - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals      - Values of each scalar basis function at each integration point
    gvals      - Gradient of each scalar basis function at each integration point
  """

  fvals = np.zeros((6, len(triPoints)*len(vertPoints))) # Basis functions values at each point
  gvals = np.zeros((6, len(triPoints)*len(vertPoints), 3)) # Gradient values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)

  ## Get scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  L1vals, L1grad = basis2Dscalar_numba(triNodes, 0, triPoints)
  L2vals, L2grad = basis2Dscalar_numba(triNodes, 1, triPoints)
  L3vals, L3grad = basis2Dscalar_numba(triNodes, 2, triPoints)

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the basis functions for each node
    fvals[0,startInd:endInd] = (1-vertCoord[ii])*L1vals # Bottom face
    fvals[1,startInd:endInd] = (1-vertCoord[ii])*L2vals
    fvals[2,startInd:endInd] = (1-vertCoord[ii])*L3vals
    fvals[3,startInd:endInd] = vertCoord[ii]*L1vals # Top face
    fvals[4,startInd:endInd] = vertCoord[ii]*L2vals
    fvals[5,startInd:endInd] = vertCoord[ii]*L3vals
    
    ## Calculate the gradients
    gvals[0,startInd:endInd,:] = (1-vertCoord[ii])*L1grad - np.outer(L1vals, vertGrad) # Bottom face
    gvals[1,startInd:endInd,:] = (1-vertCoord[ii])*L2grad - np.outer(L2vals, vertGrad)
    gvals[2,startInd:endInd,:] = (1-vertCoord[ii])*L3grad - np.outer(L3vals, vertGrad)
    gvals[3,startInd:endInd,:] = vertCoord[ii]*L1grad + np.outer(L1vals, vertGrad) # Top face
    gvals[4,startInd:endInd,:] = vertCoord[ii]*L2grad + np.outer(L2vals, vertGrad)
    gvals[5,startInd:endInd,:] = vertCoord[ii]*L3grad + np.outer(L3vals, vertGrad)

  return fvals, gvals

def basis3DVWedgeQuad(nodes, globNodeNums, triPoints, vertPoints, vertDir):
  """
  Creates 2nd order vector basis functions for each edge of a "Wedge" (i.e.: triangular prism) element.
  These are 2nd order only in the transverse direction (and 1st in the longitudinal)
  Inputs:
    nodes        - Coordinates of the 6 nodes of the wedge
    globNodeNums - Global node numbers of the wedge nodes (for determining edge directions)
    triPoints    - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints   - Integration points along the vertical direction
    vertDir      - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals        - Values of each vector basis function at each integration point
    cvals        - Curl of each vector basis function at each integration point
  """

  fvals = np.zeros((22, len(triPoints)*len(vertPoints), 3)) # Basis functions values at each point
  cvals = np.zeros((22, len(triPoints)*len(vertPoints), 3)) # Curl values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)

  ## Get vector and scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  Wvals, Wcurls = basis2DVquad(triNodes, triPoints)
  Lvals, Lgrads = basis2DSquad(triNodes, triPoints)

  ## Set up basis functions for the longitudinal field
  longVals = np.array([np.outer(Lvals[ii], vertGrad) for ii in range(len(Lvals))])
  longCurls = np.cross(Lgrads, vertGrad)
  
  # Adjust the signs and ordering of the top face to account for the nodes not being ordered by number
  topNodeOrder = np.argsort(globNodeNums[3:])
  WvalsTop, WcurlsTop = basis2DVquad(triNodes[topNodeOrder], triPoints)
  if np.any(topNodeOrder != np.array([0, 1, 2])): # Reorder edges
    topEdges = np.array([np.sort([topNodeOrder[0], topNodeOrder[1]]),
                         np.sort([topNodeOrder[0], topNodeOrder[2]]),
                         np.sort([topNodeOrder[1], topNodeOrder[2]])])
    topEdgeOrder = np.argsort(np.sum(topEdges, axis=1))
    topBasisOrder = np.arange(8) # The order for the face basis funcs never changes
    topBasisOrder[:6:2] = 2*topEdgeOrder
    topBasisOrder[1:6:2] = 2*topEdgeOrder + 1
    WvalsTop = WvalsTop[topBasisOrder]
    WcurlsTop = WcurlsTop[topBasisOrder]


  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the edge elements for the triangle edges
    fvals[0,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[0] # Bottom face
    fvals[1,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[1]
    fvals[2,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[2]
    fvals[3,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[3]
    fvals[4,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[4]
    fvals[5,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[5]
    fvals[6,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[6]
    fvals[7,startInd:endInd,:] = (1-vertCoord[ii])*Wvals[7]
    fvals[8,startInd:endInd,:] = vertCoord[ii]*WvalsTop[0] # Top face
    fvals[9,startInd:endInd,:] = vertCoord[ii]*WvalsTop[1]
    fvals[10,startInd:endInd,:] = vertCoord[ii]*WvalsTop[2]
    fvals[11,startInd:endInd,:] = vertCoord[ii]*WvalsTop[3]
    fvals[12,startInd:endInd,:] = vertCoord[ii]*WvalsTop[4]
    fvals[13,startInd:endInd,:] = vertCoord[ii]*WvalsTop[5]
    fvals[14,startInd:endInd,:] = vertCoord[ii]*WvalsTop[6]
    fvals[15,startInd:endInd,:] = vertCoord[ii]*WvalsTop[7]

    ## Form the edge elements for the vertical edges
    fvals[16:,startInd:endInd,:] = longVals
    
    ## Calculate the curls
    cvals[0,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[0] - np.cross(vertGrad, Wvals[0]) # Bottom face
    cvals[1,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[1] - np.cross(vertGrad, Wvals[1])
    cvals[2,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[2] - np.cross(vertGrad, Wvals[2])
    cvals[3,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[3] - np.cross(vertGrad, Wvals[3])
    cvals[4,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[4] - np.cross(vertGrad, Wvals[4])
    cvals[5,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[5] - np.cross(vertGrad, Wvals[5])
    cvals[6,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[6] - np.cross(vertGrad, Wvals[6])
    cvals[7,startInd:endInd,:] = (1-vertCoord[ii])*Wcurls[7] - np.cross(vertGrad, Wvals[7])
    cvals[8,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[0] + np.cross(vertGrad, WvalsTop[0]) # Top face
    cvals[9,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[1] + np.cross(vertGrad, WvalsTop[1])
    cvals[10,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[2] + np.cross(vertGrad, WvalsTop[2])
    cvals[11,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[3] + np.cross(vertGrad, WvalsTop[3])
    cvals[12,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[4] + np.cross(vertGrad, WvalsTop[4])
    cvals[13,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[5] + np.cross(vertGrad, WvalsTop[5])
    cvals[14,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[6] + np.cross(vertGrad, WvalsTop[6])
    cvals[15,startInd:endInd,:] = vertCoord[ii]*WcurlsTop[7] + np.cross(vertGrad, WvalsTop[7])
    cvals[16:,startInd:endInd,:] = longCurls # Sides

  return fvals, cvals

def basis3DVWedge_2x2(nodes, globNodeNums, triPoints, vertPoints, vertDir):
  """
  Creates 2nd order vector basis functions for each edge of a "Wedge" (i.e.: triangular prism) element
  This version is 2nd order in both the longitudinal and transverse direction.
  Inputs:
    nodes        - Coordinates of the 6 nodes of the wedge
    globNodeNums - Global node numbers of the wedge nodes (for determining edge directions)
    triPoints    - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints   - Integration points along the vertical direction
    vertDir      - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals        - Values of each vector basis function at each integration point
    cvals        - Curl of each vector basis function at each integration point
  """
  numTranFuncs = 3*8 # Number of transverse functions
  numLongFuncs = 6*2 # Number of longitudinal functions
  numTotFuncs = numTranFuncs + numLongFuncs

  fvals = np.zeros((numTotFuncs, len(triPoints)*len(vertPoints), 3)) # Basis functions values at each point
  cvals = np.zeros((numTotFuncs, len(triPoints)*len(vertPoints), 3)) # Curl values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)

  ## Get vector and scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  Wvals, Wcurls = basis2DVquad(triNodes, triPoints)
  Lvals, Lgrads = basis2DSquad(triNodes, triPoints)

  ## Set up basis functions for the longitudinal field
  longVals = np.array([np.outer(Lvals[ii], vertGrad) for ii in range(len(Lvals))])
  longCurls = np.cross(Lgrads, vertGrad)
  
  # Adjust the signs and ordering of the top face to account for the nodes not being ordered by number
  topNodeOrder = np.argsort(globNodeNums[3:])
  WvalsTop, WcurlsTop = basis2DVquad(triNodes[topNodeOrder], triPoints)
  if np.any(topNodeOrder != np.array([0, 1, 2])): # Reorder edges
    topEdges = np.array([np.sort([topNodeOrder[0], topNodeOrder[1]]),
                         np.sort([topNodeOrder[0], topNodeOrder[2]]),
                         np.sort([topNodeOrder[1], topNodeOrder[2]])])
    topEdgeOrder = np.argsort(np.sum(topEdges, axis=1))
    topBasisOrder = np.arange(8) # The order for the face basis funcs never changes
    topBasisOrder[:6:2] = 2*topEdgeOrder
    topBasisOrder[1:6:2] = 2*topEdgeOrder + 1
    WvalsTop = WvalsTop[topBasisOrder]
    WcurlsTop = WcurlsTop[topBasisOrder]

  # Calculate vertical scaling for each type of function
  tranVertScale = np.array([(1-vertCoord)*(vertCoord - 1/2), (1-vertCoord)*vertCoord, (vertCoord - 1/2)*vertCoord]) # Transverse functions
  longVertScale = np.array([vertCoord-2/3, vertCoord-1/3]) # Vertical functions

  # Also get the corresponding gradient terms
  tranVertGrad = np.multiply.outer(np.array([-2*vertCoord + 3/2, 1-2*vertCoord, 2*vertCoord-1/2]), vertGrad) # Transverse functions
  longVertGrad = np.multiply.outer(np.ones(longVertScale.shape), vertGrad) # Vertical functions

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)
    vertRange = np.arange(startInd, endInd)

    ## Form transverse functions at each level
    for jj in range(len(Wvals)):
      fvals[jj, vertRange, :] = tranVertScale[0][ii]*Wvals[jj] # Bottom face
      fvals[jj + len(Wvals), vertRange, :] = tranVertScale[-1][ii]*WvalsTop[jj] # Top face
      fvals[jj + 2*len(Wvals), vertRange, :] = tranVertScale[1][ii]*Wvals[jj] # Center

      cvals[jj, vertRange, :] = tranVertScale[0][ii]*Wcurls[jj] + np.cross(tranVertGrad[0][ii], Wvals[jj]) # Bottom face
      cvals[jj + len(Wvals), vertRange, :] = tranVertScale[-1][ii]*WcurlsTop[jj] + np.cross(tranVertGrad[-1][ii], WvalsTop[jj]) # Top face
      cvals[jj + 2*len(Wvals), vertRange, :] = tranVertScale[1][ii]*Wcurls[jj] + np.cross(tranVertGrad[1][ii], Wvals[jj]) # Center


    ## Form the longitudinal functions
    for jj in range(len(longVals)):
      # basisRange = numTranFuncs + np.arange(2*jj, 2*(jj+1)) # Basis functions to add at the current transverse position
      fvals[numTranFuncs + 2*jj, vertRange, :] = longVertScale[0][ii]*longVals[jj]
      fvals[numTranFuncs + 2*jj + 1, vertRange, :] = longVertScale[1][ii]*longVals[jj]

      cvals[numTranFuncs + 2*jj, vertRange, :] = longVertScale[0][ii]*longCurls[jj] + np.cross(longVertGrad[0][ii], longVals[jj])
      cvals[numTranFuncs + 2*jj + 1, vertRange, :] = longVertScale[1][ii]*longCurls[jj] + np.cross(longVertGrad[1][ii], longVals[jj])

  return fvals, cvals

def basis3DSWedgeQuad(nodes, globNodeNums, triPoints, vertPoints, vertDir):
  """
  Creates the 2nd order scalar basis functions for each node of a "Wedge" (i.e.: triangular prism) element
  Inputs:
    nodes        - Coordinates of the 6 nodes of the wedge
    globNodeNums - Global node numbers of the wedge nodes (for determining edge directions)
    triPoints    - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints   - Integration points along the vertical direction
    vertDir      - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals        - Values of each scalar basis function at each integration point
    gvals        - Gradient of each scalar basis function at each integration point
  """

  fvals = np.zeros((12, len(triPoints)*len(vertPoints))) # Basis functions values at each point
  gvals = np.zeros((12, len(triPoints)*len(vertPoints), 3)) # Gradient values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)

  ## Get scalar basis functions for one of the triangles (the second will be the same)
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  Lvals, Lgrads = basis2DSquad(triNodes, triPoints)

  # Adjust the signs and ordering of the top face to account for the nodes not being ordered by number
  topNodeOrder = np.argsort(globNodeNums[3:])
  LvalsTop, LgradsTop = basis2DSquad(triNodes[topNodeOrder], triPoints)
  if np.any(topNodeOrder != np.array([0, 1, 2])): # Reorder edges
    topEdges = np.array([np.sort([topNodeOrder[0], topNodeOrder[1]]),
                         np.sort([topNodeOrder[0], topNodeOrder[2]]),
                         np.sort([topNodeOrder[1], topNodeOrder[2]])])
    topEdgeOrder = np.argsort(np.sum(topEdges, axis=1))
    topBasisOrder = np.arange(6)
    topBasisOrder[:3] = topNodeOrder
    topBasisOrder[3:] = topEdgeOrder + 3
    LvalsTop = LvalsTop[topBasisOrder]
    LgradsTop = LgradsTop[topBasisOrder]

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the basis functions for each node
    fvals[0,startInd:endInd] = (1-vertCoord[ii])*Lvals[0] # Bottom face
    fvals[1,startInd:endInd] = (1-vertCoord[ii])*Lvals[1] 
    fvals[2,startInd:endInd] = (1-vertCoord[ii])*Lvals[2] 
    fvals[3,startInd:endInd] = (1-vertCoord[ii])*Lvals[3] 
    fvals[4,startInd:endInd] = (1-vertCoord[ii])*Lvals[4] 
    fvals[5,startInd:endInd] = (1-vertCoord[ii])*Lvals[5] 
    fvals[6,startInd:endInd] = vertCoord[ii]*LvalsTop[0] # Top face
    fvals[7,startInd:endInd] = vertCoord[ii]*LvalsTop[1] 
    fvals[8,startInd:endInd] = vertCoord[ii]*LvalsTop[2] 
    fvals[9,startInd:endInd] = vertCoord[ii]*LvalsTop[3] 
    fvals[10,startInd:endInd] = vertCoord[ii]*LvalsTop[4] 
    fvals[11,startInd:endInd] = vertCoord[ii]*LvalsTop[5] 
    
    ## Calculate the gradients
    gvals[0,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[0] - np.outer(Lvals[0], vertGrad) # Bottom face
    gvals[1,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[1] - np.outer(Lvals[1], vertGrad)
    gvals[2,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[2] - np.outer(Lvals[2], vertGrad)
    gvals[3,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[3] - np.outer(Lvals[3], vertGrad) 
    gvals[4,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[4] - np.outer(Lvals[4], vertGrad)
    gvals[5,startInd:endInd,:] = (1-vertCoord[ii])*Lgrads[5] - np.outer(Lvals[5], vertGrad)
    gvals[6,startInd:endInd,:] = vertCoord[ii]*LgradsTop[0] + np.outer(LvalsTop[0], vertGrad) # Top face
    gvals[7,startInd:endInd,:] = vertCoord[ii]*LgradsTop[1] + np.outer(LvalsTop[1], vertGrad)
    gvals[8,startInd:endInd,:] = vertCoord[ii]*LgradsTop[2] + np.outer(LvalsTop[2], vertGrad)
    gvals[9,startInd:endInd,:] = vertCoord[ii]*LgradsTop[3] + np.outer(LvalsTop[3], vertGrad) 
    gvals[10,startInd:endInd,:] = vertCoord[ii]*LgradsTop[4] + np.outer(LvalsTop[4], vertGrad)
    gvals[11,startInd:endInd,:] = vertCoord[ii]*LgradsTop[5] + np.outer(LvalsTop[5], vertGrad)

  return fvals, gvals

def basis3DSWedgeQuad_new(nodes, globNodeNums, edgeNums, triPoints, vertPoints, vertDir):
  """
  Creates the 2nd order scalar basis functions for each node of a "Wedge" (i.e.: triangular prism) element
  Inputs:
    nodes      - Coordinates of the 6 nodes of the wedge
    triPoints  - Integration points along the surface of the triangle (for the triangular portion)
    vertPoints - Integration points along the vertical direction
    vertDir    - Unit vector of the vertical direction (either '0', '1', or '2' for x, y, or z)
  Outputs:
    fvals      - Values of each scalar basis function at each integration point
    gvals      - Gradient of each scalar basis function at each integration point
  """

  fvals = np.zeros((15, len(triPoints)*len(vertPoints))) # Basis functions values at each point
  gvals = np.zeros((15, len(triPoints)*len(vertPoints), 3)) # Gradient values at each point
  
  ## Form a normalized vertical coordinate
  minH = nodes[0,vertDir]
  maxH = nodes[3,vertDir]
  vertCoord = (vertPoints - minH)/(maxH-minH)
  vertGrad = np.zeros((3,), dtype=np.float64)
  vertGrad[vertDir] = 1/(maxH-minH)
  botVertCoord = -(1-vertCoord)*(2*vertCoord-1)
  topVertCoord = vertCoord*(2*vertCoord-1)
  centerVertCoord = 1 - (2*vertCoord - 1)**2
  botVertGrad = np.outer((4*vertCoord-3), vertGrad)
  topVertGrad = np.outer((4*vertCoord-1), vertGrad)
  centerVertGrad = -4*np.outer((2*vertCoord-1), vertGrad)

  # altVertCoord = 2*vertCoord - 1
  # altBotVertCoord = -(1 - altVertCoord)*altVertCoord/2
  # altTopVertCoord = (1 + altVertCoord)*altVertCoord/2
  # altCenterVertCoord = 1 - altVertCoord**2
  # altVertGrad = 2*vertGrad
  # altBotVertGrad = np.outer(altVertCoord-0.5, altVertGrad)
  # altTopVertGrad = np.outer(altVertCoord+0.5, altVertGrad)
  # altCenterVertGrad = np.outer(-2*altVertCoord, altVertGrad)

  ## Get scalar basis functions for the triangles
  triNodes = nodes[:3,:] # Form a set of 2D node coordinates along the planar surface
  Lvals, Lgrads = basis2DSquad(triNodes, triPoints)
  # topNodeOrder = np.argsort(globNodeNums[3:])
  # LvalsTop, LgradsTop = basis2DSquad(triNodes[topNodeOrder], triPoints)
  L1vals, L1grad = basis2Dscalar_numba(triNodes, 0, triPoints)
  L2vals, L2grad = basis2Dscalar_numba(triNodes, 1, triPoints)
  L3vals, L3grad = basis2Dscalar_numba(triNodes, 2, triPoints)

  ## Evaluate at all triangular points for each vertical point
  for ii in range(len(vertPoints)): 
    startInd = ii*len(triPoints)
    endInd = (ii+1)*len(triPoints)

    ## Form the basis functions for each node
    fvals[0,startInd:endInd] = botVertCoord[ii]*Lvals[0] # Bottom face
    fvals[1,startInd:endInd] = botVertCoord[ii]*Lvals[1] 
    fvals[2,startInd:endInd] = botVertCoord[ii]*Lvals[2] 
    fvals[3,startInd:endInd] = botVertCoord[ii]*Lvals[3] 
    fvals[4,startInd:endInd] = botVertCoord[ii]*Lvals[4] 
    fvals[5,startInd:endInd] = botVertCoord[ii]*Lvals[5] 
    fvals[6,startInd:endInd] = topVertCoord[ii]*Lvals[0] # Top face
    fvals[7,startInd:endInd] = topVertCoord[ii]*Lvals[1] 
    fvals[8,startInd:endInd] = topVertCoord[ii]*Lvals[2] 
    fvals[9,startInd:endInd] = topVertCoord[ii]*Lvals[3] 
    fvals[10,startInd:endInd] = topVertCoord[ii]*Lvals[4] 
    fvals[11,startInd:endInd] = topVertCoord[ii]*Lvals[5] 
    fvals[12,startInd:endInd] = centerVertCoord[ii]*L1vals # Side edges
    fvals[13,startInd:endInd] = centerVertCoord[ii]*L2vals
    fvals[14,startInd:endInd] = centerVertCoord[ii]*L3vals
    
    ## Calculate the gradients
    gvals[0,startInd:endInd,:] = botVertCoord[ii]*Lgrads[0] + np.outer(Lvals[0], botVertGrad[ii]) # Bottom face
    gvals[1,startInd:endInd,:] = botVertCoord[ii]*Lgrads[1] + np.outer(Lvals[1], botVertGrad[ii])
    gvals[2,startInd:endInd,:] = botVertCoord[ii]*Lgrads[2] + np.outer(Lvals[2], botVertGrad[ii])
    gvals[3,startInd:endInd,:] = botVertCoord[ii]*Lgrads[3] + np.outer(Lvals[3], botVertGrad[ii]) 
    gvals[4,startInd:endInd,:] = botVertCoord[ii]*Lgrads[4] + np.outer(Lvals[4], botVertGrad[ii])
    gvals[5,startInd:endInd,:] = botVertCoord[ii]*Lgrads[5] + np.outer(Lvals[5], botVertGrad[ii])
    gvals[6,startInd:endInd,:] = topVertCoord[ii]*Lgrads[0] + np.outer(Lvals[0], topVertGrad[ii]) # Top face
    gvals[7,startInd:endInd,:] = topVertCoord[ii]*Lgrads[1] + np.outer(Lvals[1], topVertGrad[ii])
    gvals[8,startInd:endInd,:] = topVertCoord[ii]*Lgrads[2] + np.outer(Lvals[2], topVertGrad[ii])
    gvals[9,startInd:endInd,:] = topVertCoord[ii]*Lgrads[3] + np.outer(Lvals[3], topVertGrad[ii]) 
    gvals[10,startInd:endInd,:] = topVertCoord[ii]*Lgrads[4] + np.outer(Lvals[4], topVertGrad[ii])
    gvals[11,startInd:endInd,:] = topVertCoord[ii]*Lgrads[5] + np.outer(Lvals[5], topVertGrad[ii])
    gvals[12,startInd:endInd,:] = centerVertCoord[ii]*L1grad + np.outer(L1vals, centerVertGrad[ii]) # Side edges
    gvals[13,startInd:endInd,:] = centerVertCoord[ii]*L2grad + np.outer(L2vals, centerVertGrad[ii])
    gvals[14,startInd:endInd,:] = centerVertCoord[ii]*L3grad + np.outer(L3vals, centerVertGrad[ii])

  # ## Adjust the signs of the top face to account for the nodes not being ordered by number
  # 
  # if np.any(topNodeOrder != np.array([0, 1, 2])):
  #   fvals[6:9,:] = fvals[topNodeOrder+6,:]
  #   gvals[6:9,:,:] = gvals[topNodeOrder+6,:,:]
  # topEdgeOrder = np.argsort(edgeNums[3:6])
  # if np.any(topEdgeOrder != np.array([0, 1, 2])): # Reorder edges if they aren't ordered properly
  #   fvals[9:12,:] = fvals[topEdgeOrder+9,:]
  #   gvals[9:12,:,:] = gvals[topEdgeOrder+9,:,:]
  # if globNodeNums[3] > globNodeNums[4]:
  #   temp = fvals[9,:]
  #   fvals[9,:] = fvals[10,:]
  #   fvals[10,:] = temp

  #   temp = gvals[9,:,:]
  #   gvals[9,:,:] = gvals[10,:,:]
  #   gvals[10,:,:] = temp
  # if globNodeNums[3] > globNodeNums[5]:
  #   temp = fvals[9,:]
  #   fvals[9,:] = fvals[11,:]
  #   fvals[11,:] = temp

  #   temp = gvals[9,:,:]
  #   gvals[9,:,:] = gvals[11,:,:]
  #   gvals[11,:,:] = temp
  # if globNodeNums[4] > globNodeNums[5]:
  #   temp = fvals[10,:]
  #   fvals[10,:] = fvals[11,:]
  #   fvals[11,:] = temp

  #   temp = gvals[10,:,:]
  #   gvals[10,:,:] = gvals[11,:,:]
  #   gvals[11,:,:] = temp

  return fvals, gvals

# nEval is 0 1 2 3
def hierarchical_basis3D_order1_helper(scalar1, scalar2, grad1, grad2):

  full_func = np.zeros((scalar1.shape[0], grad1.shape[0]))
  mix_func = np.zeros((scalar1.shape[0], grad1.shape[0]))
  curl_mix = np.zeros((scalar1.shape[0], grad1.shape[0]))
  curl_full = np.zeros((scalar1.shape[0], grad1.shape[0]))
  # implement mix and full first-order equation from P.303
  for i in range(scalar1.shape[0]):
    for j in range(grad1.shape[0]):
      mix_func[i, j] = (scalar1[i] * grad2[j] - scalar2[i] * grad1[j])
      full_func[i, j] = (scalar1[i] * grad2[j] + scalar2[i] * grad1[j])
      
  for i in range(scalar1.shape[0]):
    curl_mix[i] = 2*np.cross(grad1, grad2)
    curl_full[i] = [0,0,0]

  all_combined = np.stack((mix_func, full_func), axis=0)
  return full_func, mix_func, all_combined, curl_mix, curl_full

def hierarchical_basis3D_order1_all(nodes, points):
  scalar1, grad1 = basis3Dscalar_numba(nodes, 0, points)
  scalar2, grad2 = basis3Dscalar_numba(nodes, 1, points)
  scalar3, grad3 = basis3Dscalar_numba(nodes, 2, points)
  scalar4, grad4 = basis3Dscalar_numba(nodes, 3, points)

  scalar_list = [scalar1, scalar2, scalar3, scalar4]
  grad_list = [grad1, grad2, grad3, grad4]
  
  result = []
  curl_list = []

  for i, j in itertools.combinations(range(4), 2):
    _, _, combined, curl_mix, curl_full = hierarchical_basis3D_order1_helper(scalar_list[i], scalar_list[j], grad_list[i], grad_list[j])
    result.append(combined)
    curl_list.append(curl_mix)
    curl_list.append(curl_full)

  combined_func = np.concatenate(result, axis=0)
  combined_curl = np.stack(curl_list, axis=0)

  return combined_func, combined_curl

# use this
def hierarchical_basis3D_order1(nodes, points):
  """
  Description: This is a function that returns the first order mixed hierarchical functions and curls at integral points
in a tetrahedral.

  Inputs:
    nodes - coordinates of each node (vertex) of the tetrahedron
    points - coordinates of each of the evaluation points

  Outputs:
    combined_func_all - the first order mixed hierarchical functions and curls at integral points
in a tetrahedral.
  """
  scalar1, grad1 = basis3Dscalar_numba(nodes, 0, points)
  scalar2, grad2 = basis3Dscalar_numba(nodes, 1, points)
  scalar3, grad3 = basis3Dscalar_numba(nodes, 2, points)
  scalar4, grad4 = basis3Dscalar_numba(nodes, 3, points)

  scalar_list = [scalar1, scalar2, scalar3, scalar4]
  grad_list = [grad1, grad2, grad3, grad4]
  result = []

  curl_mix_list = []

  for i, j in itertools.combinations(range(4), 2):
    _, mixed1, _, mixed_curl, _ = hierarchical_basis3D_order1_helper(scalar_list[i], scalar_list[j], grad_list[i], grad_list[j])
    result.append(mixed1)
    curl_mix_list.append(mixed_curl)

  combined_func = np.stack(result, axis=0)
  combined_curl = np.stack(curl_mix_list, axis=0)

  return combined_func, combined_curl

def hierarchical_basis3D_order2_edge_helper(scalar1, scalar2, grad1, grad2):
  full_func_edge = np.zeros((scalar1.shape[0], 3))
  curl_edge = np.zeros((scalar1.shape[0], 3))
  for i in range(scalar1.shape[0]):
    curl_edge[i] = [0,0,0]
    for j in range(grad1.shape[0]):
      full_func_edge[i,j] = scalar1[i]*scalar2[i] * (grad1[j] - grad2[j]) + (scalar1[i] - scalar2[i]) * (scalar1[i]*grad2[j] + scalar2[i]*grad1[j])
  
  return full_func_edge, curl_edge

def hierarchical_basis3D_order2_face_helper(scalar1, scalar2, scalar3, grad1, grad2, grad3):

  full_func_face = np.zeros((scalar1.shape[0], 3))
  full_curl_face = np.zeros((scalar1.shape[0], 3))
  mix_func_face1 = np.zeros((scalar1.shape[0], 3))
  curl_func_face1 = np.zeros((scalar1.shape[0], 3))
  mix_func_face2 = np.zeros((scalar1.shape[0], 3))
  curl_func_face2 = np.zeros((scalar1.shape[0], 3))

  # calculate full functions
  for i in range(scalar1.shape[0]):
    for j in range(3):
      full_func_face[i,j] = grad1[j]*scalar2[i]*scalar3[i] + scalar1[i]*grad2[j]*scalar3[i] + scalar1[i]*scalar2[i]*grad3[j]
      
  # calculate mix functions
  _, mix_func_ki, _, mix_curl_ki, _ = hierarchical_basis3D_order1_helper(scalar3, scalar1, grad3, grad1)
  _, mix_func_kj, _, mix_curl_kj, _ = hierarchical_basis3D_order1_helper(scalar3, scalar2, grad3, grad2)
  _, mix_func_ji, _, mix_curl_ji, _ = hierarchical_basis3D_order1_helper(scalar2, scalar1, grad2, grad1)

  mix_func_face1 = scalar2[:, None] * mix_func_ki + scalar1[:, None] * mix_func_kj
  mix_func_face2 = scalar3[:, None] * mix_func_ji

  # calculate curl for mix functions
  for i in range(scalar1.shape[0]):
      # curl_func_face1[i] = 3*(scalar1[i]*(np.cross(grad3, grad2)) + scalar2[i]*np.cross(grad3, grad1))
      # curl_func_face2[i] = (-2)*scalar3[i]*(np.cross(grad1, grad2)) - scalar2[i]*(np.cross(grad1, grad3)) + scalar1[i]*(np.cross(grad2, grad3))
      curl_func_face1[i] = scalar2[i]*mix_curl_ki[i] + np.cross(grad2, mix_func_ki[i]) + scalar1[i]*mix_curl_kj[i] + np.cross(grad1, mix_func_kj[i])
      curl_func_face2[i] = scalar3[i]*mix_curl_ji[i] + np.cross(grad3, mix_func_ji[i])
      full_curl_face[i] = [0,0,0]

  # combine all the basis functions
  all_combined = np.stack((full_func_face, mix_func_face1, mix_func_face2), axis=0)
  return full_func_face, mix_func_face1, mix_func_face2, all_combined, curl_func_face1, curl_func_face2, full_curl_face

def hierarchical_basis3D_order2_all(nodes, points):

  scalar1, grad1 = basis3Dscalar_numba(nodes, 0, points)
  scalar2, grad2 = basis3Dscalar_numba(nodes, 1, points)
  scalar3, grad3 = basis3Dscalar_numba(nodes, 2, points)
  scalar4, grad4 = basis3Dscalar_numba(nodes, 3, points)

  scalar_list = [scalar1, scalar2, scalar3, scalar4]
  grad_list = [grad1, grad2, grad3, grad4]

  mixed_face_list = []
  full_edge_list = []
  full_face_list = []

  curl_mix_list = []
  curl_full1_list = []
  curl_full2_list = []

  # combine all the edge func
  for i, j in itertools.combinations(range(4), 2):
    edge, curl = hierarchical_basis3D_order2_edge_helper(scalar_list[i], scalar_list[j], grad_list[i], grad_list[j])
    full_edge_list.append(edge)
    curl_full1_list.append(curl)

  # combine all the faces func
  for i, j, k in itertools.combinations(range(4), 3):
    full, mixed1, mixed2, _, curl1, curl2, curl3 = hierarchical_basis3D_order2_face_helper(scalar_list[i], scalar_list[j], scalar_list[k],  grad_list[i],  grad_list[j],  grad_list[k])
    mixed_face_list.append(mixed1)
    mixed_face_list.append(mixed2)
    curl_mix_list.append(curl1)
    curl_mix_list.append(curl2)
    
    full_face_list.append(full)
    curl_full2_list.append(curl3)

  full_edge_stack = np.stack(full_edge_list, axis=0)
  full_face_stack = np.stack(full_face_list, axis=0)
  mixed_face_stack = np.stack(mixed_face_list, axis=0)

  curl_mix_stack = np.stack(curl_mix_list, axis=0)
  curl_full1_stack = np.stack(curl_full1_list, axis=0)
  curl_full2_stack = np.stack(curl_full2_list, axis=0)

  # combine order2 and order1
  combined_func_order1, combined_curl_order1 = hierarchical_basis3D_order1_all(nodes, points)
  
  combined_func_order2 = np.concatenate((mixed_face_stack, full_edge_stack, full_face_stack), axis=0)
  combined_func_all = np.concatenate((combined_func_order1, combined_func_order2), axis=0)

  combined_curl_order2 = np.concatenate((curl_mix_stack, curl_full1_stack, curl_full2_stack), axis=0)
  combined_curl_all = np.concatenate((combined_curl_order1, combined_curl_order2), axis=0)

  return combined_func_all, combined_curl_all

# use this
def hierarchical_basis3D_order2(nodes, points):
  """
  Description: This is a function that returns the second order mixed and all first order hierarchical basis functions and curls at integral points
in a tetrahedral.

  Inputs:
    nodes - coordinates of each node (vertex) of the tetrahedron
    points - coordinates of each of the evaluation points

  Outputs:
    combined_func_all - the second order mixed and all first order hierarchical basis functions and curls at integral points
in a tetrahedral.
  """
  scalar1, grad1 = basis3Dscalar_numba(nodes, 0, points)
  scalar2, grad2 = basis3Dscalar_numba(nodes, 1, points)
  scalar3, grad3 = basis3Dscalar_numba(nodes, 2, points)
  scalar4, grad4 = basis3Dscalar_numba(nodes, 3, points)

  scalar_list = [scalar1, scalar2, scalar3, scalar4]
  grad_list = [grad1, grad2, grad3, grad4]
  mixed_list = []

  curl_mix_list = []

  # combine all the mixed faces func
  for i, j, k in itertools.combinations(range(4), 3):
    _, mixed1, mixed2, _, curl1, curl2, _ = hierarchical_basis3D_order2_face_helper(scalar_list[i], scalar_list[j], scalar_list[k],  grad_list[i],  grad_list[j],  grad_list[k])
    mixed_list.append(mixed1)
    mixed_list.append(mixed2)

    curl_mix_list.append(curl1)
    curl_mix_list.append(curl2)

  # combine order2 and order1
  combined_func_order2 = np.stack(mixed_list, axis=0)
  combined_curl_order2 = np.stack(curl_mix_list, axis=0)
  combined_func_order1, combined_curl_order1 = hierarchical_basis3D_order1_all(nodes, points)
  

  combined_func_all = np.concatenate((combined_func_order1, combined_func_order2), axis=0)
  combined_curl_all = np.concatenate((combined_curl_order1, combined_curl_order2), axis=0)

  return combined_func_all, combined_curl_all

def order3_calc_helper(scalar1, scalar2, scalar3, grad1, grad2, grad3, grad4):
  """
  this is a function to calculate curl of 
  "scalar1*scalar2*scalar3*grad4"
  """
  result = np.zeros((scalar1.shape[0], 3))

  for i in range(scalar1.shape[0]):
      result[i] = scalar2[i]*scalar3[i]*np.cross(grad1, grad4) + scalar1[i]*scalar3[i]*np.cross(grad2, grad4) + scalar1[i]*scalar2[i]*np.cross(grad3, grad4)
  return result

def hierarchical_basis3D_order3_edge_helper(scalar1, scalar2, grad1, grad2):
  full_func_edge = np.zeros((scalar1.shape[0], 3))
  for i in range(scalar1.shape[0]):
    for j in range(3):
      full_func_edge[i,j] = grad1[j]*scalar2[i]*(362*scalar1[i]*scalar1[i] + 362*scalar2[i]*scalar2[i] - 504*scalar1[i]*scalar2[i] - 117*scalar1[i] - 117*scalar2[i] - 16)
      + scalar1[i]*grad2[j]*(362*scalar1[i]*scalar1[i] + 362*scalar2[i]*scalar2[i] - 504*scalar1[i]*scalar2[i] - 117*scalar1[i] - 117*scalar2[i] - 16)
      + scalar1[i]*scalar2[i]*(362*2*scalar1[i]*grad1[j] + 362*2*scalar2[i]*grad2[j] - 504*(grad1[j]*scalar2[i] + scalar1[i]*grad2[j]) - 117*grad1[j] - 117*grad2[j])

  return full_func_edge

def hierarchical_basis3D_order3_face_helper(scalar1, scalar2, scalar3, grad1, grad2, grad3):

  full_func_face1 = np.zeros((scalar1.shape[0], 3))
  full_func_face2 = np.zeros((scalar1.shape[0], 3))

  mix_func_face1 = np.zeros((scalar1.shape[0], 3))
  mix_func_face2 = np.zeros((scalar1.shape[0], 3))
  mix_func_face3 = np.zeros((scalar1.shape[0], 3))

  mix_curl_face1 = np.zeros((scalar1.shape[0], 3))
  mix_curl_face2 = np.zeros((scalar1.shape[0], 3))
  mix_curl_face3 = np.zeros((scalar1.shape[0], 3))

  mix_curl_face1 = order3_calc_helper(scalar2-scalar3, scalar2, scalar3, grad2-grad3, grad2, grad3, grad1) + order3_calc_helper(scalar3-scalar1, scalar3, scalar1, grad3-grad1, grad3, grad1, grad2) + order3_calc_helper(scalar1-scalar2, scalar1, scalar2, grad1-grad2, grad1, grad2, grad3)
  
  mix_curl_face2 = order3_calc_helper(80*scalar2-212*scalar3+393*scalar1, scalar2, scalar3, 80*grad2-212*grad3+393*grad1, grad2, grad3, grad1) + order3_calc_helper(212*scalar3-80*scalar1+393*scalar2, scalar3, scalar1, 212*grad3-80*grad1+393*grad2, grad3, grad1, grad2) - 292*order3_calc_helper(scalar1-scalar2, scalar1, scalar2, grad1-grad2, grad1, grad2, grad3)

  mix_curl_face3 = order3_calc_helper(186*scalar2-124*scalar3-131*scalar1, scalar2, scalar3, 186*grad2-124*grad3-131*grad1, grad2, grad3, grad1) + order3_calc_helper(124*scalar3-168*scalar1+393*scalar2, scalar3, scalar1, 124*grad3-168*grad1+393*grad2, grad3, grad1, grad2) - order3_calc_helper(44*scalar1+44*scalar2-262*scalar3, scalar1, scalar2, 44*grad1+44*grad2-262*grad3, grad1, grad2, grad3)

  for i in range(scalar1.shape[0]):
    for j in range(3):
      mix_func_face1[i,j] = (scalar2[i] - scalar3[i])*scalar2[i]*scalar3[i]*grad1[j] \
      + (scalar3[i] - scalar1[i])*scalar3[i]*scalar1[i]*grad2[j] \
      + (scalar1[i] - scalar2[i])*scalar1[i]*scalar2[i]*grad3[j]

      mix_func_face2[i,j] = (80*scalar2[i] - 212*scalar3[i] + 393*scalar1[i])*scalar2[i]*scalar3[i]*grad1[j] \
      + (212*scalar3[i] - 80*scalar1[i] + 393*scalar2[i])*scalar3[i]*scalar1[i]*grad2[j] \
      - 292*(scalar1[i] - scalar2[i])*scalar1[i]*scalar2[i]*grad3[j]

      mix_func_face3[i,j] = (186*scalar2[i] - 124*scalar3[i] - 131*scalar1[i])*scalar2[i]*scalar3[i]*grad1[j] \
      + (124*scalar3[i] - 168*scalar1[i] + 393*scalar2[i])*scalar3[i]*scalar1[i]*grad2[j] \
      - (44*scalar1[i] + 44*scalar2[i] - 262*scalar3[i])*scalar1[i]*scalar2[i]*grad3[j]

      full_func_face1[i,j] = grad1[j]*scalar2[i]*scalar3[i]*(scalar1[i] + scalar2[i] - 2*scalar3[i]) \
      + scalar1[i]*grad2[j]*scalar3[i]*(scalar1[i] + scalar2[i] - 2*scalar3[i]) \
      + scalar1[i]*scalar2[i]*grad3[j]*(scalar1[i] + scalar2[i] - 2*scalar3[i]) \
      + scalar1[i]*scalar2[i]*scalar3[i]*(grad1[j] + grad2[j] - 2*grad3[j])

      full_func_face2[i,j] = grad1[j]*scalar2[i]*scalar3[i]*(scalar1[i] - scalar2[i]) \
      + scalar1[i]*grad2[j]*scalar3[i]*(scalar1[i] - scalar2[i]) \
      + scalar1[i]*scalar2[i]*grad3[j]*(scalar1[i] - scalar2[i]) \
      + scalar1[i]*scalar2[i]*scalar3[i]*(grad1[j] - grad2[j])

  all_combined = np.stack((mix_func_face1, mix_func_face2, mix_func_face3, full_func_face1, full_func_face2), axis=0)

  return mix_func_face1, mix_func_face2, mix_func_face3, full_func_face1, full_func_face2, all_combined, mix_curl_face1, mix_curl_face2, mix_curl_face3

def hierarchical_basis3D_order3_volume_helper(scalar1, scalar2, scalar3, scalar4, grad1, grad2, grad3, grad4):

  mix_func_volume1 = np.zeros((scalar1.shape[0], 3))
  mix_func_volume2 = np.zeros((scalar1.shape[0], 3))
  mix_func_volume3 = np.zeros((scalar1.shape[0], 3))
  full_func_volume1 = np.zeros((scalar1.shape[0], 3))

  mix_curl_volume1 = np.zeros((scalar1.shape[0], 3))
  mix_curl_volume2 = np.zeros((scalar1.shape[0], 3))
  mix_curl_volume3 = np.zeros((scalar1.shape[0], 3))

  _, mix_func_kl, _, mix_curl_kl, _ = hierarchical_basis3D_order1_helper(scalar3, scalar4, grad3, grad4)
  _, mix_func_kj, _, mix_curl_kj, _ = hierarchical_basis3D_order1_helper(scalar3, scalar2, grad3, grad2)
  _, mix_func_ji, _, mix_curl_ji, _ = hierarchical_basis3D_order1_helper(scalar2, scalar1, grad2, grad1)
  _, mix_func_il, _, mix_curl_il, _ = hierarchical_basis3D_order1_helper(scalar1, scalar4, grad1, grad4)

  for i in range(scalar1.shape[0]):
    mix_curl_volume1[i] = (scalar2[i]*np.cross(grad1, mix_func_kl[i]) + scalar1[i]*np.cross(grad2, mix_func_kl[i])
    + scalar1[i]*scalar2[i]*mix_curl_kl[i] + scalar4[i]*np.cross(grad3, mix_func_ji[i]) + scalar3[i]*np.cross(grad4, mix_func_ji[i])
    + scalar3[i]*scalar4[i]*mix_curl_ji[i])
    
    mix_curl_volume2[i] = scalar1[i]*np.cross(grad4, mix_func_kj[i]) + scalar4[i]*np.cross(grad1, mix_func_kj[i]) + scalar4[i]*scalar1[i]*mix_curl_kj[i]
    
    mix_curl_volume3[i] = (scalar3[i]*np.cross(grad2, mix_func_il[i]) + scalar2[i]*np.cross(grad3, mix_func_il[i])
    + scalar2[i]*scalar3[i]*mix_curl_il[i])


    for j in range(3):
      mix_func_volume1[i,j] = (scalar1[i]*scalar2[i]*(scalar3[i]*grad4[j] - scalar4[i]*grad3[j])
      + scalar3[i]*scalar4[i]*(scalar2[i]*grad1[j] - scalar1[i]*grad2[j]))

      mix_func_volume2[i,j] = scalar4[i]*scalar1[i]*(scalar3[i]*grad2[j] - scalar2[i]*grad3[j])

      mix_func_volume3[i,j] = scalar2[i]*scalar3[i]*(scalar1[i]*grad4[j] - scalar4[i]*grad1[j])

      full_func_volume1[i,j] = (grad1[j]*scalar2[i]*scalar3[i]*scalar4[i] + scalar1[i]*grad2[j]*scalar3[i]*scalar4[i]
      + scalar1[i]*scalar2[i]*grad3[j]*scalar4[i] + scalar1[i]*scalar2[i]*scalar3[i]*grad4[j])

  all_combined = np.stack((mix_func_volume1, mix_func_volume2, mix_func_volume3, full_func_volume1), axis=0)

  return mix_func_volume1, mix_func_volume2, mix_func_volume3, full_func_volume1, all_combined, mix_curl_volume1, mix_curl_volume2, mix_curl_volume3

# use this
def hierarchical_basis3D_order3(nodes, points):
  """
  Description: This is a function that returns mixed third order and all second with first order hierarchical basis functions at integral points
in a tetrahedral.

  Inputs:
    nodes - coordinates of each node (vertex) of the tetrahedron
    points - coordinates of each of the evaluation points

  Outputs:
    combined_func_all - mixed third order and all second with first order hierarchical basis functions at integral points
in a tetrahedral.
  """
  scalar1, grad1 = basis3Dscalar_numba(nodes, 0, points)
  scalar2, grad2 = basis3Dscalar_numba(nodes, 1, points)
  scalar3, grad3 = basis3Dscalar_numba(nodes, 2, points)
  scalar4, grad4 = basis3Dscalar_numba(nodes, 3, points)

  scalar_list = [scalar1, scalar2, scalar3, scalar4]
  grad_list = [grad1, grad2, grad3, grad4]
  
  mix_face = []
  curl_face = []
  mix_volume = []
  curl_volume = []

  for i, j, k in itertools.combinations(range(4), 3):
    mix1, mix2, mix3, _, _, _, curl1, curl2, curl3 = hierarchical_basis3D_order3_face_helper(scalar_list[i], scalar_list[j], scalar_list[k], grad_list[i], grad_list[j], grad_list[k])
    mix_face.append(mix1)
    mix_face.append(mix2)
    mix_face.append(mix3)
    curl_face.append(curl1)
    curl_face.append(curl2)
    curl_face.append(curl3)

  mix1, mix2, mix3, _, _, curl1, curl2, curl3 = hierarchical_basis3D_order3_volume_helper(scalar_list[0], scalar_list[1], scalar_list[2], scalar_list[3], grad_list[0], grad_list[1], grad_list[2], grad_list[3])
  mix_volume.append(mix1)
  mix_volume.append(mix2)
  mix_volume.append(mix3)
  curl_volume.append(curl1)
  curl_volume.append(curl2)
  curl_volume.append(curl3)
  

  mix_face = np.stack(mix_face, axis=0)
  mix_volume = np.stack(mix_volume, axis=0)
  combined_order2, curl_order2 = hierarchical_basis3D_order2_all(nodes, points)
  combined_order3 = np.concatenate((mix_face, mix_volume), axis=0)
  combined_func_all = np.concatenate((combined_order2, combined_order3), axis=0)

  curl_face = np.stack(curl_face, axis=0)
  curl_volume = np.stack(curl_volume, axis=0)
  curl_order3 = np.concatenate((curl_face, curl_volume), axis=0)
  curl_all = np.concatenate((curl_order2, curl_order3), axis=0)

  return combined_func_all, curl_all

def hierarchical_basis3D_grad_order1_helper(scalar1, grad1):
  first_order_grad = np.zeros((scalar1.shape[0], 3))
  for i in range(scalar1.shape[0]):
    for j in range(3):
      first_order_grad[i,j] = grad1[j]
  return first_order_grad

# use this
def hierarchical_basis3D_scalar_order1(nodes, points):
  """
  This function returns scalar basis functions up to the second order, and the gradient of them.
  
  Inputs:
  nodes: coordinates of each node (vertex) of the tetrahedron
  points: coordinates of each of the evaluation points

  Outputs:
  together_scalar_stack: scalar basis function including order1. It has order of [first order]. Detail in same as Jin's book p.297.
  together_grad_stack: gradient of the scalar basis functions including order1. It has order of [first order]. Detail in same as Jin's book p.297.
  """

  f1, g1 = basis3Dscalar_numba(nodes, 0, points)
  f2, g2 = basis3Dscalar_numba(nodes, 1, points)
  f3, g3 = basis3Dscalar_numba(nodes, 2, points)
  f4, g4 = basis3Dscalar_numba(nodes, 3, points)
  simplex_coords = [f1, f2, f3, f4]
  simplex_grad = [g1, g2, g3, g4]
  
  first_order_grad_list = []

  first_order_scalar_list = simplex_coords
  first_order_scalar_stack = np.stack(first_order_scalar_list, axis=0)

  for i in range(4):
    first_order_grad = hierarchical_basis3D_grad_order1_helper(simplex_coords[i], simplex_grad[i])
    first_order_grad_list.append(first_order_grad)

  first_order_grad_stack = np.stack( first_order_grad_list, axis=0)

  return first_order_scalar_stack, first_order_grad_stack

def hierarchical_basis3D_scalar_order2_helper(scalar1, scalar2):
  basis_edge = np.zeros((scalar1.shape[0]))

  for i in range(scalar1.shape[0]):
    basis_edge[i] = scalar1[i]*scalar2[i]

  return basis_edge

# use this
def hierarchical_basis3D_scalar_order2(nodes, points):
  """
  This function returns scalar basis functions up to the second order, and the gradient of them.
  
  Inputs:
  nodes: coordinates of each node (vertex) of the tetrahedron
  points: coordinates of each of the evaluation points

  Outputs:
  together_scalar_stack: scalar basis function including order1 and order2. It has order of [first order, second order]. Detail in same as Jin's book p.297.
  together_grad_stack: gradient of the scalar basis functions including order1 and order2. It has order of [first order, second order]. Detail in same as Jin's book p.297.
  """

  f1, g1 = basis3Dscalar_numba(nodes, 0, points)
  f2, g2 = basis3Dscalar_numba(nodes, 1, points)
  f3, g3 = basis3Dscalar_numba(nodes, 2, points)
  f4, g4 = basis3Dscalar_numba(nodes, 3, points)
  simplex_coords = [f1, f2, f3, f4]
  simplex_grads = [g1, g2, g3, g4]

  second_order_scalar_list = []
  second_order_grad_list = []
  
  for i, j in itertools.combinations(range(4), 2):
    basis_edge = hierarchical_basis3D_scalar_order2_helper(simplex_coords[i], simplex_coords[j])
    grad_edge, _, _, _, _ = hierarchical_basis3D_order1_helper(simplex_coords[i], simplex_coords[j], simplex_grads[i], simplex_grads[j])
    second_order_scalar_list.append(basis_edge)
    second_order_grad_list.append(grad_edge)

  first_order_scalar_stack, first_order_grad_stack = hierarchical_basis3D_scalar_order1(nodes, points)
  second_order_scalar_stack = np.stack(second_order_scalar_list, axis=0)
  second_order_grad_stack = np.stack(second_order_grad_list, axis=0)

  together_scalar_stack = np.concatenate((first_order_scalar_stack, second_order_scalar_stack), axis=0)
  together_grad_stack = np.concatenate((first_order_grad_stack, second_order_grad_stack), axis=0)

  return together_scalar_stack, together_grad_stack

def hierarchical_basis3D_scalar_order3_edge_helper(scalar1, scalar2):
  basis_edge = np.zeros((scalar1.shape[0]))
  
  for i in range(scalar1.shape[0]):
    basis_edge[i] = scalar1[1]*scalar2[i]*(scalar1[i] - scalar2[i])

  return basis_edge

def hierarchical_basis3D_scalar_order3_face_helper(scalar1, scalar2, scalar3):
  basis_face = np.zeros((scalar1.shape[0]))

  for i in range(scalar1.shape[0]):
    basis_face[i] = scalar1[i]*scalar2[i]*scalar3[i]

  return basis_face

# use this
def hierarchical_basis3D_scalar_order3(nodes, points):
  """
  This function returns scalar basis functions up to the third order, and the gradient of them.
  
  Inputs:
  nodes: coordinates of each node (vertex) of the tetrahedron
  points: coordinates of each of the evaluation points

  Outputs:
  together_scalar_stack: scalar basis function including order1, order2, and order3. It has order of [first order, second order, third edge, third face]. Detail in same as Jin's book p.297.
  together_grad_stack: gradient of the scalar basis functions including order1, order2, and order3. It has order of [first order, second order, third edge, third face]. Detail in same as Jin's book p.297.
  """
  f1, g1 = basis3Dscalar_numba(nodes, 0, points)
  f2, g2 = basis3Dscalar_numba(nodes, 1, points)
  f3, g3 = basis3Dscalar_numba(nodes, 2, points)
  f4, g4 = basis3Dscalar_numba(nodes, 3, points)
  simplex_coords = [f1, f2, f3, f4]
  simplex_grads = [g1, g2, g3, g4]

  third_order_scalar_list = []
  third_order_grad_list = []

  for i, j in itertools.combinations(range(4), 2):
    basis_edge = hierarchical_basis3D_scalar_order3_edge_helper(simplex_coords[i], simplex_coords[j])
    third_order_scalar_list.append(basis_edge)

    grad_edge, _ = hierarchical_basis3D_order2_edge_helper(simplex_coords[i], simplex_coords[j], simplex_grads[i], simplex_grads[j])
    third_order_grad_list.append(grad_edge)

  for i, j, k in itertools.combinations(range(4), 3):
    basis_face = hierarchical_basis3D_scalar_order3_face_helper(simplex_coords[i], simplex_coords[j], simplex_coords[k])
    third_order_scalar_list.append(basis_face)
    grad_face, _, _, _, _, _, _ = hierarchical_basis3D_order2_face_helper(simplex_coords[i], simplex_coords[j], simplex_coords[k], simplex_grads[i], simplex_grads[j], simplex_grads[k])
    third_order_grad_list.append(grad_face)

  second_order_scalar_stack, second_order_grad_stack = hierarchical_basis3D_scalar_order2(nodes, points)
  third_order_scalar_stack = np.stack(third_order_scalar_list, axis=0)
  third_order_grad_stack = np.stack(third_order_grad_list, axis=0)

  together_scalar_stack = np.concatenate((second_order_scalar_stack, third_order_scalar_stack), axis=0)
  together_grad_stack = np.concatenate((second_order_grad_stack, third_order_grad_stack), axis=0)

  return together_scalar_stack, together_grad_stack

