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
import sys
import time
from itertools import repeat
from multiprocessing import Pool
import gauss_quadrature_tetra as q
from basis_funcs import basis_vector_numba, basis3Dscalar_numba, basis2DVquad, basis2DSquad, basis2Dscalar_numba
from assemble_mats import selectInds3D, basisSetupTet, basisSetupWedge, getWedgeLocs, getUnknownCount, splitTet, selectVecInds2D, selectScalInds2D
# import meshio
from itertools import repeat
from meshToMats import splitArr



def getPortVoltages(x, voltVecInds, voltVecVals, voltScalInds, voltScalVals, numVecUn, portP, simType, probType, freqs=-1, dt=-1):
  if probType == "FD":
    V_ports = np.zeros((np.size(freqs), portP["num"]), dtype=x.dtype)
    for ii in range(len(freqs)):
      for portNum in range(portP["num"]):
        x_port = np.copy(x[voltVecInds[portNum],ii]) # Isolate the solution at the port
        V_ports[ii,portNum] = np.sum(voltVecVals[portNum]*x_port)
        if simType == "A": # Need to scale the magnetic vector potential for A-phi
          V_ports[ii,portNum] *= 1j*2*np.pi*freqs[ii] # Scale the magnetic vector potential
          x_port = np.copy(x[voltScalInds[portNum]+numVecUn,ii]) # Add in the effect of the scalar potential
          V_ports[ii,portNum] += -np.sum(voltScalVals[portNum]*x_port)
      
  elif probType == "TD":
    V_ports = np.zeros((portP["num"],), dtype=x.dtype)
    for portNum in range(portP["num"]):
        x_port = np.copy(x[voltVecInds[portNum]]) # Isolate the solution at the port
        if simType == "E":
          V_ports[portNum] = np.sum(voltVecVals[portNum]*x_port)
        elif simType == "A": # Need to scale the magnetic vector potential for A-phi
          V_ports[portNum] = np.sum(voltVecVals[portNum]*(-1/dt*(x_port[:,1] - x_port[:,0]))) # Use finite differencing to evaluate the time derivative
          # Add in the effect of the scalar potential
          x_port = np.copy(x[voltScalInds[portNum]+numVecUn,ii]) # Add in the effect of the scalar potential
          V_ports[ii,portNum] += -np.sum(voltScalVals[portNum]*x_port)

  return V_ports

def getPortCharges(qVec, scalNodeMap, nodeList, portList, portP):
  """
  Given charge density at each node of the mesh, calculates the charge across a set of ports

  Inputs:
    qVec        - Vector of charge densities at each node of the mesh
    scalNodeMap - Mapping from nodes of the mesh to scalar unknowns
    nodeList    - Renumbers nodes around PEC and contains node coordinates
    portList    - List of nodes forming each triangle of each port
    portP       - Parameters of each port

  Outputs:
    portQ - Charge across each port
  """
  portQ = np.zeros((portP["num"],), dtype=np.float64)
  numNodes = int(np.max(nodeList[:,0])) + 1

  for portNum in range(portP["num"]):
    startInd = portP[f"{portNum}"]["startInd"]
    if portNum == portP["num"] - 1:
      endInd = len(portList)
    else:
      endInd = portP[f"{portNum+1}"]["startInd"]

    portNodes = np.unique(portList[startInd:endInd])
    portPECNodes = portNodes[scalNodeMap[portNodes] >= numNodes]
    _, pecInds = np.unique(scalNodeMap[portPECNodes], return_index=True)
    numPEC = len(pecInds)
    if numPEC == 1:
      # Only 1 PEC on the port, just look at that charge density
      portQ[portNum] = qVec[scalNodeMap[portPECNodes[pecInds[0]]]]
    elif numPEC == 2:
      # 2 PEC on the port, so take the difference
      pecNodeCoords = nodeList[portPECNodes[pecInds], 1:]
      if pecNodeCoords[0] @ portP[f"{portNum}"]["direction"] > pecNodeCoords[1] @ portP[f"{portNum}"]["direction"]:
        portQ[portNum] = qVec[scalNodeMap[portPECNodes[pecInds[0]]]] - qVec[scalNodeMap[portPECNodes[pecInds[1]]]]
      else:
        portQ[portNum] = qVec[scalNodeMap[portPECNodes[pecInds[1]]]] - qVec[scalNodeMap[portPECNodes[pecInds[0]]]]

  return portQ

def getOutInts(x, connLists, basisMaps, singEleLists, outP, quadOrder, basisOrder, simType, numVecUn, probType, freqs=-1, dt=-1):

  outList, outEdgeList, outFaceList, nodeList = \
    [connLists["outList"], connLists["outEdgeList"], connLists["outFaceList"], connLists["nodeList"]]
  vecEdgeMap, vecFaceMap, scalNodeMap, scalEdgeMap = \
    [basisMaps["vecEdgeMap"], basisMaps["vecFaceMap"], basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"]]

  match basisOrder: # Handle different basis function numbering for higher order cases
    case 1:
      numVecBasis = 3
      numScalBasis = 3
    case 2:
      numVecBasis = 8
      numScalBasis = 6
  
  ep0 = 1e-9 / (36 * np.pi)
  muInv = 1/(4e-7*np.pi)
  outIntegrals = np.zeros((len(freqs), outP["num"]), dtype=x.dtype)
  
  # Check for invalid field type combinations and if surface participation is calculated
  invalidSurfaces = []
  surfPartFlag = False
  for ii in range(outP["num"]):
    if outP[f"{ii}"]["fieldType"] == "A" and simType == "E":
      invalidSurfaces.append(ii)
    elif outP[f"{ii}"]["fieldType"] == "surfPart":
      surfPartFlag = True
  
  if len(invalidSurfaces) > 0:
    print(f"Error: A-field output not supported for E-field formulation (simType='E')")
    print(f"Invalid surfaces: {invalidSurfaces}")
    exit(1)
  
  # If surface participation calculations are included, calculate the total field in the volume
  if surfPartFlag:
    totFieldEnergy = np.zeros((len(freqs),), dtype=np.float64) # Total field magnitude throughout the geometry at each frequency
    for ii, freq in enumerate(freqs):
      _, totFieldEnergy[ii] = getTotField(connLists, basisMaps, singEleLists, simType, quadOrder, basisOrder, x, probType, "E", "all", 2*np.pi*freq)
  
  edgeNums = np.array([[0,1], [0,2], [1,2]])
  for ii in range(outP["num"]): # Loop through each output surface
    startInd = outP[f"{ii}"]["startInd"]
    if ii == outP["num"] - 1:
      endInd = len(outList)
    else:
      endInd = outP[f"{ii+1}"]["startInd"]

    vecInts = np.zeros((len(outList), numVecBasis))
    if simType == "A" or simType == "TCS":
      scalInts = np.zeros((len(outList), numScalBasis))
      outVecVals = np.zeros(((numVecBasis+numScalBasis)*len(outList),), dtype=np.float64)
    else:
      scalInts = []
      outVecVals = np.zeros(((numVecBasis)*len(outList),), dtype=np.float64)
    outVecInds = np.zeros(outVecVals.shape, dtype=np.int32)
    
    for triNum in range(startInd, endInd):
      tri = outList[triNum]
      coords = nodeList[tri][:,1:]
      loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, coords[0], coords[1], coords[2])) 
      if basisOrder == 1:
        edgeValsTri = [basis_vector_numba(coords, edgeNums[jj,0], edgeNums[jj,1], loc)[0] for jj in range(3)]
        curlBasisTri = [basis_vector_numba(coords, edgeNums[jj,0], edgeNums[jj,1], loc)[1] for jj in range(3)]
        if simType == "A" or simType == "TCS":
          gradBasisTri = [basis2Dscalar_numba(coords, nodeInd, loc)[1] for nodeInd in range(3)]
      elif basisOrder == 2:
        edgeValsTri, curlBasisTri = basis2DVquad(coords, loc)
        if simType == "A" or simType == "TCS":
          _, gradBasisTri = basis2DSquad(coords, loc)
      curEleVals = np.zeros((len(freqs), len(loc), 3), dtype=x.dtype) # Value of the field at each integration point (only for surface participation)

      for edgeInd in range(numVecBasis):
        colVal = selectVecInds2D(basisOrder, edgeInd, vecEdgeMap, vecFaceMap, outEdgeList[triNum], outFaceList[triNum])
        if colVal == -1:
          continue

        edgeVals = edgeValsTri[edgeInd]
        curlBasis = curlBasisTri[edgeInd]
        
        # Integrate the basis function over the triangle and store the result
        
        match outP[f"{ii}"]["fieldType"]:
          case "E":
            vecInts[triNum, edgeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], edgeVals @ outP[f"{ii}"]["direction"])
            for jj in range(len(freqs)):
              if simType == "E":
                outIntegrals[jj, ii] += -vecInts[triNum, edgeInd] * np.copy(x[colVal, jj])
              elif simType == "A":
                outIntegrals[jj, ii] += -vecInts[triNum, edgeInd] * np.copy(x[colVal, jj]) * (-1j*2*np.pi*freqs[jj])
          case "H":
            for jj in range(len(freqs)):
              vecInts[triNum, edgeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], muInv * curlBasis @ outP[f"{ii}"]["direction"])
              if simType == "E":
                outIntegrals[jj, ii] += -vecInts[triNum, edgeInd] * np.copy(x[colVal, jj])/(1j*2*np.pi*freqs[jj])
              elif simType == "A":
                outIntegrals[jj, ii] += -vecInts[triNum, edgeInd] * np.copy(x[colVal, jj])
          case "A":
            vecInts[triNum, edgeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], edgeVals @ outP[f"{ii}"]["direction"])
            for jj in range(len(freqs)):
              outIntegrals[jj, ii] += -vecInts[triNum, edgeInd] * np.copy(x[colVal, jj])
          case "surfPart": # Surface participation
            for jj in range(len(freqs)):
              if simType == "E":
                curEleVals[jj] += edgeVals*x[colVal, jj]
              elif simType == "A":
                curEleVals[jj] += edgeVals*x[colVal, jj]*(-1j*2*np.pi*freqs[jj])

      
      if (outP[f"{ii}"]["fieldType"] == "E" or outP[f"{ii}"]["fieldType"] == "surfPart") and simType == "A": # Loop through nodes outside the edge loop
        nodeNums = nodeList[outList[triNum], 0]
        for nodeInd in range(numScalBasis):
          colValScal = selectScalInds2D(nodeInd, scalNodeMap, scalEdgeMap, outList[triNum], outEdgeList[triNum], nodeNums, edgeNums)
          if colValScal == -1:
            continue
          
          # Evaluate the scalar basis function at the integration points
          gradVals = gradBasisTri[nodeInd]

          if outP[f"{ii}"]["fieldType"] == "E":
            # Integrate the basis function over the triangle and store the result
            if basisOrder == 1: # Handle 1st order case seperately b/c the gradient is constant and defined at only a single point
              scalInts[triNum, nodeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.array(list(repeat(gradVals @ outP[f"{ii}"]["direction"],len(loc)))))
            else:
              scalInts[triNum, nodeInd] = q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], gradVals @ outP[f"{ii}"]["direction"])

            for jj in range(len(freqs)):
              outIntegrals[jj, ii] += -scalInts[triNum, nodeInd] * np.copy(x[numVecUn + colValScal, jj])
          elif outP[f"{ii}"]["fieldType"] == "surfPart":
            for jj in range(len(freqs)):
              if basisOrder == 1:
                curEleVals[jj] += np.array(list(repeat(gradVals*x[colValScal,jj], len(loc))))
              else:
                curEleVals[jj] += gradVals*x[colValScal, jj]

      if outP[f"{ii}"]["fieldType"] == "surfPart":
        for jj in range(len(freqs)): # Once all the contributions from basis functions are summed, integrate the magnitude squared
          outIntegrals[jj, ii] += q.gauss_quadrature_trisimp(coords[0,:], coords[1,:], coords[2,:], np.real(np.sum(np.conj(curEleVals[jj]) * curEleVals[jj], axis=1)))
            

    if outP[f"{ii}"]["fieldType"] == "surfPart": 
      for jj in range(len(freqs)): # For surface participation, add scaling factors
        outIntegrals[jj,ii] *= outP[f"{ii}"]["thickness"] * (ep0*outP[f"{ii}"]["epr"]) / totFieldEnergy
            
  return outIntegrals


def getOutInts_tets_loop(x, connLists, matStruct, basisMaps, singEleLists, outP, quadOrder, basisOrder, singOrder, basisOrder_long, singOrder_long, N_tet, \
                         simType, numVecUn, probType, numProcs, freqs=-1, dt=-1):
  """
  Function for enabling parallelization of 'getOutInts_tets'. See that function for details on inputs and outputs.
  """
  outList, outEdgeList, outFaceList, outTetList = \
    [connLists["outList"], connLists["outEdgeList"], connLists["outFaceList"], connLists["outTetList"]]

  outSplit, startInd = splitArr(outList, numProcs)
  outEdgeSplit, _ = splitArr(outEdgeList, numProcs)
  outFaceSplit, _ = splitArr(outFaceList, numProcs)
  outTetSplit, _ = splitArr(outTetList, numProcs)

  p = Pool(numProcs) # Set up parallel pool
  ## Perform integrals over each sub-list
  outIntsArr = list(p.starmap(getOutInts_tets, zip(repeat(x), repeat(connLists), outSplit, outEdgeSplit, outFaceSplit, outTetSplit, \
                                                   repeat(matStruct), repeat(basisMaps), repeat(singEleLists), repeat(outP), startInd, \
                                                   repeat(quadOrder), repeat(basisOrder), repeat(singOrder), repeat(basisOrder_long), repeat(singOrder_long), repeat(N_tet), repeat(simType), repeat(numVecUn), \
                                                   repeat(probType), repeat(freqs), repeat(dt))))
  outInts = np.sum(outIntsArr, axis=0)

  return outInts

def getOutInts_tets(x, connLists, outList, outEdgeList, outFaceList, outTetList, matStruct, basisMaps, singEleLists, \
                    outP, startTri, quadOrder, basisOrder, singOrder, basisOrder_long, singOrder_long, \
                    N_tet, simType, numVecUn, probType, freqs=-1, dt=-1):
  """
  Calculates surface integrals on output surfaces using neighboring tets (allowing correct surface participation and magnetic field calculations)

  Inputs:
    x            - The vector output by the FEM
    connLists    - Dictionary containing connectivity information
    matStruct    - Dictionary containing FEM matrices
    basisMaps    - Dictionary containing mappings between geometric entities and basis functions
    singEleLists - Dictionary containing information about singular elements
    outP         - Dictionary with parameters for each output surface
    startEle     - Marks the element number of the first entry of outList (needed for parallelization to work correctly)
    quadOrder    - Gaussian quadrature order for integration
    basisOrder   - Order used for basis functions
    simType      - Type of simulation the results are extracted from ("E" or "A" currently supported)
    numVecUn     - Number of vector unknowns / vector basis functions in the simulation
    probType     - Type of problem being solved (currently only "FD" is supported)
    freqs        - List of frequencies the problem is solved at
    dt           - Timestep for time domain simulations (not currently supported)
  Outputs:
    outIntegrals - The integrated quantity for each output surface at each frequency
  """
  # outList, outEdgeList, outFaceList, outTetList = \
  #   [connLists["outList"], connLists["outEdgeList"], connLists["outFaceList"], connLists["outTetList"]]

  tetVolP = connLists["tetVolP"]
  tetStartInds = np.array([tetVolP[f"vol{ii}"]["startInd"] for ii in range(tetVolP["num"])])
  tetStartInds = np.append(tetStartInds, len(connLists["tetList"]))
  
  edgeVals, nodeVals = [x[:numVecUn], x[numVecUn:]]
  outIntegrals = np.zeros((len(freqs), outP["num"]), dtype=np.complex128)
  muInv = np.identity(3) / (4e-7*np.pi)
  ep0 = 1e-9 / (36 * np.pi)
  edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int32)

  # Check for invalid field type combinations and if surface participation is calculated
  invalidSurfaces = []
  surfPartFlag = False
  for ii in range(outP["num"]):
    if outP[f"{ii}"]["fieldType"] == "A" and simType == "E":
      invalidSurfaces.append(ii)
    elif outP[f"{ii}"]["fieldType"] == "surfPart":
      surfPartFlag = True
  
  if len(invalidSurfaces) > 0:
    print(f"Error: Vector potential output (fieldType='A') not supported for E-field formulation (simType='E')")
    print(f"Invalid surfaces: {invalidSurfaces}")
    exit(1)
  
  # If surface participation calculations are included, calculate the total field in the volume
  if surfPartFlag:
    totFieldEnergy = np.zeros((len(freqs),), dtype=np.float64) # Total field magnitude throughout the geometry at each frequency
    for ii, freq in enumerate(freqs):
      # _, totFieldEnergy[ii] = getTotField(connLists, basisMaps, singEleLists, simType, quadOrder, basisOrder, x, probType, "E", "all", 2*np.pi*freq)
      if simType == "E": # TODO: Add calculation for A-phi
        totFieldEnergy[ii] = np.real(x[:,ii].conj().T @ matStruct["K2"] @ x[:,ii]) / 2
    # print(f"Electric field energy integrated over the full geometry: {totFieldEnergy} J")

    singEleCoeffs = singEleLists[f"singTetCoeffs"]
    singCoeffs = np.unique(singEleCoeffs[singEleCoeffs != 0])
    surfIntMats = {}
    for coeff in singCoeffs:
      coeffStr = f"0p{int(np.round(100*coeff))}"
      surfIntMats[coeffStr] = np.load(f"./BasisMats/singFaceInts_tets_{coeffStr}_p={basisOrder}_s={singOrder}_numSplit=6_shiftedGrid.npz")["surfInts"]

  ## Loop through each output surface
  for ii in range(outP["num"]): 
    # startInd = outP[f"{ii}"]["startInd"]
    # if ii == outP["num"] - 1:
    #   endInd = len(outList)
    # else:
    #   endInd = outP[f"{ii+1}"]["startInd"]

    startInd = outP[f"{ii}"]["startInd"] - startTri
    if ii == outP["num"] - 1:
      endInd = len(outList)
    else:
      endInd = outP[f"{ii+1}"]["startInd"] - startTri

    if endInd <= 0:
      continue # If the element list doesn't include the current volume, move to the next
    elif endInd > len(outList):
      endInd = len(outList) # If the volume exceeds the element list, shift the end index

    if startInd < 0:
      startInd = 0 # If the element list starts in the middle of the current volume, shift the start index
    elif startInd > endInd:
      continue # If the list doesn't include this surface, skip it


    fieldType = outP[f"{ii}"]["type"] # Type of field being integrated
    for triNum in range(startInd, endInd): # Loop through triangles forming the surface
      triNodes = outList[triNum]
      triCoords = connLists["nodeList"][triNodes][:,1:]
      triEdges = outEdgeList[triNum]
      triFace = outFaceList[triNum]

      for kk, tetInd in enumerate(outTetList[triNum]): # Loop through tets connected to the tri
        if tetInd == -1: # -1 marks cases where there's only 1 neighboring tet (typically due to being on PEC)
          continue

        tetNodes = connLists["tetList"][tetInd]
        nodeNums = connLists["nodeList"][tetNodes, 0]
        tetCoords = connLists["nodeList"][tetNodes, 1:]
        tetEdges = connLists["tetEdgeList"][tetInd]
        tetFaces = connLists["tetFaceList"][tetInd]
        tetSingEdges = singEleLists["tetSingEdgeList"][tetInd]

        singFlag = singEleLists["singTetMap"][tetInd]
        if singFlag != -1:
          singCoeffs = singEleLists["singTetCoeffs"][singFlag]
        else:
          singCoeffs = []

        if fieldType == "surfPart":
          # If surface participation is being calculated, only count the current tet if 
          # it's in the interface volume
          curVolInd = np.sum(tetInd >= tetStartInds)-1 # Determines the volume that the tet is in
          if "vol" in outP[f"{ii}"]:
            if curVolInd != outP[f"{ii}"]["vol"]:
              continue # Skip tets that aren't in the correct volume
            normDir = np.cross(triCoords[0]-triCoords[1], triCoords[0]-triCoords[2]) 
            normDir /= np.linalg.norm(normDir)

            # Make sure the normal is pointing towards the center of the tet
            tetCenter = np.sum(tetCoords/4, axis=0)
            triCenter = np.sum(triCoords/3, axis=0)
            if normDir @ (tetCenter - triCenter) < 0:
              normDir *= -1
            # botZ = np.sum(normDir*triCoords[0]) # Bottom coordinate in the Z direction
            # topZ = botZ + outP[f"{ii}"]["thickness"] # Top coordinate in the Z direction

          elif "normal" in outP[f"{ii}"]: # If the volume isn't provided directly, use the normal direction
            normDir = np.array(outP[f"{ii}"]["normal"])
            botZ = np.sum(normDir*triCoords[0]) # Bottom coordinate in the Z direction
            topZ = botZ + outP[f"{ii}"]["thickness"] # Top coordinate in the Z direction
            tetCenter = np.sum(tetCoords/4, axis=0)
            if np.sum(normDir*tetCenter) < botZ:
              continue # Skip this tet since it doesn't contain the interface volume

          # Get relevant quantities for using precalculated integrals
          if singFlag != -1:
            gradLoc = np.array(q.gauss_quadrature_tetra_numba_loc(tetCoords, 2))
            gradVals = [basis3Dscalar_numba(tetCoords, ii, gradLoc)[1] for ii in range(3)]
            # Scale the gradients of the barycentric coordinates according to the change in epsilon to enforce boundary conditions
            volEpr = tetVolP[f"vol{curVolInd}"]["epr"]
            eprNorm = normDir.T @ volEpr @ normDir # Epsilon_r in the normal direction
            for ind in range(len(gradVals)):
              gradVals[ind] = eprNorm/outP[f"{ii}"]["epr"]*np.outer(gradVals[ind] @ normDir, normDir) - np.cross(np.cross(gradVals[ind], normDir), normDir)
            gradInds = np.array([[0,0], [1,0], [1,1], [2,0], [2,1], [2,2]])
            gradProducts = np.array([np.sum(gradVals[gradInds[ii,0]] * gradVals[gradInds[ii,1]]) for ii in range(len(gradInds))])

        elif kk == 1: 
          # If this is the second tet, eliminate the degrees of freedom associated 
          # with the shared face to avoid counting them twice
          locTriNodes = np.array([np.where(triNodes[ind] == tetNodes)[0][0] for ind in range(len(triNodes))])
          nodeNums[locTriNodes] = -1
          locEdgeNums = np.array([[locTriNodes[0], locTriNodes[1]], [locTriNodes[0], locTriNodes[2]], [locTriNodes[1], locTriNodes[2]]])
          locTriEdges = np.array([np.where(np.all(locEdgeNums[ind] == edgeNums, axis=1))[0][0] for ind in range(len(triEdges))])
          tetEdges[locTriEdges] = -1
          if triFace != -1:
            locTriFace = np.where(tetFaces == triFace)[0]
            tetFaces[locTriFace] = -1

        if (fieldType == "surfPart") and (singFlag != -1):
          # # For surface participation in singular elements, use a combined 2D-1D integration approach
          # triPoints = q.gauss_quadrature_trisimp_init(9, *triCoords) # 2D quadrature in the XY direction
          # zPoints = q.gauss_quadrature_1D_init(11, 0, outP[f"{ii}"]["thickness"]) # 1D quadrature in the z direction
          # numTriPoints, numZPoints = [len(triPoints), len(zPoints)]
          # loc = np.zeros((0,3), dtype=np.float64)
          # for jj in range(numZPoints):
          #   curTriPoints = np.copy(triPoints)
          #   curTriPoints += normDir*zPoints[jj]
          #   loc = np.append(loc, curTriPoints, axis=0)

          # # Just doing a surface integral instead, with some offset to avoid being directly on the face
          # triCoords_off = triCoords + 1e-3*outP[f"{ii}"]["thickness"]*normDir
          # loc = q.gauss_quadrature_trisimp_init(9, *triCoords_off)

          # # Split the triangle into a set of smaller triangles and integrate over them
          # triCoords_off = triCoords + 1e-3*outP[f"{ii}"]["thickness"]*normDir
          # splitTriCoords, splitTriList = splitTri(triCoords_off, 1)
          # splitLoc = []
          # for kk in range(len(splitTriList)):
          #   splitLoc.append(q.gauss_quadrature_trisimp_init(9, *splitTriCoords[splitTriList[kk]]))
          # loc = np.vstack(splitLoc)

          # Using precalculated integrals
          faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)
          locTriFace = -1 # Determine which face on the tet is the current triangle
          for faceInd in range(len(faceNums)):
            if np.all(tetNodes[faceNums[faceInd]] == triNodes):
              locTriFace = faceInd
              break
          curCoeffs = np.unique(singCoeffs[singCoeffs != 0])
          coeffStr = f"0p{int(np.round(100*curCoeffs[0]))}" # TODO: Add support for tets with multiple coefficients
          curFaceIntMat = surfIntMats[coeffStr][locTriFace]

          # Calculate triangle area
          p1, p2, p3 = triCoords
          triArea = np.sqrt(
            ((p1[1] - p3[1]) * (p2[2] - p3[2]) - (p2[1] - p3[1]) * (p1[2] - p3[2]))** 2
            + ((p1[0] - p3[0]) * (p2[2] - p3[2]) - (p2[0] - p3[0]) * (p1[2] - p3[2]))**2
            + ((p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]))**2
          )
          
          
        else:
          # For nonsingular elements, just use gaussian quadrature over the surface
          loc = np.array(q.gauss_quadrature_trisimp_init(quadOrder, triCoords[0], triCoords[1], triCoords[2]))  

        
        if fieldType == "surfPart": # For surface participation, calculate electric field before getting surface participation
          simFieldType = "E"
        else:
          simFieldType = fieldType
      
        # Add contribution from the current tet to the overall integral
        for jj, freq in enumerate(freqs):
          if fieldType == "E" or fieldType == "H" or fieldType == "A":
            fieldVals = evalField(edgeVals, nodeVals, 2*np.pi*freq, loc, tetCoords, nodeNums, \
                                connLists["tetList"][tetInd], connLists["tetEdgeList"][tetInd], singEleLists["tetSingEdgeList"][tetInd], \
                                connLists["tetFaceList"][tetInd], tetInd, muInv, \
                                singFlag, basisMaps, simType, "Tet", simFieldType, basisOrder, singOrder, N_tet, singCoeffs)
            intVal = fieldVals @ outP[f"{ii}"]["direction"]
            outIntegrals[jj, ii] += q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], intVal)

          elif fieldType == "surfPart":

            if singFlag == -1:
              # For nonsingular elements, just calculate the integral via quadrature
              fieldVals = evalField(edgeVals, nodeVals, 2*np.pi*freq, loc, tetCoords, nodeNums, \
                                connLists["tetList"][tetInd], connLists["tetEdgeList"][tetInd], singEleLists["tetSingEdgeList"][tetInd], \
                                connLists["tetFaceList"][tetInd], tetInd, muInv, \
                                singFlag, basisMaps, simType, "Tet", simFieldType, basisOrder, singOrder, basisOrder_long, singOrder_long, N_tet, singCoeffs)
              
              # Scale the normal part of the field to account for the boundary conditions
              volEpr = tetVolP[f"vol{curVolInd}"]["epr"]
              eprNorm = normDir.T @ volEpr @ normDir # Epsilon_r in the normal direction
              fieldVals = eprNorm/outP[f"{ii}"]["epr"]*np.outer(fieldVals @ normDir, normDir) - np.cross(np.cross(fieldVals, normDir), normDir)
              intVal = np.real(np.sum(np.conj(fieldVals) * fieldVals, axis=1))
              outIntegrals[jj, ii] += outP[f"{ii}"]["thickness"]*ep0*outP[f"{ii}"]["epr"]*q.gauss_quadrature_trisimp(triCoords[0], triCoords[1], triCoords[2], intVal)

            else:
              # # In singular elements, integrate over the volume using combined 2D-1D integrals
              # intVals2D = np.zeros((numZPoints,), dtype=np.float64) # Results of 2D integrals at each height
              # for kk in range(numZPoints):
              #   curTriInds = np.arange(numTriPoints*kk, numTriPoints*(kk+1))
              #   curVal = np.real(np.sum((np.conj(fieldVals[curTriInds]) * ep0*outP[f"{ii}"]["epr"]) * fieldVals[curTriInds], axis=1))
              #   intVals2D[kk] = q.gauss_quadrature_trisimp(*triCoords, curVal)
              # # Integrate over each of the 2D integrals in 1D
              # intVal3D = q.gauss_quadrature_1D(0, outP[f"{ii}"]["thickness"], intVals2D)

              # # Alt approach of just doing a surface integral
              # intVal = np.real(np.sum(np.conj(fieldVals) * fieldVals, axis=1))
              # intVal3D = outP[f"{ii}"]["thickness"]*ep0*outP[f"{ii}"]["epr"]*q.gauss_quadrature_trisimp(*triCoords, intVal)

              # # Alt approach of doing surface integrals over a set of split triangles
              # intVal3D = 0
              # numTriLoc = len(splitLoc[0]) # Number of points in the integral for each triangle
              # for kk in range(len(splitTriList)):
              #   curInds = kk*numTriLoc + np.arange(numTriLoc)
              #   intVal = np.real(np.sum(np.conj(fieldVals[curInds]) * fieldVals[curInds], axis=1))
              #   intVal3D += outP[f"{ii}"]["thickness"]*ep0*outP[f"{ii}"]["epr"]*q.gauss_quadrature_trisimp(*splitTriCoords[splitTriList[kk]], intVal)

              # Alt approach using saved integral values
              intVal3D = 0
              vecInds, _ = selectInds3D(basisOrder, singOrder, basisOrder_long, singOrder_long, "Tet", basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"], basisMaps["vecEdgeMap"], \
                                        basisMaps["vecSingEdgeMap"], basisMaps["vecFaceMap"], basisMaps["vecVolMap"], tetNodes, tetEdges, tetSingEdges, \
                                        tetFaces, tetInd, nodeNums, edgeNums, 0, 0, N_tet, singFlag, singCoeffs, simType)
              
              start_time = time.time()
              for mm, ind_m in enumerate(vecInds):
                if ind_m <= -1:
                  continue
                for nn, ind_n in enumerate(vecInds):
                  if nn <= -1:
                    continue

                  intVal3D += outP[f"{ii}"]["thickness"]*ep0*outP[f"{ii}"]["epr"]\
                              *triArea*np.sum(gradProducts*curFaceIntMat[:,mm,nn])*x[ind_m,jj]*np.conj(x[ind_n,jj])
              
              if np.isnan(intVal3D):
                singEdgeNums = singEleLists["tetSingEdgeList"][tetInd]
                print(f"Integral produces NaN for tet {tetInd} for integration surface {ii}. Singular coefficients: {singCoeffs}. Singular edge numbers: {singEdgeNums}.")
              outIntegrals[jj, ii] += intVal3D              
          

    if outP[f"{ii}"]["fieldType"] == "surfPart": 
      for jj in range(len(freqs)): # For surface participation, add scaling factors
        outIntegrals[jj,ii] /= (2*totFieldEnergy[jj])

  return outIntegrals

def getSurfaceCurrents(x, freqs, curVecVals, curVecInds, simType):
  """
  Calculates the surface current flowing along a set of surfaces
  """
  I_surfs = np.zeros((len(freqs), len(curVecVals)), dtype=np.complex128)

  for ii in range(len(freqs)):
    for surfNum in range(len(curVecVals)):
      x_surf = np.copy(x[curVecInds[surfNum],ii]) # Isolate the solution at the surface
      if simType == "E" or simType == "TCS": # Need to scale the field for E or TCS
        x_surf /= -1j*2*np.pi*freqs[ii]

      I_surfs[ii,surfNum] = np.sum(curVecVals[surfNum]*x_surf)

      
  # # Eliminate dirichlet boundaries so surface currents can be calculated
  # matStruct["knownVals"] = []
  # matStruct["knownInds"] = []
  # for ii in range(len(freqs)):
  #   # Form full mats with dirichlet boundaries not included
  #   A, b, M_diag, _, _ = buildSysFD(matStruct, [freqs[ii]], simType)
  #   A = A[0]
  #   b = b[0]
  #   M_diag = M_diag[0]
  #   M_diag_inv = spa.diags_array(1/M_diag.diagonal(), format='csr')
  #   curDen = M_diag_inv @ (A @ (M_diag_inv @ x[:,ii]) - b) # Current density for each unknown
  #   for surfNum in range(len(voltVecVals)):
  #     # if simType == "A": # Need to scale the magnetic vector potential for A-phi
  #     #   vecInds = np.where(voltVecInds[surfNum] < numVecUn)[0]
  #     #   nodeInds = np.where(voltVecInds[surfNum] >= numVecUn)[0]
  #     #   x_port[vecInds] *= 1j*2*np.pi*freqs[ii]
  #     #   x_port[nodeInds] *= -1
    
  #     I_ports[ii,surfNum] = np.sum(voltVecVals[surfNum]*curDen[voltVecInds[surfNum]])/(1j*2*np.pi*freqs[ii])

  return I_surfs


def getTotField(connLists, basisMaps, singEleLists, simType, probP, N_tet, totVals, probType, fieldType, eleToSave, omega=-1, dt=-1):
  """
  Calculates the total electric field for a geometry across different element types

  Inputs:
    connLists  - Contains connectivity between different geometric entities
    See getEleField for remaining input information

  Outputs:
    fieldVals    - Field vector at the center of each element (inclusive of all element types)
    normFactor   - Integral of the electric field magnitude over the whole geometry (for normalizing eigenvectors)
  """

  fieldVals = np.zeros((0,3), dtype=totVals.dtype)
  normFactor = 0
  for _, eleType in enumerate(["Tet", "Wedge"]):
    eleTypeLower = eleType.casefold()
    if len(connLists[f"{eleTypeLower}List"]) != 0:
      curFieldVals, curNormFactor = getEleField(connLists[f"{eleTypeLower}List"], connLists[f"{eleTypeLower}EdgeList"], singEleLists[f"{eleTypeLower}SingEdgeList"], connLists[f"{eleTypeLower}FaceList"], connLists[f"{eleTypeLower}VolP"], \
                                                connLists["nodeList"], basisMaps, singEleLists, simType, eleType, fieldType, probP, N_tet, totVals, probType, eleToSave, omega, dt)
      fieldVals = np.append(fieldVals, curFieldVals, axis=0)
      normFactor += curNormFactor
  
  return fieldVals, normFactor

def getEleField(eleList, eleEdgeList, eleSingEdgeList, eleFaceList, volP, nodeList, basisMaps, singEleLists, simType, \
                eleType, fieldType, probP, N_tet, totVals, probType, eleToSave, omega=-1, dt=-1):
  """
  Calculates the electric field for a single element type from the FEM output

  Inputs:
    eleList         - List of node numbers forming each element
    eleEdgeList     - List of edge numbers forming each element
    eleSingEdgeList - List of singular edge numbers forming each element
    eleFaceList     - List of face numbers forming each element
    volP            - Material parameters of each volume
    nodeList        - List of global numbers for each node and the corresponding coordinates
    basisMaps       - Mappings between geometric entities and basis functions
    singEleLists    - Dictionary containing information about singular elements
    simType         - The type of solver used (E, A, TCS)
    eleType         - The type of elements being used (currently either "Tet" or "Wedge")
    fieldType       - The type of field to evaluate (either "E" or "H")
    probP           - Dictionary containing problem info (such as basis function orders)
    N_tet           - Number of tetrahedra in the mesh
    totVals         - The overall solution vector (including edge and nodal unknowns)
    probType        - The type of problem being solved (FD, TD)
    omega           - Angular frequency the solve was performed at (for FD)
    dt              - The timestep size for TD simulations

  Outputs:
    fieldVals    - Field vector at the center of each element
    normFactor   - Integral of the electric field magnitude over the whole geometry (for normalizing eigenvectors)

  """

  if simType != "A" and (fieldType == "A" or fieldType == "Phi"):
    print(f"Error: The requested plot is for {fieldType} but the potential-based solver is not being used.")
    sys.exit()

  vecEdgeMap = basisMaps["vecEdgeMap"]
  vecFaceMap = basisMaps["vecFaceMap"]
  vecVolMap = basisMaps["vecVolMap"]
  vecSingEdgeMap = basisMaps["vecSingEdgeMap"]
  scalNodeMap = basisMaps["scalNodeMap"]
  scalEdgeMap = basisMaps["scalEdgeMap"]
  mu0 = 4e-7*np.pi

  quadOrder = probP["3DIntOrder"]
  basisOrder = probP["bOrder"]
  singOrder = probP["sOrder"]
  basisOrder_long = probP["bOrder_long"]
  singOrder_long = probP["sOrder_long"]

  if fieldType == "E" or fieldType == "A" or fieldType == "H":
    fieldVals = np.zeros((len(eleList),3), dtype=totVals.dtype)
  elif fieldType == "Phi":
    fieldVals = np.zeros((len(eleList),), dtype=totVals.dtype)
  else:
    print("Error: Invalid field type specified for plotting.")
    sys.exit()
  normFactor = 0

  if eleType == "Tet":
    edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    centerLoc = 0 # Marks the integration point at the center of the tet
    if basisOrder == 1:
      numVecBasis = 6
      numScalBasis = 4

    elif basisOrder == 2:
      numVecBasis = 20
      numScalBasis = 10
    
  elif eleType == "Wedge":
    edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]])
    if basisOrder == 1:
      numVecBasis = 9
      numScalBasis = 6
    elif basisOrder == 2:
      numVecBasis = 22
      numScalBasis = 12

  numVecUn, _ = getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap)

  ## Create mappings for singular elements
  singEles = singEleLists[f"sing{eleType}s"]
  singEleMap = singEleLists[f"sing{eleType}Map"]
  singEleCoeffs = singEleLists[f"sing{eleType}Coeffs"]

  # Determine which unknowns are used based on problem and simulation types
  if probType == "FD":
    if simType == "E" or simType == "TCS":
      edgeVals = totVals[:numVecUn]
      if simType == "TCS": 
        nodeVals = totVals[numVecUn:]
    elif simType == "A":
      edgeVals = 1j*omega*totVals[:numVecUn]
      nodeVals = -totVals[numVecUn:]
  elif probType == "TD":
    if simType == "E" or simType == "TCS":
      edgeVals = totVals[:numVecUn, 1]
      if simType == "TCS":
        nodeVals = totVals[numVecUn:, 1]
    elif simType == "A":
      edgeVals = -(totVals[:numVecUn, 2] - totVals[:numVecUn, 0])/(2*dt)
      nodeVals = totVals[numVecUn:, 1]

  for vol in range(volP["num"]): 
    volStartInd = volP[f"vol{vol}"]["startInd"]
    if vol == volP["num"] - 1:
      volEndInd = len(eleList)
    else:
      volEndInd = volP[f"vol{vol+1}"]["startInd"]

    muInv = volP[f"vol{vol}"]["murInv"]/mu0

    for eleInd in range(volStartInd, volEndInd):
      ## Manage where the current element should be saved (if at all)
      if eleToSave == "all":
        saveInd = eleInd
      else:
        saveInd = np.where(eleToSave == eleInd)[0]
        if len(saveInd) == 0:
          continue

      # Get coordinates and global node numbers for the element
      coords = nodeList[eleList[eleInd]][:,1:]
      nodeNums = nodeList[eleList[eleInd], 0] # Also get global node numbers (including PEC)

      # Check whether the current element is singular or not and get indices for the rows and columns
      if singEleMap[eleInd] != -1:
        singCoeffs = singEleCoeffs[singEleMap[eleInd]]
      else:
        singCoeffs = []
      vecInds, scalInds = selectInds3D(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, \
                                       scalNodeMap, scalEdgeMap, vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, \
                                       eleList[eleInd], eleEdgeList[eleInd], eleSingEdgeList[eleInd], eleFaceList[eleInd], eleInd, \
                                       nodeNums, edgeNums, numVecBasis, numScalBasis, N_tet, singEleMap[eleInd], singCoeffs, simType)
      numVecInds, numScalInds = [len(vecInds), len(scalInds)]

      ## Set up integration points
      # Evaluate basis functions at each integration point
      if eleType == "Tet":
        if singEleMap[eleInd] != -1:
          loc = np.array(q.gauss_quadrature_tetra_numba_loc(coords, 8)) # Determine integration points in the tetrahedron
        else:
          loc = np.array(q.gauss_quadrature_tetra_numba_loc(coords, quadOrder)) # Determine integration points in the tetrahedron
        
        vecBasis, curlBasis, scalBasis, gradBasis = basisSetupTet(coords, loc, simType, basisOrder, edgeNums, singFlag=singEleMap[eleInd], singCoeffs=singCoeffs, singOrder=singOrder)
        curEleVals = np.zeros((len(loc), 3), dtype=totVals.dtype)
        centerLoc = np.array([np.sum(coords, axis=0)/4])
        vecCenterBasis, curlCenterBasis, scalCenterBasis, gradCenterBasis = basisSetupTet(coords, centerLoc, simType, basisOrder, edgeNums, singFlag=singEleMap[eleInd], singCoeffs=singCoeffs, singOrder=singOrder)
      elif eleType == "Wedge":
        if singEleMap[eleInd] != -1:
          vertDir, triLoc, vertLoc = getWedgeLocs(coords, 9) # Integration points in each direction
        else:
          vertDir, triLoc, vertLoc = getWedgeLocs(coords, quadOrder) # Integration points in each direction
        vecBasis, curlBasis, scalBasis, gradBasis \
          = basisSetupWedge(coords, eleList[eleInd], eleEdgeList[eleInd], triLoc, vertLoc, vertDir, simType, basisOrder, basisOrder_long, \
                            singFlag=singEleMap[eleInd], singCoeffs=singCoeffs, singOrder=singOrder, singOrder_long=singOrder_long)
        curEleVals = np.zeros((len(vertLoc)*len(triLoc), 3), dtype=totVals.dtype)
        centerTriLoc = np.array([np.sum(coords[:3], axis=0)/3])
        centerVertLoc = np.array([(coords[0,vertDir] + coords[3,vertDir])/2])
        vecCenterBasis, curlCenterBasis, scalCenterBasis, gradCenterBasis \
          = basisSetupWedge(coords, eleList[eleInd], eleEdgeList[eleInd], centerTriLoc, centerVertLoc, vertDir, simType, basisOrder, basisOrder_long, \
                            singFlag=singEleMap[eleInd], singCoeffs=singCoeffs, singOrder=singOrder, singOrder_long=singOrder_long)
        
      
      ## Evaluate fields at the center point and save the total field in the element
      for ii in range(numVecInds):
        curEdgeNum = vecInds[ii]
        if curEdgeNum != -1:
          curEleVals += edgeVals[curEdgeNum] * vecBasis[ii]
          if fieldType == "E":
            fieldVals[eleInd,:] += edgeVals[curEdgeNum] * (vecCenterBasis[ii][0])
          elif fieldType == "H":
            fieldVals[eleInd,:] += edgeVals[curEdgeNum] * (muInv @ curlCenterBasis[ii][0]) / (1j*omega)
          elif fieldType == "A":
            fieldVals[eleInd,:] += edgeVals[curEdgeNum] * (vecCenterBasis[ii][0]) / (1j*omega)

      if (fieldType == "E" or fieldType == "Phi") and (simType == "A" or simType == "TCS"):
        for ii in range(numScalInds):
          curNode = scalInds[ii]
          if curNode == -1:
            continue
          
          if fieldType == "E":
            if eleType == "Tet" and basisOrder == 1: # The gradient is constant in this case, so it's only specified for one position
              curEleVals += np.array(list(repeat(nodeVals[curNode]*gradBasis[ii], len(loc))))
            else:
              curEleVals += nodeVals[curNode]*gradBasis[ii]
            fieldVals[eleInd,:] += nodeVals[curNode]*gradCenterBasis[ii][0]
          elif fieldType == "Phi":
            fieldVals[eleInd] += nodeVals[curNode]*scalCenterBasis[ii][0]

          
      if eleType == "Tet":
        normFactor += q.gauss_quadrature_tetra(coords, quadOrder, np.real(np.sum((np.conj(curEleVals) @ volP[f"vol{vol}"]["epr"]) * curEleVals, axis=1)))
      elif eleType == "Wedge":
        normFactor += q.gauss_quadrature_wedge(coords, vertDir, vertLoc, np.real(np.sum((np.conj(curEleVals) @ volP[f"vol{vol}"]["epr"]) * curEleVals, axis=1)))

  return fieldVals, normFactor

def getPointFields(fieldPoints, connLists, basisMaps, singEleLists, simType, probP, totVals, omega, numVecUn, probType, fieldType):
  """
  Calculates the desired field / potential at a set of provided points throughout the mesh

  Inputs:
    connLists  - Contains connectivity between different geometric entities
    See getTotField/getEleField for remaining input information

  Outputs:
    pointFields - Value of the desired field / potential at each point (len(fieldPoints) x 3)
  """

  # Orders of each type of basis functions
  basisOrder = probP["bOrder"]
  singOrder = probP["sOrder"]
  basisOrder_long = probP["bOrder_long"]
  singOrder_long = probP["sOrder_long"]

  pointFields = np.zeros((len(fieldPoints), 3), totVals.dtype)
  pointEles = np.zeros((len(fieldPoints,)), dtype=np.int32)
  nodeCoords = connLists["nodeList"][:,1:] # Coordinates of every node in the mesh
  edgeVals, nodeVals = [totVals[:numVecUn], totVals[numVecUn:]]
  tetVolStartInds = np.array([connLists["tetVolP"][f"vol{vol}"]["startInd"] for vol in range(connLists["tetVolP"]["num"])])
  wedgeVolStartInds = np.array([connLists["wedgeVolP"][f"vol{vol}"]["startInd"] for vol in range(connLists["wedgeVolP"]["num"])])
  N_tet = len(connLists["tetList"])
  mu0 = 4e-7*np.pi
  for ii, point in enumerate(fieldPoints):
    # First isolate which element the point is in
    pointDist = np.linalg.norm(nodeCoords - point, axis=1) # Calculate the distance from each node to the point
    nearNodes = np.argsort(pointDist) # Sort the nodes closest to the point to find the tet it's in
    curEle = -1
    for nearNode in nearNodes:
      nodeTets = np.where(connLists["tetList"] == nearNode)[0]
      for tet in nodeTets: # Check whether each tet on the node contains the point
        eleNodes = connLists["tetList"][tet]
        scalBasis = np.array([basis3Dscalar_numba(nodeCoords[eleNodes], ii, point.reshape(1,3))[0] for ii in range(4)])
        if np.all(scalBasis >= 0.0) and np.all(scalBasis <= 1.0):
          curEle = tet # If true, the point is in this tet
          eleType = "Tet"
          break

      if curEle != -1: # Only move to the next nearest node if the tet or wedge has not been found
        break

      nodeWedges = np.where(connLists["wedgeList"] == nearNode)[0]
      for wedge in nodeWedges:
        eleNodes = connLists["wedgeList"][wedge]
        # Check whether the point is within the triangle of the wedge
        scalTriBasis = np.array([basis2Dscalar_numba(nodeCoords[eleNodes[:3]], ii, point.reshape(1,3))[0] for ii in range(3)])
        if np.all(scalTriBasis >= 0.0) and np.all(scalTriBasis <= 1.0):
          # Now check whether it's vertically within the wedge
          for testDir in [0, 1, 2]: # Find the vertical direction
            if np.isclose(nodeCoords[eleNodes[0],testDir], nodeCoords[eleNodes[1],testDir], atol=1e-8) \
              and np.isclose(nodeCoords[eleNodes[0],testDir], nodeCoords[eleNodes[2],testDir], atol=1e-8):
              vertDir = testDir

          if (point[vertDir] > nodeCoords[eleNodes[0], vertDir] and point[vertDir] < nodeCoords[eleNodes[3], vertDir]) \
              or (point[vertDir] < nodeCoords[eleNodes[0], vertDir] and point[vertDir] > nodeCoords[eleNodes[3], vertDir]):
            # If the point is between the top and bottom faces, it's in this wedge
            curEle = wedge
            eleType = "Wedge"
            break

      if curEle != -1: # Only move to the next nearest node if the tet or wedge has not been found
        break

    if curEle == -1:
      print(f"Valid element not found for point {ii} at position {point}.")
      pointFields[ii] = np.nan
      continue

    # Now evaluate the field at the point using the basis functions for the element
    nodeNums = connLists["nodeList"][:,0]
    if eleType == "Tet":
      pointEles[ii] = curEle
      vol = np.where(tetVolStartInds - curEle <= 0)[0][-1]
    elif eleType == "Wedge":
      pointEles[ii] = curEle + N_tet
      vol = np.where(wedgeVolStartInds - (curEle+N_tet) <= 0)[0][-1]
    singFlag = singEleLists[f"sing{eleType}Map"][curEle]
    if singFlag != -1:
      singCoeffs = singEleLists[f"sing{eleType}Coeffs"][singFlag]
    else:
      singCoeffs = []
    
    eleTypeLower = eleType.lower()
    pointFields[ii] = evalField(edgeVals, nodeVals, omega, point.reshape(1,3), nodeCoords[eleNodes], nodeNums, \
                                connLists[f"{eleTypeLower}List"][curEle], connLists[f"{eleTypeLower}EdgeList"][curEle], singEleLists[f"{eleTypeLower}SingEdgeList"][curEle], \
                                connLists[f"{eleTypeLower}FaceList"][curEle], curEle, connLists[f"{eleTypeLower}VolP"][f"vol{vol}"]["murInv"]/mu0, \
                                singFlag, basisMaps, simType, eleType, fieldType, basisOrder, singOrder, basisOrder_long, singOrder_long, N_tet, singCoeffs)
    if np.any(np.isnan(pointFields)):
      if eleType == "Tet":
        print(f"NaN for point {ii} at position {point} in tet {curEle}. Scalar functions: {scalBasis}.")
      elif eleType == "Wedge":
        print(f"NaN for point {ii} at position {point} in wedge {curEle}. Scalar functions: {scalTriBasis}.")
    
  return pointFields, pointEles


def evalField(edgeVals, nodeVals, omega, loc, coords, nodeNums, eleNodes, eleEdges, eleSingEdges, \
              eleFaces, eleInd, muInv, singFlag, \
              basisMaps, simType, eleType, fieldType, basisOrder, singOrder, basisOrder_long, singOrder_long, N_tet, singCoeffs):
  """
  Evaluates the field at a set of points in the element

  """

  vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap = [basisMaps["vecEdgeMap"], basisMaps["vecSingEdgeMap"], basisMaps["vecFaceMap"], basisMaps["vecVolMap"]]
  scalNodeMap, scalEdgeMap = [basisMaps["scalNodeMap"], basisMaps["scalEdgeMap"]]

  if eleType == "Tet":
    edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    if basisOrder == 1:
      numVecBasis = 6
      numScalBasis = 4
    elif basisOrder == 2:
      numVecBasis = 20
      numScalBasis = 10

  elif eleType == "Wedge":
    edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]])
    if basisOrder == 1:
      numVecBasis = 9
      numScalBasis = 6
    elif basisOrder == 2:
      numVecBasis = 22
      numScalBasis = 12

  # Check whether the current element is singular or not and get indices for the rows and columns
  vecInds, scalInds = selectInds3D(basisOrder, singOrder, basisOrder_long, singOrder_long, eleType, scalNodeMap, scalEdgeMap, vecEdgeMap, vecSingEdgeMap, vecFaceMap, vecVolMap, eleNodes, eleEdges, eleSingEdges, eleFaces, eleInd, nodeNums, edgeNums, \
                                    numVecBasis, numScalBasis, N_tet, singFlag, singCoeffs, simType)
  numVecInds, numScalInds = [len(vecInds), len(scalInds)]

  ## Set up integration points
  # Evaluate basis functions at each integration point
  if eleType == "Tet":
    vecBasis, curlBasis, scalBasis, gradBasis = basisSetupTet(coords, loc, simType, basisOrder, edgeNums, singFlag=singFlag, singCoeffs=singCoeffs, singOrder=singOrder)
  elif eleType == "Wedge": 
    for testDir in [0, 1, 2]: # Find the vertical direction
      if np.isclose(coords[0,testDir], coords[1,testDir], atol=1e-8) \
        and np.isclose(coords[0,testDir], coords[2,testDir], atol=1e-8):
        vertDir = testDir
    vertLoc = [loc[0,vertDir]]
    vecBasis, curlBasis, scalBasis, gradBasis = basisSetupWedge(coords, eleNodes, eleEdges, loc, vertLoc, vertDir, simType, basisOrder, basisOrder_long, \
                                                                singFlag=singFlag, singCoeffs=singCoeffs, singOrder=singOrder, singOrder_long=singOrder_long)
  fieldVals = np.zeros((len(loc), 3), dtype=edgeVals.dtype)
    
  
  ## Evaluate fields at each location and save the total field in the element
  for ii in range(numVecInds):
    curEdgeNum = vecInds[ii]
    if curEdgeNum != -1:
      if fieldType == "E":
        fieldVals += edgeVals[curEdgeNum] * (vecBasis[ii])
      elif fieldType == "H":
        fieldVals += edgeVals[curEdgeNum] * (curlBasis[ii] @ muInv) / (1j*omega)
      elif fieldType == "A":
        fieldVals += edgeVals[curEdgeNum] * (vecBasis[ii]) / (1j*omega)

  if (fieldType == "E" or fieldType == "Phi") and (simType == "A" or simType == "TCS"):
    for ii in range(numScalInds):
      curNode = scalInds[ii]
      if curNode == -1:
        continue
      
      if fieldType == "E":
        if eleType == "Tet" and basisOrder == 1: # The gradient is constant in this case, so it's only specified for one position
          fieldVals += nodeVals[curNode]*gradBasis[ii]
        else:
          fieldVals += nodeVals[curNode]*gradBasis[ii]
      elif fieldType == "Phi":
        fieldVals += nodeVals[curNode]*scalBasis[ii]

  return fieldVals


"""
Calculates the electric field throughout the geometry at a given point in time from FETD solution

Inputs:
  eleList      - List of node numbers forming each tetrahedron, alongside material properties in the tet
  eleEdgeList  - List of edge numbers forming each tetrahedron
  eleToSave    - List of elements to ave the field for, or "all" to save all elements
  nodeList     - List of global numbers for each node and the corresponding coordinates
  simType      - The type of solver used (E, A, TCS)
  quadOrder - Integration order to use for 3D gaussian quadrature
  dt           - Timestep length
  solVals      - Solution vector at the 3 most recent timesteps
  cumSolVals   - Accumulator for prior solution values (only for A-phi)
  N_e          - Number of edge unknowns

Outputs:
  fieldVals    - Field vector at the center of each tetrahedron

"""
def getField_TD(eleList, eleEdgeList, eleToSave, nodeList, simType, quadOrder, dt, solVals, N_e):
  if eleToSave == "all":
    fieldVals = np.zeros((len(eleList),3), dtype=solVals.dtype)
  else:
    fieldVals = np.zeros((len(eleToSave),3), dtype=solVals.dtype)
  N_n = len(solVals[:,0]) - N_e

  edgeNums = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
  for eleInd in range(len(eleList)):
    if eleToSave == "all":
      saveInd = eleInd
    else:
      saveInd = np.where(eleToSave == eleInd)[0]
      if len(saveInd) == 0:
        continue

    coords = nodeList[eleList[eleInd, :4].astype(np.int32)][:,1:] # Coordinates for the 4 nodes of the tet
    loc = np.array(q.gauss_quadrature_tetra(coords, quadOrder)) # Determine integration points in the tetrahedron
    vecBasis = [basis_vector_numba(coords, edgeNums[ii, 0], edgeNums[ii, 1], loc)[0] for ii in range(6)]
    for edge in range(6):
      edgeNum = eleEdgeList[eleInd,edge]
      if edgeNum >= 0:
        if simType == "E" or simType == "TCS":
          fieldVals[saveInd,:] += solVals[edgeNum,1] * (vecBasis[edge][0])
        elif simType == "A":
          fieldVals[saveInd,:] -= (solVals[edgeNum,2] - solVals[edgeNum,0])/(2*dt) * (vecBasis[edge][0])

    if simType == "A" or simType == "TCS":
      gradBasis = [basis3Dscalar_numba(coords, ii, loc)[1] for ii in range(4)]
      for node in range(4):
        nodeNum = nodeList[eleList[eleInd,node].astype(np.int32),0].astype(np.int32)
        if nodeNum == -1:
          continue
        elif nodeNum < -1:
          nodeNum = N_n + nodeNum + 1
        
        nodeLoc = N_e + nodeNum
        fieldVals[saveInd,:] += solVals[nodeLoc,1]*gradBasis[node]

  return fieldVals


"""
Computes S-parameters for each port from the voltages along those ports

Inputs:
  portP      - Dictionary containing information about each port
  V_ports    - Voltage across each port at each frequency

Outputs:
  sParams    - S-parameters for each port at each frequency
"""
def getSParams(portP, V_ports):
  numFreqs = int(np.size(V_ports) / portP["num"])
  sParams_tot = np.zeros((numFreqs, portP["num"]), dtype=np.complex128)

  for ii in range(numFreqs):
    if portP["0"]["portType"].casefold() == "open":
      ## Calculate S-parameters from Z-parameters (for short/open terminations) 
      ## TODO: Generalize for an arbitrary number of ports
      zParams = np.concatenate((V_ports[ii].reshape((2,1)), V_ports[ii][::-1].reshape((2,1))), axis=1) # Only valid when the device is reciprocal
      sParams = np.zeros((2,), dtype=np.complex128)
      deltaZ = (zParams[0,0]+portP["0"]["Z"])*(zParams[1,1]+portP["0"]["Z"]) - zParams[0,1]*zParams[1,0]
      sParams[0] = ((zParams[0,0]-portP["0"]["Z"])*(zParams[1,1]+portP["0"]["Z"]) - zParams[0,1]*zParams[1,0]) / deltaZ
      # sParams[0,1] = 2*portP["0"]["Z"]*zParams[0,1] / deltaZ
      sParams[1] = 2*portP["0"]["Z"]*zParams[1,0] / deltaZ
      # sParams[1,1] = ((zParams[0,0]+portP["0"]["Z"])*(zParams[1,1]-portP["0"]["Z"]) - zParams[0,1]*zParams[1,0]) / deltaZ
    else:
      ## Calculate S-parameters directly from voltages (for impedance terminations)
      sParams = np.zeros((portP["num"],), dtype=np.complex128)
      # curFreqV = np.copy(V_ports[ii])
      # for jj in range(portP["num"]):
      #   totZ = np.sum([portP[f"port{kk}"]["Z"] for kk in range(portP["num"])])
      #   otherZ = totZ - portP[f"port{jj}"]["Z"]
      #   curFreqV[jj] -= portP[f"port{jj}"]["Vsmag"]*otherZ/totZ # Remove the contribution of the incident wave
      #   if portP[f"port{jj}"]["Vsmag"] != 0:
      #     sParams = curFreqV / (portP[f"port{jj}"]["Vsmag"]*otherZ/totZ)

      # Isolate the driving port
      for jj in range(portP["num"]):
        if portP[f"{jj}"]["Vsmag"] != 0:
          Vinc = portP[f"{jj}"]["Vsmag"]/2
          break
      for jj in range(portP["num"]):
        if portP[f"{jj}"]["Vsmag"] != 0:
          sParams[jj] = V_ports[ii,jj] / Vinc - 1
        else:
          sParams[jj] = V_ports[ii,jj] / Vinc
    
    sParams_tot[ii] = sParams

  return sParams_tot

def zMatToS(zMat):
  """
  Function for converting from Z-parameters into S-parameters

  Inputs:
    zMat - Z matrix containing impedance parameters for the device

  Outupts:
    sMat - S matrix containing scattering parameters for the device
  """

  # Assume each port has an impedance of 50 ohms
  sqrtY = np.zeros(zMat.shape, dtype=np.float64)
  for ii in range(zMat.shape[0]):
    sqrtY[ii,ii] = np.sqrt(1/50)

  scaledZ = sqrtY @ zMat @ sqrtY
  eyeMat = np.identity(zMat.shape[0])
  sMat = (scaledZ - eyeMat) @ np.linalg.inv(scaledZ + eyeMat)

  return sMat

