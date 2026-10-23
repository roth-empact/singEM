import numpy as np
import scipy.sparse as spa
import time
import sys
# Parallelization modeles
from multiprocessing import Pool
from itertools import repeat 
from assemble_mats import assembleMats, assemblePortMats, assemblePortMats_vols, \
    assembleImpSurfMats, getPortInts, getSurfCurrentInts, getUnknownCount, assignVoltageFields, sortSingNodes



# Function to form matrices for E or A-phi throughout a geometry with ports
# This version parallelizes the process to solve simultaneously over frequencies

# Inputs:
# sp - dictionary of simulation parameters
# portParams - dictionary of port parameters
def meshToMats(connLists, subDLists, singEleLists, basisOrder, singOrder, basisOrder_long, singOrder_long, singEdgeFlag, quadOrder3D, quadOrder2D, simType, outBC, meshStats, numProcs): 
  """
  A function to process mesh information into FEM matrices

  Inputs:
    connLists       - Contains connectivity information on mesh entities
    subDLists       - Dictionary containing subdomain information. Only non-empty when domain decomposition is used
    singEleLists    - Dictionary containing singular element information
    basisOrder      - Order of the standard basis functions
    singOrder       - Order of the singular basis functions
    basisOrder_long - Order of the standard basis functions in the longitudinal directions. 
                      Only valid for wedge/prism elements.
    singOrder_long  - Order of the standard basis functions in the longitudinal directions. 
                      Only valid for wedge/prism elements.
    singEdgeFlag    - Boolean flag indicating whether to include edge-singular functions
    quadOrder3D     - Gaussian quadrature order for 3D integrals
    quadOrder2D     - Gaussian quadrature order for 2D integrals
    simType         - Type of simulation to run ("E" or "A")
    outBC           - Boundary condition applied to the outer boundary ("PEC" or "ABC")
    meshStats       - Dictionary of statistics about the numbering of each mesh entity
    numProcs        - Number of processors available to use
  Outputs:
    matStruct       - Dictionary containing the matrices and driving vector
    basisMaps       - Dictionary containing mappings between geometric entities and 
  """
  tetList = connLists["tetList"]
  tetEdgeList = connLists["tetEdgeList"]
  tetSingEdgeList = singEleLists["tetSingEdgeList"]
  tetFaceList = connLists["tetFaceList"]
  tetVolP = connLists["tetVolP"]
  wedgeList = connLists["wedgeList"]
  wedgeEdgeList = connLists["wedgeEdgeList"]
  wedgeSingEdgeList = singEleLists["wedgeSingEdgeList"]
  wedgeFaceList = connLists["wedgeFaceList"]
  wedgeVolP = connLists["wedgeVolP"]
  portList = connLists["portList"]
  portEdgeList = connLists["portEdgeList"]
  portFaceList = connLists["portFaceList"]
  portP = connLists["portP"]
  qubitList = connLists["qubitList"]
  qubitEdgeList = connLists["qubitEdgeList"]
  qubitFaceList = connLists["qubitFaceList"]
  qubitP = connLists["qubitP"]
  nodeList = connLists["nodeList"]
  abcList = connLists["abcList"]
  singTets = singEleLists["singTets"]
  singWedges = singEleLists["singWedges"]

  ## Form mappings from geometric entities to basis functions
  start_time = time.time()  
  if subDLists == []: # DD is not being used
    if len(singTets) != 0 or len(singWedges) != 0: # Singular edges are present
      vecEdgeMap, vecFaceMap, vecVolMap, vecSingEdgeMap, scalNodeMap, scalEdgeMap, unknownCnt = \
        buildBasisMapsSingular_orderN(singEleLists, tetList, tetEdgeList, tetFaceList, wedgeList, wedgeEdgeList, wedgeFaceList, \
                                      meshStats, simType, basisOrder, singOrder, basisOrder_long, singOrder_long, singEdgeFlag)
    else:
      vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap, unknownCnt = \
        buildBasisMaps(nodeList, wedgeEdgeList, meshStats, simType, basisOrder, basisOrder_long)
      vecSingEdgeMap = np.zeros((0,0), dtype=np.int32)
  else:
    raise ValueError("Domain decomposition is not supported in the current version of this package.")

  print(f"Time spent building basis maps: {(time.time() - start_time):.3f} s. Element order: {basisOrder}. " \
        + f"Singular element order: {singOrder} Total number of unknowns: {unknownCnt}")
  basisMaps = {"vecEdgeMap": vecEdgeMap, "vecFaceMap": vecFaceMap, "vecVolMap": vecVolMap, "vecSingEdgeMap": vecSingEdgeMap, \
               "scalNodeMap": scalNodeMap, "scalEdgeMap": scalEdgeMap}

  start_time = time.time()  
  print("Beginning matrix assembly")
  
  ## Construct the matrices from the discretized volume
  if len(tetList) != 0 and len(wedgeList) != 0: # Mesh contains tetrahedrons and wedges
    matStruct = buildVolMats(tetList, tetEdgeList, tetSingEdgeList, tetFaceList, singEleLists, tetVolP, nodeList, abcList, basisMaps, \
                             basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder3D, simType, "Tet", outBC, meshStats, numProcs)
    matStructWedges = buildVolMats(wedgeList, wedgeEdgeList, wedgeSingEdgeList, wedgeFaceList, singEleLists, wedgeVolP, nodeList, abcList, basisMaps, \
                                   basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder3D, simType, "Wedge", outBC, meshStats, numProcs)
    for key in matStruct:
      matStruct[key] += matStructWedges[key]
  elif len(tetList) != 0: # Mesh only contains tetrahedrons
    matStruct = buildVolMats(tetList, tetEdgeList, tetSingEdgeList, tetFaceList, singEleLists, tetVolP, nodeList, abcList, basisMaps, \
                             basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder3D, simType, "Tet", outBC, meshStats, numProcs)
  elif len(wedgeList) != 0: # Mesh only contains wedges
    matStruct = buildVolMats(wedgeList, wedgeEdgeList, wedgeSingEdgeList, wedgeFaceList, singEleLists, wedgeVolP, nodeList, abcList, basisMaps, \
                             basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder3D, simType, "Wedge", outBC, meshStats, numProcs)

  print(f"Matrix assembly done, time elapsed = {(time.time() - start_time):.3f} s")

  numVecUn, numScalUn = getUnknownCount(vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap)

  ## Form port matrices / source vector
  matStruct["f"] = np.zeros((unknownCnt,), dtype=np.float64)
  matStruct["Tee"], matStruct["See"] = [spa.csr_array((numVecUn, numVecUn), dtype=np.float64) for _ in range(2)]
  if simType == "A" or simType == "TCS":
    matStruct["Ten"], matStruct["Sen"] = [spa.csr_array((numVecUn, numScalUn), dtype=np.float64) for _ in range(2)]
    if simType == "TCS":
      matStruct["Tne"], matStruct["Sne"] = [spa.csr_array((numScalUn, numVecUn), dtype=np.float64) for _ in range(2)]
      matStruct["Snn"] = spa.csr_array((numScalUn, numScalUn), dtype=np.float64)
  
  if basisOrder == 1:
    vecTriBasis = 3
    scalTriBasis = 3
  elif basisOrder == 2:
    vecTriBasis = 8
    scalTriBasis = 6 
  portVecInts = np.zeros((portList.shape[0],vecTriBasis), dtype=np.float64)
  portVoltVecVals, qubitVoltVecVals = [[], []] # Vectors for calculating voltages across each port
  portVoltVecInds, qubitVoltVecInds = [[], []] # Indices of values in portVoltVecVals
  portVoltScalVals, qubitVoltScalVals = [[], []] # Vectors for calculating voltages across each port (containing scalar unknowns)
  portVoltScalInds, qubitVoltScalInds = [[], []] # Indices of scalar unknowns in portVoltScalVals
  if simType == "A" or simType == "TCS":
    portScalInts = np.zeros((portList.shape[0], scalTriBasis), dtype=np.float64)
  else:
    portScalInts = []


  for ii in range(portP["num"]):
    ## Assemble matrices for the current port
    startInd = portP[f"{ii}"]["startInd"]
    if ii == portP["num"] - 1:
      endInd = len(portList)
    else:
      endInd = portP[f"{ii+1}"]["startInd"]

    ## First save the integrals of basis functions for each port triangle
    res = getPortInts(portList[startInd:endInd], portEdgeList[startInd:endInd], portFaceList[startInd:endInd], nodeList, vecEdgeMap, vecFaceMap, \
                        scalNodeMap, scalEdgeMap, portP[f"{ii}"], quadOrder2D, basisOrder, simType, numVecUn)
    portVecInts[startInd:endInd] = res[0]
    portVoltVecVals.append(res[2])
    portVoltVecInds.append(res[3])
    if simType == "A" or simType == "TCS":
      portScalInts[startInd:endInd] = res[1]
      portVoltScalVals.append(res[4])
      portVoltScalInds.append(res[5])
    portP[f"{ii}"]["len"] = res[6]

    # res = assemblePortMats_vols(tetList, tetEdgeList, tetSingEdgeList, tetFaceList, connLists["faceTetList"], "Tet", \
    #                             portList[startInd:endInd], portEdgeList[startInd:endInd], portFaceList[startInd:endInd], \
    #                             nodeList, singEleLists, basisMaps, portP[f"{ii}"], basisOrder, quadOrder2D, simType)
    # portVoltVecVals.append(res[0])
    # portVoltVecInds.append(res[1])
    # if simType == "A" or simType == "TCS":
    #   portVoltScalVals.append(res[2])
    #   portVoltScalInds.append(res[3])
    # matArr = res[4]

    ## Now form the matrices
    matArr = assemblePortMats(portList[startInd:endInd], portEdgeList[startInd:endInd], portFaceList[startInd:endInd], nodeList, vecEdgeMap, vecFaceMap, \
                                scalNodeMap, scalEdgeMap, portP[f"{ii}"], portVecInts[startInd:endInd], portScalInts[startInd:endInd], basisOrder, simType, numVecUn, numScalUn)
    matStruct["f"] += matArr[0]

    match portP[f"{ii}"]["portType"]: # Map the results into the correct matrices
      case "R": # Resistive port
        matStr = "R"
      case "C": # Capacitive port
        matStr = "T"
      case "L": # Inductive port
        matStr = "S"
      case "Open": # Open port (do not add matrices)
        matStr = "Open"
    if matStr != "Open":
      matStruct[f"{matStr}ee"] += matArr[1]
      if simType == "A" or simType == "TCS":
        matStruct[f"{matStr}en"] += matArr[2]
        if simType == "TCS":
          matStruct[f"{matStr}ne"] += matArr[3]
          matStruct[f"{matStr}nn"] += matArr[4]

  # Account for any qubit sources (if included)
  for ii in range(qubitP["num"]):
    startInd = qubitP[f"{ii}"]["startInd"]
    if ii == qubitP["num"] - 1:
      endInd = len(qubitList)
    else:
      endInd = qubitP[f"{ii+1}"]["startInd"]

    ## Save the basis function integrals for performing voltage caclulations
    res = getPortInts(qubitList[startInd:endInd], qubitEdgeList[startInd:endInd], qubitFaceList[startInd:endInd], nodeList, vecEdgeMap, vecFaceMap, \
                        scalNodeMap, scalEdgeMap, qubitP[f"{ii}"], quadOrder2D, basisOrder, simType, numVecUn)
    qubitVoltVecVals.append(res[2])
    qubitVoltVecInds.append(res[3])
    if simType == "A" or simType == "TCS":
      qubitVoltScalVals.append(res[4])
      qubitVoltScalInds.append(res[5])
    qubitP[f"{ii}"]["len"] = res[6]

    # ## Now calculate the effect of the qubit acting as a source
    # if qubitP[f"{ii}"]["Vsmag"] != 0:
    #   matArr = assemblePortMats(qubitList[startInd:endInd], qubitEdgeList[startInd:endInd], qubitFaceList[startInd:endInd], nodeList, vecEdgeMap, vecFaceMap, \
    #                             qubitP[f"{ii}"], quadOrder2D, basisOrder, simType, N_e, N_w_e, N_n, N_f)
    #   matStruct["f"] += matArr[0]

  ## Account for any impedance surfaces (if included)
  for ii in range(connLists["impP"]["num"]):
    startInd = connLists["impP"][f"{ii}"]["startInd"]
    if ii == connLists["impP"]["num"] - 1:
      endInd = len(connLists["impList"])
    else:
      endInd = connLists["impP"][f"{ii+1}"]["startInd"]
    matArr = assembleImpSurfMats(connLists["impList"][startInd:endInd], connLists["impEdgeList"][startInd:endInd], connLists["impFaceList"][startInd:endInd], \
                                 nodeList, vecEdgeMap, vecFaceMap, scalNodeMap, scalEdgeMap, connLists["impP"][f"{ii}"], basisOrder, quadOrder2D, simType, numVecUn, numScalUn)
    match connLists["impP"][f"{ii}"]["impType"]: # Map the results into the correct matrices
      case "R": # Resistive surface
        matStr = "R"
      case "C": # Capacitive surface
        matStr = "T"
      case "L": # Inductive surface
        matStr = "S"
    matStruct[f"{matStr}ee"] += matArr[0]
    if simType == "A" or simType == "TCS":
      matStruct[f"{matStr}en"] += matArr[1]
      if simType == "TCS":
        matStruct[f"{matStr}ne"] += matArr[2]
        matStruct[f"{matStr}nn"] += matArr[3]

  # basisMaps["portVecInts"] = portVecInts
  # basisMaps["portScalInts"] = portScalInts
  basisMaps["portVoltVecVals"] = portVoltVecVals
  basisMaps["portVoltVecInds"] = portVoltVecInds
  basisMaps["qubitVoltVecVals"] = qubitVoltVecVals
  basisMaps["qubitVoltVecInds"] = qubitVoltVecInds
  basisMaps["portVoltScalVals"] = portVoltScalVals
  basisMaps["portVoltScalInds"] = portVoltScalInds
  basisMaps["qubitVoltScalVals"] = qubitVoltScalVals
  basisMaps["qubitVoltScalInds"] = qubitVoltScalInds

  ## Identify fields on the voltage boundaries
  matStruct["knownVals"], matStruct["knownInds"] \
    = assignVoltageFields(nodeList, connLists["edgeList"], connLists["voltList"], \
                          connLists["voltEdgeList"], connLists["voltFaceList"], connLists["voltP"], \
                          basisMaps, basisOrder, quadOrder2D, len(matStruct["f"]), simType)
  
  if simType == "E" or simType == "A" or simType == "TCS":
    ## Save vectors for calculating current on each voltage boundary
    basisMaps["voltCurVecVals"], basisMaps["voltCurVecInds"] \
      = getSurfCurrentInts(tetList, tetEdgeList, tetFaceList, connLists["faceTetList"], "Tet", connLists["voltList"], connLists["voltEdgeList"], connLists["voltFaceList"], \
                          nodeList, vecEdgeMap, vecFaceMap, connLists["voltP"], connLists["tetVolP"], quadOrder2D, basisOrder, simType)

  return matStruct, basisMaps, subDLists

def buildBasisMaps(nodeList, wedgeEdgeList, meshStats, simType, basisOrder, basisOrder_long):
  """
  Forms mappnings between geometric entities and basis function indices

  Inputs:
    nodeList        - Array of node numbers (accounting for PEC) and coordinates for each node
    wedgeEdgeList   - Array of edge numbers forming each wedge
    meshStats       - Dictionary with mesh statistics
    simType         - The type of simulation to use (either 'E', 'A', or 'TCS')
    basisOrder      - Order of the basis functions (currently either '1' or '2')
    basisOrder_long - Order of the basis functions in the longitudinal direction (for wedges) (either '1' or '2')
  Outputs:
    vecEdgeMap    - Array mapping from edges to vector basis functions
    vecFaceMap    - Array mapping from faces to vector basis functions
    vecVolMap     - Array mapping from element volumes to vector basis functions
    scalNodeMap   - Array mapping from nodes to scalar basis functions
    scalEdgeMap   - Array mapping from edges to scalar basis functions
    unknownCnt    - Total number of unknowns for the problem
  """
  N_n, N_e, N_w_e, N_f, N_q_f, N_tet, N_wedge \
    = meshStats["N_n"], meshStats["N_e"], meshStats["N_w_e"], meshStats["N_f"], meshStats["N_q_f"], meshStats["N_tet"], meshStats["N_wedge"]

  # Enumerate each type of basis function
  numEdgeVecBasis = basisOrder # Number of vector basis functions on each triangular edge
  numLongEdgeVecBasis = basisOrder_long # Number of vector basis functions on each longitudinal / vertical edge
  numFaceVecBasis = basisOrder*(basisOrder-1) # Number of vector basis functions on each triangular face
  numQuadFaceVecBasis = basisOrder*(basisOrder_long-1) \
                        + basisOrder_long*(basisOrder-1) # Number of vector basis functions on each quadrilateral face
  numTetVolVecBasis = int(basisOrder*(basisOrder-1)*(basisOrder-2)/2) # Number of scalar basis functions in each wedge volume
  numWedgeVolVecBasis = basisOrder*(basisOrder-1)*(basisOrder_long-1) \
                        + int(basisOrder_long*(basisOrder-1)*(basisOrder-2)/2) # Number of vector basis functions in each wedge volume
  numTotVecBasis = N_e*numEdgeVecBasis + N_w_e*numLongEdgeVecBasis + N_f*numFaceVecBasis \
                   + N_q_f*numQuadFaceVecBasis + N_tet*numTetVolVecBasis + N_wedge*numWedgeVolVecBasis

  # TODO: Add support for higher order longitudinal scalar basis functions
  numNodeScalBasis = 1 # Number of scalar basis functions on each node
  numEdgeScalBasis = basisOrder-1
  numLongEdgeScalBasis = basisOrder_long-1 # TODO: Validate
  numFaceScalBasis = int(basisOrder*(basisOrder-3)/2) + 1
  numQuadFaceScalBasis = 0 # TODO: Update with actual numbering
  numTetVolScalBasis = int((basisOrder+1)*(basisOrder+2)*(basisOrder+3)/6) - 4*numNodeScalBasis - 6*numEdgeScalBasis - 4*numFaceScalBasis
  numWedgeVolScalBasis = 0 # TODO: Update with actual numbering
  numTotScalBasis = N_n*numNodeScalBasis + N_e*numEdgeScalBasis + N_w_e*numLongEdgeScalBasis + N_f*numFaceScalBasis \
                     + N_q_f*numQuadFaceScalBasis + N_tet*numTetVolScalBasis + N_wedge*numWedgeVolScalBasis

  # Determine which functions are used based on the simulation type
  if simType == "E" or simType == "A" or simType == "TCS":
    # unknownCnt = numTotVecBasis # Total number of unknowns

    ## Form a mapping from edges to unknowns
    maxEdgeVecBasis = np.max([numEdgeVecBasis, numLongEdgeVecBasis])
    vecEdgeMap = -np.ones((N_e+N_w_e, maxEdgeVecBasis), dtype=np.int32)
    longEdgeNums = np.unique(wedgeEdgeList[:,6:]) # Numbers of longitudinal edges of wedges
    ind = 0
    for ii in range(N_e+N_w_e):
      if ii in longEdgeNums: # Longitudinal edge
        vecEdgeMap[ii,:numLongEdgeVecBasis] = np.arange(ind, ind+numLongEdgeVecBasis)
        ind += numLongEdgeVecBasis
      else: # Triangular edge
        vecEdgeMap[ii,:numEdgeVecBasis] = np.arange(ind, ind+numEdgeVecBasis)
        ind += numEdgeVecBasis

    ## Form another mapping from faces to unknowns
    maxFaceVecBasis = np.max([numFaceVecBasis, numQuadFaceVecBasis])
    vecFaceMap = -np.ones((N_f+N_q_f, maxFaceVecBasis), dtype=np.int32)
    if numFaceVecBasis != 0:
      vecFaceMap[:N_f,0] = np.arange(ind, ind + numFaceVecBasis*N_f, numFaceVecBasis) # Triangular faces
      for ii in range(1,numFaceVecBasis):
        vecFaceMap[:N_f,ii] = vecFaceMap[:N_f,0] + ii
      ind += numFaceVecBasis*N_f

    if numQuadFaceVecBasis != 0 and N_q_f != 0:
      vecFaceMap[N_f:,0] = np.arange(ind, ind + numQuadFaceVecBasis*N_q_f, numQuadFaceVecBasis) # Quadrilateral faces
      for ii in range(1,numQuadFaceVecBasis):
        vecFaceMap[N_f:,ii] = vecFaceMap[N_f:,0] + ii
      ind += numQuadFaceVecBasis*N_q_f

    ## Form another mapping from volumes to unknowns
    maxVolVecBasis = np.max([numTetVolVecBasis, numWedgeVolVecBasis])
    vecVolMap = -np.ones((N_tet+N_wedge, maxVolVecBasis), dtype=np.int32)
    if numTetVolVecBasis != 0 and N_tet != 0:
      vecVolMap[:N_tet,0] = np.arange(ind, ind + numTetVolVecBasis*N_tet, numTetVolVecBasis) # Tetrahedron volumes
      for ii in range(1,numTetVolVecBasis):
        vecVolMap[:N_tet,ii] = vecVolMap[:N_tet,0] + ii
      ind += numTetVolVecBasis*N_tet

    if numWedgeVolVecBasis != 0 and N_wedge != 0:
      vecVolMap[N_tet:,0] = np.arange(ind, ind + numWedgeVolVecBasis*N_wedge, numWedgeVolVecBasis) # Wedge volumes
      for ii in range(1,numWedgeVolVecBasis):
        vecVolMap[N_tet:,ii] = vecVolMap[N_tet:,0] + ii
      ind += numWedgeVolVecBasis*N_wedge

    unknownCnt = ind # Total number of unknowns

  else:
    vecEdgeMap, vecFaceMap, vecVolMap \
      = [-np.ones((0,0), dtype=np.int32), -np.ones((0,0), dtype=np.int32), -np.ones((0,0), dtype=np.int32)]
    unknownCnt = 0

  if simType == "Phi" or simType == "A" or simType == "TCS":
    unknownCnt += numTotScalBasis # Total number of unknowns

    ## Form a mapping from nodes to unknowns
    scalNodeMap = nodeList[:,0].astype(np.int32) # Since there's always only one unknown per node, just use their numbers
    PECNodes = scalNodeMap < -1 # Marks the non-GND PEC
    scalNodeMap[PECNodes] = N_n + scalNodeMap[PECNodes] + 1 # Shift these to a non-negative shared index for that PEC

    ## Form a mapping from edges to unknowns
    maxEdgeScalBasis = np.max([numEdgeScalBasis, numLongEdgeScalBasis])
    scalEdgeMap = -np.ones((N_e+N_w_e, maxEdgeScalBasis), dtype=np.int32)
    longEdgeNums = np.unique(wedgeEdgeList[:,6:]) # Numbers of longitudinal edges of wedges
    ind = N_n*numNodeScalBasis
    for ii in range(N_e+N_w_e):
      if ii in longEdgeNums: # Longitudinal edge
        scalEdgeMap[ii,:numLongEdgeScalBasis] = np.arange(ind, ind+numLongEdgeScalBasis)
        ind += numLongEdgeScalBasis
      else: # Triangular edge
        scalEdgeMap[ii,:numEdgeScalBasis] = np.arange(ind, ind+numEdgeScalBasis)
        ind += numEdgeScalBasis

    ## Form another mapping from faces to unknowns
    maxFaceScalBasis = np.max([numFaceScalBasis, numQuadFaceScalBasis])
    scalFaceMap = -np.ones((N_f+N_q_f, maxFaceScalBasis), dtype=np.int32)
    if numFaceScalBasis != 0:
      scalFaceMap[:N_f,0] = np.arange(ind, ind + numFaceScalBasis*N_f, numFaceScalBasis) # Triangular faces
      for ii in range(1,numFaceScalBasis):
        scalFaceMap[:N_f,ii] = scalFaceMap[:N_f,0] + ii
      ind += numFaceScalBasis*N_f

    if numQuadFaceScalBasis != 0 and N_q_f != 0:
      scalFaceMap[N_f:,0] = np.arange(ind, ind + numQuadFaceScalBasis*N_q_f, numQuadFaceScalBasis) # Quadrilateral faces
      for ii in range(1,numQuadFaceScalBasis):
        scalFaceMap[N_f:,ii] = scalFaceMap[N_f:,0] + ii
      ind += numQuadFaceScalBasis*N_q_f

    ## Form another mapping from volumes to unknowns
    maxVolScalBasis = np.max([numTetVolScalBasis, numWedgeVolScalBasis])
    scalVolMap = -np.ones((N_tet+N_wedge, maxVolScalBasis), dtype=np.int32)
    if numTetVolScalBasis != 0 and N_tet != 0:
      scalVolMap[:N_tet,0] = np.arange(ind, ind + numTetVolScalBasis*N_tet, numTetVolScalBasis) # Tetrahedron volumes
      for ii in range(1,numTetVolScalBasis):
        scalVolMap[:N_tet,ii] = scalVolMap[:N_tet,0] + ii
      ind += numTetVolScalBasis*N_tet

    if numWedgeVolScalBasis != 0 and N_wedge != 0:
      scalVolMap[N_tet:,0] = np.arange(ind, ind + numWedgeVolScalBasis*N_wedge, numWedgeVolScalBasis) # Wedge volumes
      for ii in range(1,numWedgeVolScalBasis):
        scalVolMap[N_tet:,ii] = scalVolMap[N_tet:,0] + ii
      ind += numWedgeVolScalBasis*N_wedge

  else:
    scalNodeMap, scalEdgeMap, scalFaceMap, scalVolMap \
          = [-np.ones((0,0), dtype=np.int32), -np.ones((0,0), dtype=np.int32), -np.ones((0,0), dtype=np.int32), -np.ones((0,0), dtype=np.int32)]

  return vecEdgeMap, vecFaceMap, vecVolMap, scalNodeMap, scalEdgeMap, unknownCnt

def buildBasisMapsSingular_orderN(singEleLists, tetList, tetEdgeList, tetFaceList, wedgeList, wedgeEdgeList, wedgeFaceList, \
                                  meshStats, simType, basisOrder, singOrder, basisOrder_long, singOrder_long, singEdgeFlag):
  """
  Forms mappings between geometric entities and basis function indices when singular edges are present
  This version of the function is designed to be used with arbitrarily high-order basis functions (Nth order)

  Inputs:
    singEleLists    - Dicitionary containing information about singular elements
    tetList         - Array of nodes forming each tet
    tetEdgeList     - Array of edges forming each tet
    tetFaceList     - Array of faces forming each tet
    wedgeList       - Array of nodes forming each wedge
    wedgeEdgeList   - Array of edges forming each wedge
    wedgeFaceList   - Array of faces forming each wedge
    meshStats       - Dictionary with mesh statistics
    simType         - The type of simulation to use (currently only "E" is supported")
    basisOrder      - Order of the basis functions
    singOrder       - Order of the singular functions
    basisOrder_long - Order of the basis functions in the longitudinal direction for wedge/prism elements
    singOrder_long  - Order of the singular functions in the longitudinal direction for wedge/prism elements
    singEdgeFlag    - Flag indicating whether to use the functions singular with respect to the edge
  Outputs:
    vecEdgeMap      - Array mapping from edges to vector basis functions
    singVecEdgeMap  - Array mapping from singular edges to vector basis functions
    vecFaceMap      - Array mapping from faces to vector basis functions
    vecVolMap       - Array mapping from the element volume to vector basis functions
    scalNodeMap     - Array mapping from nodes to scalar basis functions
    scalEdgeMap     - Array mapping from edges to scalar basis functions
    unknownCnt      - Total number of unknowns for the problem
    singEleLists    - 
  """

  singTets = singEleLists["singTets"]
  tetSingEdgeList = singEleLists["tetSingEdgeList"]
  singWedges = singEleLists["singWedges"]
  wedgeSingEdgeList = singEleLists["wedgeSingEdgeList"]
  singTetCoeffs = singEleLists["singTetCoeffs"]
  singWedgeCoeffs = singEleLists["singWedgeCoeffs"]
  singTetMap = singEleLists["singTetMap"]
  singWedgeMap = singEleLists["singWedgeMap"]

  # Sort the singular elements by number of singular nodes
  if len(singTets) > 0:
    tetSingNodeCnt = np.sum(singTetCoeffs != 0, axis=1)
    singTets = singTets[np.argsort(tetSingNodeCnt)]
  if len(singWedges) > 0:
    raise ValueError("Singular wedge/prism elements are not supported in the current version of this package.")
    wedgeSingNodeCnt = np.sum(singWedgeCoeffs != 0, axis=1)
    singWedges = singWedges[np.argsort(wedgeSingNodeCnt)]

  N_n, N_e, N_w_e, N_f, N_q_f, N_tet, N_wedge, N_sing_e = \
    meshStats["N_n"], meshStats["N_e"], meshStats["N_w_e"], meshStats["N_f"], \
    meshStats["N_q_f"], meshStats["N_tet"], meshStats["N_wedge"], meshStats["N_sing_e"]

  if simType == "A" or simType == "TCS":
    raise ValueError("Error: The potential-based and tree-cotree solvers are not currently supported with singular basis functions.")
  else:
    scalNodeMap = np.zeros((0,), dtype=np.int32)
    scalEdgeMap = np.zeros((0,), dtype=np.int32)

  ## Map edges to rotational basis functions
  numVecEdgeStdBasis = basisOrder # Number of standard (nonsingular) functions on each edge
  numVecEdgeSingBasis = singOrder # Number of singular functions on each edge
  # Total number of functions per edge (accounting for multiple sets of singular functions)
  numVecEdgeBasis = numVecEdgeStdBasis + 2*numVecEdgeSingBasis 

  # Numbering for longitudinal edges in wedges / prisms
  numVecLongEdgeStdBasis = basisOrder_long # Number of standard (nonsingular) functions on each longitudinal edge
  numVecLongEdgeSingBasis = singOrder_long # Number of singular functions on each longitudinal edge
  # Total number of functions per edge (accounting for multiple sets of singular functions)
  numVecLongEdgeBasis = numVecLongEdgeStdBasis + 2*numVecLongEdgeSingBasis 

  maxVecEdgeBasis = np.max([numVecEdgeBasis, numVecLongEdgeBasis]) # Make sure the array is sized for the longer of the two
  vecEdgeMap = -1*np.ones((N_e+N_w_e, maxVecEdgeBasis), dtype=np.int32) # Mark unused edges with -1
  longEdgeNums = np.unique(wedgeEdgeList[:,6:]) # Numbers of longitudinal edges of wedges
  curInd = 0
  for ii in range(N_e+N_w_e):
    if ii not in longEdgeNums: # Triangular / transverse edges
      vecEdgeMap[ii,:numVecEdgeStdBasis] = np.arange(curInd, curInd+numVecEdgeStdBasis)
      curInd += numVecEdgeStdBasis
    else: # Longitudinal edges
      vecEdgeMap[ii,:numVecLongEdgeStdBasis] = np.arange(curInd, curInd+numVecLongEdgeStdBasis)
      curInd += numVecLongEdgeStdBasis

  ## Map edges to singular basis functions
  for eleType in ["Tet", "Wedge"]:
    if eleType == "Tet":
      singEles = singTets
      eleList = tetList
      eleEdgeList = tetEdgeList
      singEleCoeffs = singTetCoeffs
      singEleMap = singTetMap
      singEdgeNums = np.array([[0,1,2], [0,3,4], [1,3,5], [2,4,5]], dtype=np.int32) # Edges touching each singular node
      numEdges = 6 # Number of edges per tet
    elif eleType == "Wedge":
      singEles = singWedges
      eleList = wedgeList
      eleEdgeList = wedgeEdgeList
      singEleCoeffs = singWedgeCoeffs
      singEleMap = singWedgeMap
      singEdgeNums = np.array([[0,1,6], [0,2,7], [1,2,8], [3,4,6], [3,5,7], [4,5,8]], dtype=np.int32) # Edges touching each singular node
      numEdges = 9 # Number of edges per wedge

    for eleInd in singEles:
      # Keep count of how many times each edge is visited to get the correct basis # assignment
      # print((eleInd, eleList[eleInd], eleEdgeList[eleInd], wedgeFaceList[eleInd], singEleCoeffs[singEleMap[eleInd]]))
      edgeCnt = np.zeros((numEdges,), dtype=np.int32) 
      nodeSort = sortSingNodes(eleList[eleInd], eleType) # Use the sorted nodes to correct for wedges with top nodes out of order
      for nodeInd, edgeSet in enumerate(singEdgeNums[nodeSort]):
        if singEleCoeffs[singEleMap[eleInd], nodeSort[nodeInd]] == 0: # If the singular coefficient is 0, skip this node since it won't contribute singular functions
          continue

        for ii, edgeInd in enumerate(eleEdgeList[eleInd][edgeSet]): 
          if edgeSet[ii] < 6: # Tet edges or triangular edges of a wedge
            prevBasisCnt = numVecEdgeStdBasis
            curBasisCnt = numVecEdgeSingBasis
          else: # Longitudinal edges of a wedge
            prevBasisCnt = numVecLongEdgeStdBasis
            curBasisCnt = numVecLongEdgeSingBasis

          basisRange = prevBasisCnt + edgeCnt[edgeSet[ii]]*curBasisCnt \
                      + np.arange(curBasisCnt)
          edgeCnt[edgeSet[ii]] += 1

          # Make sure the edge isn't on PEC and hasn't already been visited
          if edgeInd >= 0 and len(basisRange) > 0 and vecEdgeMap[edgeInd,basisRange[0]] == -1:
            vecEdgeMap[edgeInd,basisRange] = np.arange(curInd, curInd+curBasisCnt)
            curInd += curBasisCnt

  vecSingEdgeMap = -np.ones((N_sing_e,1), dtype=np.int32)
  
  ## Form another mapping from faces to basis functions
  numVecFaceStdBasis = basisOrder*(basisOrder-1) # Number of standard (nonsingular) functions on each face
  numVecFaceSingBasis = int(singOrder*(singOrder-1)/2) # Number of singular functions on each face
  # numVecFaceRotBasis = singOrder # Number of rotational functions on each face - eliminating dependent functions
  numVecFaceRotBasis = int((singOrder+1)*singOrder/2) # Number of rotational functions on each face
  numVecFaceSingLineBasis = numVecFaceRotBasis # Number of singular line functions on each face

  # Numbering for quadrilateral faces in wedges / prisms
  numVecQuadFaceStdBasis = basisOrder*(basisOrder_long-1) + (basisOrder-1)*basisOrder_long # Number of standard (nonsingular) functions on each quadrilateral face
  numVecQuadFaceSingBasis = singOrder*(singOrder_long-1) # Number of singular functions built from triangular edges on each quadrilateral face
  numVecQuadFaceSingBasis_long = (singOrder-1)*singOrder_long # Number of singular functions built from longitudinal edges on each quadrilateral face
  numVecQuadFaceRotBasis = singOrder*singOrder_long # Number of rotational functions on each quadrilateral face
  numVecQuadFaceSingLineBasis = numVecQuadFaceRotBasis # Number of singular line functions on each quadrilateral face

  # Total number of functions per face (accounting for multiple sets of singular functions)
  numVecFaceBasis = numVecFaceStdBasis + 3*(numVecFaceSingBasis + numVecFaceRotBasis)
  numVecFaceBasis += 3*numVecFaceSingLineBasis # Add singular line functions
  numVecQuadFaceBasis = numVecQuadFaceStdBasis + 4*(numVecQuadFaceSingBasis + numVecQuadFaceSingBasis_long + numVecQuadFaceRotBasis)
  numVecQuadFaceBasis += 2*numVecQuadFaceSingLineBasis # TODO: Sort out if longitudinal edges need to be added - this is just for the 2 triangular edges

  maxVecFaceBasis = np.max([numVecFaceBasis, numVecQuadFaceBasis]) # Make sure the array is sized for the longer of the two
  vecFaceMap = -1*np.ones((N_f+N_q_f, maxVecFaceBasis), dtype=np.int32)
  if numVecFaceStdBasis != 0:
    vecFaceMap[:N_f,0] = np.arange(curInd, curInd+numVecFaceStdBasis*N_f, numVecFaceStdBasis) 
    for jj in range(numVecFaceStdBasis):
      vecFaceMap[:N_f,jj] = vecFaceMap[:N_f,0] + jj
    curInd += numVecFaceStdBasis*N_f
  if numVecQuadFaceStdBasis != 0:
    vecFaceMap[N_f:,0] = np.arange(curInd, curInd+numVecQuadFaceStdBasis*N_q_f, numVecQuadFaceStdBasis) 
    for jj in range(numVecQuadFaceStdBasis):
      vecFaceMap[N_f:,jj] = vecFaceMap[N_f:,0] + jj
    curInd += numVecQuadFaceStdBasis*N_q_f

  ## Map faces to singular basis functions
  for eleType in ["Tet", "Wedge"]:
    if eleType == "Tet":
      singEles = singTets
      eleList = tetList
      eleFaceList = tetFaceList
      eleSingEdgeList = tetSingEdgeList
      singEleCoeffs = singTetCoeffs
      singEleMap = singTetMap
      singFaceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32) # Faces touching each singular node
      singFaceEdgeNums = np.array([[0,1], [0,2], [1,2], [0,3], [1,3], [2,3]], dtype=np.int32) # Faces touching each singular edge
      numFaces = 4 # Number of faces per tet
    elif eleType == "Wedge":
      singEles = singWedges
      eleList = wedgeList
      eleFaceList = wedgeFaceList
      eleSingEdgeList = wedgeSingEdgeList
      singEleCoeffs = singWedgeCoeffs
      singEleMap = singWedgeMap
      singFaceNums = np.array([[0,2,3], [0,2,4], [0,3,4], [1,2,3], [1,2,4], [1,3,4]], dtype=np.int32) # Faces touching each singular node
      singFaceEdgeNums = np.array([[0,2], [0,3], [0,4], [1,2], [1,3], [1,4]], dtype=np.int32) # Faces touching each singular edge
      quadFaceNums = np.array([[0, 1, 3, 4], [0, 2, 3, 5], [1, 2, 4, 5]], dtype=np.int32) # Nodes touching each face
      numFaces = 5 # Number of faces per wedge

    for eleInd in singEles:
      # Keep count of how many times each face is visited to get the correct basis # assignment
      faceCnt = np.zeros((numFaces,), dtype=np.int32) 
      nodeSort = sortSingNodes(eleList[eleInd], eleType) # Use the sorted nodes to correct for wedges with top nodes out of order
      for nodeInd, faceSet in enumerate(singFaceNums[nodeSort]):
        sortedNodeInd = nodeSort[nodeInd]
        if singEleCoeffs[singEleMap[eleInd], sortedNodeInd] == 0: # If the singular coefficient is 0, skip this node since it won't contribute singular functions
          continue

        for ii, faceInd in enumerate(eleFaceList[eleInd][faceSet]): 
          if eleType == "Wedge" and faceSet[ii] >= 2: # Quadrilateral face
            # If a prior node on the face at this z position is also singular, remove the longitudinal singular functions
            # And if a prior node on the vertical edge is also singular, remove the transverse singular functions
            quadFaceNodes = quadFaceNums[faceSet[ii]-2] # Nodes of this quadrilateral face
            nodeQuadFaceInd = np.where(sortedNodeInd == quadFaceNodes)[0][0] # Index of the current node on the quadrilateral face

            longFuncFlag, tranFuncFlag = True, True # Flags indicating whether to include transverse and longitudinal functions
            # If this is the second node on this vertical position on the face, check whether the first is singular
            if np.mod(nodeQuadFaceInd, 2) == 1 \
              and singEleCoeffs[singEleMap[eleInd], quadFaceNodes[nodeQuadFaceInd-1]] != 0:
              # If yes, don't add any longitudinal functions
              longFuncFlag = False
              # prevBasisCnt, curBasisCnt = 0, 0

            # If this is the upper node on this vertical edge, check whether the first is singular
            if nodeQuadFaceInd >= 2 and singEleCoeffs[singEleMap[eleInd], quadFaceNodes[nodeQuadFaceInd-2]] != 0:
              # If yes, don't add any transverse functions
              tranFuncFlag = False

            prevBasisCnt = numVecQuadFaceStdBasis \
                            + faceCnt[faceSet[ii]]*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) # Number of functions already added
            if not tranFuncFlag: # If transverse functions aren't included, offset the positioning to account for that
              prevBasisCnt += numVecQuadFaceSingBasis
            curBasisCnt = tranFuncFlag*numVecQuadFaceSingBasis + longFuncFlag*(numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) # Number of functions to be added
          else: # Triangular face
            prevBasisCnt = numVecFaceStdBasis + faceCnt[faceSet[ii]]*(numVecFaceSingBasis+numVecFaceRotBasis) # Number of functions already added
            curBasisCnt = numVecFaceSingBasis+numVecFaceRotBasis # Number of functions to be added

          basisRange = prevBasisCnt + np.arange(curBasisCnt)
          # basisRange = prevBasisCnt + faceCnt[faceSet[ii]]*(curBasisCnt) + np.arange(curBasisCnt)
          faceCnt[faceSet[ii]] += 1

          # Make sure the face isn't on PEC and hasn't already been visited
          if faceInd >= 0 and len(basisRange) > 0 and vecFaceMap[faceInd,basisRange[0]] == -1:
            vecFaceMap[faceInd,basisRange] = curInd + np.arange(curBasisCnt)
            curInd += curBasisCnt

    # Add contributions from singular edge functions
    if singEdgeFlag:
      for eleInd in singEles:
        # Keep count of how many times each face is visited to get the correct basis # assignment
        faceCnt = np.zeros((numFaces,), dtype=np.int32) 
        for edgeNum, faceSet in enumerate(singFaceEdgeNums):
          if eleSingEdgeList[eleInd, edgeNum] == -1: # If the edge isn't singular, skip it since it won't contribute singular functions
            continue

          for ii, faceInd in enumerate(eleFaceList[eleInd][faceSet]): 
            if eleType == "Wedge" and faceSet[ii] >= 2: # Quadrilateral face
              prevBasisCnt = numVecQuadFaceStdBasis + 4*(numVecQuadFaceSingBasis+numVecQuadFaceSingBasis_long+numVecQuadFaceRotBasis) \
                              + faceCnt[faceSet[ii]]*numVecQuadFaceSingLineBasis # Number of functions already added
              curBasisCnt = numVecQuadFaceSingLineBasis # Number of functions to be added
              
            else: # Triangular face
              prevBasisCnt = numVecFaceStdBasis + 3*(numVecFaceSingBasis+numVecFaceRotBasis) \
                              + faceCnt[faceSet[ii]]*numVecFaceSingLineBasis # Number of functions already added
              curBasisCnt = numVecFaceSingLineBasis # Number of functions to be added
            basisRange = prevBasisCnt + np.arange(curBasisCnt)
            faceCnt[faceSet[ii]] += 1

            # Make sure the face isn't on PEC and hasn't already been visited
            if faceInd >= 0 and len(basisRange) != 0 and vecFaceMap[faceInd,basisRange[0]] == -1:
              vecFaceMap[faceInd,basisRange] = curInd + np.arange(curBasisCnt)
              curInd += curBasisCnt

  ## Form another mapping from element volumes to basis functions
  for eleType in ["Tet", "Wedge"]:
    if eleType == "Tet":
      singEles = singTets
      singEleCoeffs = singTetCoeffs
      singEleMap = singTetMap
      eleList = tetList
      eleSingEdgeList = tetSingEdgeList
      numNodes = 4 # Number of nodes per tet

      numVecVolStdBasis = int((basisOrder-2)*(basisOrder-1)*basisOrder/2) # Number of standard (nonsingular) functions
      numVecVolSingBasis = int((singOrder-2)*(singOrder-1)*singOrder/6) # Singular functions
      numVecVolRotBasis = int((singOrder-1)*singOrder*(singOrder+1)/6) # Rotational singular functions (face-derived)
      numVecVolSingLineGradBasis = numVecVolRotBasis # Gradient edge-singular functions (face-derived)
      numVecVolSingLineRotBasis = singOrder + int((singOrder-1)*singOrder) + int((singOrder-2)*(singOrder-1)*singOrder/6) # Rotational edge-singular functions
      numVecVolSingLineBasis = numVecVolSingLineGradBasis + numVecVolSingLineRotBasis

      # Total number of functions in the tet (accounting for multiple sets of singular functions)                                                 
      numVecVolBasis = numVecVolStdBasis + 4*(numVecVolSingBasis + numVecVolRotBasis)
      numVecVolBasis += 6*numVecVolSingLineBasis

      ## Add nonsingular functions
      vecVolMap = -1*np.ones((N_tet+N_wedge, numVecVolBasis), dtype=np.int32)
      if numVecVolStdBasis != 0:
        vecVolMap[:N_tet,0] = np.arange(curInd, curInd+numVecVolStdBasis*N_tet, numVecVolStdBasis) 
        for jj in range(numVecVolStdBasis):
          vecVolMap[:N_tet,jj] = vecVolMap[:N_tet,0] + jj
        curInd += numVecVolStdBasis*N_tet
    elif eleType == "Wedge":
      singEles = singWedges
      singEleCoeffs = singWedgeCoeffs
      singEleMap = singWedgeMap
      eleList = wedgeList
      eleSingEdgeList = wedgeSingEdgeList
      numNodes = 6 # Number of nodes per wedge

      numVecVolStdBasis = basisOrder*(basisOrder-1)*(basisOrder_long-1) \
                          + int((basisOrder-1)*(basisOrder-2)*basisOrder_long/2) # Number of standard (nonsingular) functions
      numVecVolSingBasis = int(singOrder*(singOrder-1)*(singOrder_long-1)/2) # Singular functions
      numVecVolSingBasis_long = int((singOrder-1)*(singOrder-2)*singOrder_long/2) # Singular functions built from longitudinal functions
      numVecVolRotBasis = int((singOrder+1)*singOrder*(singOrder_long-1)/2) # Rotational singular functions (face-derived)
      numVecVolRotBasis_long = int(singOrder*(singOrder-1)*singOrder_long/2) # Rotational singular functions (quadrilateral face-derived)
      numVecVolTotBasis = numVecVolSingBasis+numVecVolRotBasis \
                        + numVecVolSingBasis_long+numVecVolRotBasis_long # Number of basis functions in the volume (including longitudinal functions)

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

      # Total number of functions in the tet (accounting for multiple sets of singular functions)                                                 
      numVecVolBasis = numVecVolStdBasis + 6*numVecVolTotBasis
      numVecVolBasis += 6*numVecVolSingLineTotBasis # TODO: Sort out if longitudinal edge functions are necessary

      if numVecVolBasis > len(vecVolMap[0]):
        # If there are more basis functions in the wedges than tets, extend the list to account for those
        vecVolMap = np.hstack([vecVolMap, -1*np.ones((N_tet+N_wedge, numVecVolBasis-len(vecVolMap[0])), dtype=np.int32)])

      ## Add nonsingular functions
      if numVecVolStdBasis != 0:
        vecVolMap[N_tet:,0] = np.arange(curInd, curInd+numVecVolStdBasis*N_wedge, numVecVolStdBasis) 
        for jj in range(numVecVolStdBasis):
          vecVolMap[N_tet:,jj] = vecVolMap[N_tet:,0] + jj
        curInd += numVecVolStdBasis*N_wedge

    for eleInd in singEles:
      singNodeFlags = singEleCoeffs[singEleMap[eleInd]] != 0 # Marks whether each node is singular
      singEdgeFlags = eleSingEdgeList[eleInd] != -1 # Marks whether each edge is singular
      numSingNodes = np.sum(singNodeFlags) # Number of singular points in the element
      if eleType == "Tet":
        # For tetrahedrons, just add functions for each node and edge
        numCurBasis = numSingNodes*(numVecVolSingBasis+numVecVolRotBasis) # Number of functions to add
        basisRange = numVecVolStdBasis + np.arange(numCurBasis)
        vecVolMap[eleInd, basisRange] = curInd + np.arange(numCurBasis) 
        curInd += numCurBasis

        # Add edge-singular functions
        if singEdgeFlag:
          numSingEdges = np.sum(eleSingEdgeList[eleInd] != -1)
          numCurBasis = numSingEdges*numVecVolSingLineBasis # Number of functions to add
          basisRange = numVecVolStdBasis + numNodes*(numVecVolSingBasis+numVecVolRotBasis) + np.arange(numCurBasis)
          vecVolMap[eleInd, basisRange] = curInd + np.arange(numCurBasis) 
          curInd += numCurBasis
      elif eleType == "Wedge":
        # For wedges, only add longitudinal functions for one singular node on the face
        singNodeCnt = 0 # Track the number of singular nodes accounted for
        nodeSort = sortSingNodes(eleList[eleInd], eleType) # Use the sorted nodes to correct for wedges with top nodes out of order
        for nodeInd in nodeSort:
          if singNodeFlags[nodeInd]:
            triNodeInd = np.mod(nodeInd,3)
            vertInd = int(np.floor(nodeInd / 3))

            longFuncFlag, tranFuncFlag = True, True # Flags indicating whether to include transverse and longitudinal functions
            if np.any(singNodeFlags[(nodeInd-triNodeInd):nodeInd] != 0):
              # If this isn't the first node on this triangular face, don't add longitudinal functions
              longFuncFlag = False
            if vertInd == 1 and singEleCoeffs[singEleMap[eleInd], nodeInd-3] != 0:
              # If this node is on the top face and the node below it also is, don't add transverse functions
              tranFuncFlag = False
            
            numCurBasis = tranFuncFlag*(numVecVolSingBasis+numVecVolRotBasis) + longFuncFlag*(numVecVolSingBasis_long+numVecVolRotBasis_long)
            basisRange = numVecVolStdBasis + singNodeCnt*numVecVolTotBasis + np.arange(numCurBasis)
            if not tranFuncFlag: # If transverse functions aren't included, offset the positioning to account for that
              basisRange += numVecVolSingBasis+numVecVolRotBasis
            vecVolMap[eleInd+N_tet, basisRange] = curInd + np.arange(numCurBasis) 
            curInd += numCurBasis
            singNodeCnt += 1

        # Add edge-singular functions
        singEdgeCnt = 0 # Track the number of singular nodes accounted for
        if singEdgeFlag:
          for edgeInd in np.arange(6): # TODO: Sort out if longitudinal edge funcs are needed
            if singEdgeFlags[edgeInd]:
              triEdgeInd = np.mod(edgeInd,3)
              vertInd = int(np.floor(edgeInd / 3))

              longFuncFlag, tranFuncFlag = True, True # Flags indicating whether to include transverse and longitudinal functions
              if np.any(singEdgeFlags[(edgeInd-triEdgeInd):edgeInd] != 0):
                # If this isn't the first edge on this triangular face, don't add longitudinal functions
                longFuncFlag = False
              if vertInd == 1 and singEdgeFlags[edgeInd-3] != 0:
                # If this edge is on the top face and the node below it also is, don't add transverse functions
                tranFuncFlag = False
              # NOTE: The longitudinal functions appear to be dependent with the node-singular functions, so those are ommitted
              numCurBasis = tranFuncFlag*(numVecVolSingLineGradBasis+numVecVolSingLineRotBasis) #\
                            #+ longFuncFlag*(numVecVolSingLineGradBasis_long+numVecVolSingLineRotBasis_long)
              
              basisRange = numVecVolStdBasis + 6*numVecVolTotBasis + singEdgeCnt*numVecVolSingLineTotBasis + np.arange(numCurBasis)
              vecVolMap[eleInd+N_tet, basisRange] = curInd + np.arange(numCurBasis) 
              curInd += numCurBasis
              singEdgeCnt += 1
      
  unknownCnt = curInd

  return vecEdgeMap, vecFaceMap, vecVolMap, vecSingEdgeMap, scalNodeMap, scalEdgeMap, unknownCnt

def buildVolMats(eleList, eleEdgeList, eleSingEdgeList, eleFaceList, singEleLists, volP, nodeList, boundaryList, basisMaps, \
                 basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder, simType, eleType, bc, meshStats, numProcs):
  """
  A function for constructing matrices from connectivity lists for a set of elements

  Inputs:
    eleList         - Contains node numbers associated with each element, as well as material properties in each element
    eleEdgeList     - Edge numbers associated with each element
    eleFaceList     - Face numbers associated with each element (only used if basisOrder == 2)
    singEleLists    - Dictionary containing information about singular elements and basis functions
    volP            - Material parameters of each volume associated with the current element type
    nodeList        - List of renumbered nodes (to account for PECs) and their coordinates
    boundaryList    - List of node numbers forming each boundary triangle
    basisMaps       - Contains information on how basis functions map to geometric entities
    basisOrder      - The order of basis functions to use. Currently '1' and '2' are supported for tetrahedral elements and wedge elements.
    singOrder       - The order of singular basis functions (if used)
    basisOrder_long - Order of the nonsingular basis functions in the longitudinal direction (for wedges/prisms)
    singOrder_long  - Order of the singular basis functions in the longitudinal direction (for wedges/prisms)
    quadOrder       - The order of gaussian quadrature to use for numerical integration
    simType         - The type of simulation to use (either 'E', 'A', or 'TCS')
    eleType         - The type of elements specified by the connectivity lists. Currently either 'tet' for tetrahedrons, or 'Wedge' for wedges (prisms)
    bc              - The boundary condition for the outer boundary
    meshStats       - Dictionary containing numbering information about mesh entities (number of edges, tets, etc)
    numProcs        - The number of parallel processes available for use

  Outputs:
    matStruct       - A dictionary containing the matrices generated by assembleMats. See assembleMats for more details on the individual matrices.
  """

  matStruct = {}
  if numProcs != 1:
    ## Split the element list into segments that are independently managed by each process
    eleSplit, startEle = splitArr(eleList, numProcs)
    eleEdgeSplit, _ = splitArr(eleEdgeList, numProcs)
    eleSingEdgeSplit, _ = splitArr(eleSingEdgeList, numProcs)
    eleFaceSplit, _ = splitArr(eleFaceList, numProcs)

    p = Pool(numProcs) # Set up parallel pool

    ## Construct the matrices for each sub-list
    matArr = list(p.starmap(assembleMats, zip(eleSplit, eleEdgeSplit, eleSingEdgeSplit, eleFaceSplit, startEle, repeat(singEleLists), \
                                              repeat(volP), repeat(nodeList), repeat(basisMaps), repeat(basisOrder), repeat(singOrder), repeat(basisOrder_long), repeat(singOrder_long), \
                                              repeat(quadOrder), repeat(boundaryList), repeat(simType), repeat(eleType), repeat(bc), repeat(meshStats))))
    
    ## Unpack the list of results, and sum them together
    matStruct = matArr[0]
    for ii in range(1,numProcs):
      for key in matStruct:
        matStruct[key] += matArr[ii][key]
    p.close()

  else:
    matStruct = assembleMats(eleList, eleEdgeList, eleSingEdgeList, eleFaceList, 0, singEleLists, volP, nodeList, basisMaps, \
                               basisOrder, singOrder, basisOrder_long, singOrder_long, quadOrder, boundaryList, simType, eleType, bc, meshStats)

  return matStruct

def splitArr(arr, numSplit):
  """
  Splits a Numpy array by rows into a list of smaller arrays for parallelization purposes

  Inputs:
    arr - The array to be split
    numSplit - The number of list entries to split into
  Outputs:
    arrSplit - List of portions of arr
    startEle - Start indices of each list entry
  """
  rowsPerEntry = np.floor(len(arr)/numSplit)
  arrSplit = [] # List to store the elements, seperated on a per-process basis
  startEle = [] # Marks which element the current split starts from
  for ii in range(numSplit-1):
    arrSplit.append(arr[int(ii*rowsPerEntry):int((ii+1)*rowsPerEntry)])
    startEle.append(int(ii*rowsPerEntry))
  arrSplit.append(arr[int((numSplit-1)*rowsPerEntry):]) # Ensure the last member of the split always takes the remaining elements
  startEle.append(int((numSplit-1)*rowsPerEntry))

  return arrSplit, startEle


