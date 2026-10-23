import numpy as np
import time
from heapq import heappush, heappop
from ReadAbaqusMesh import readAbq
from ReadExodusMesh import readEx
from readgmshMesh import readGmsh
from JSON_util import readGeometryJSON
from tree_cotree import minimum_spanning_tree


def initMesh(meshFile, geomFile, simType, numSubDs=1, bc="PEC"):
  """
  initMesh initializes the mesh elements from a Abaqus, Exodus, or gmsh mesh file
    Inputs:
      meshFile     - Path to the mesh file
      geomFile     - Path to the geometry configuration file
      simType      - The type of simulation to run ("E", "A", "TCS")
      numSubDs     - Number of subdomains (a value of 1 indicates to not use domain decomposition)
      bc           - Type of outer boundary condition to use (currently either "ABC" or "PEC")
      
    Outputs: 
      connLists    - Dictionary of connectivity lists between geometric entities
      meshStats    - Dictionary of statistics about the mesh
      subDLists    - Dictionary of subdomain info (if useDD == 1), else empty
      singEleLists - Dictionary of lists of nodes in each element which are singular, and their corresponding coefficients
  """
  ## Read in the mesh file information and do minor reordering
  if meshFile[-4:] == '.inp': # Abaqus file
    nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, quadList, surfStartInds, surfEleTypes, lineList, lineStartInds, nodesetList, nodesetStartInds = readAbq(meshFile)
  elif meshFile[-2:] == '.e' or meshFile[-2:] == '.g' or meshFile[-4:] == '.exo': # Exodus file
    nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, quadList, surfStartInds, surfEleTypes, lineList, lineStartInds = readEx(meshFile)
    nodesetList = np.zeros((0,), dtype=np.int32) # TODO: Add support for nodesets in exodus files
    nodesetStartInds = []
  elif meshFile[-4:] == '.msh': # GMSH files
    nodeList, tetList, wedgeList, volStartInds, volEleTypes, triList, quadList, surfStartInds, surfEleTypes, lineList, lineStartInds = readGmsh(meshFile)
    tetList += 1
    triList += 1
    nodesetList = np.zeros((0,), dtype=np.int32) # TODO: Add support for nodesets in exodus files
    nodesetStartInds = []

  if len(tetList != 0):
    tetList -= 1 # -1 to match python indexing
    tetList = np.sort(tetList, axis=1) # Sort node numbers in ascending order in each element
  if len(wedgeList) != 0:
    wedgeList -= 1 # -1 to match python indexing
    for ii in range(len(wedgeList)): # Sort node numbers in ascending order for the bottom triangle, and matching the ordering in the top triangle
      nodeOrder = np.argsort(wedgeList[ii,:3])
      wedgeList[ii,:3] = wedgeList[ii,nodeOrder]
      # Note - this doesn't necessarily lead to the nodes being in ascending order. This gets handled in the basis functions
      wedgeList[ii,3:6] = wedgeList[ii,nodeOrder+3] 
  if len(triList) != 0:
    triList -= 1
    triList = np.sort(triList, axis=1) # Sort node numbers in ascending order in each triangle
  if len(quadList) != 0:
    quadList -= 1 # Note: this can't be sorted because it can lead to the edges being incorrect
  if len(lineList) != 0:
    lineList -= 1
    lineList = np.sort(lineList, axis=1) # Sort node numbers in ascending order in each line
  if len(nodesetList) != 0:
    nodesetList -= 1

  ## Number edges
  start_time = time.time() 
  edgeList, tetEdgeList, wedgeEdgeList = assembleEdgeList(tetList, wedgeList)

  ## Read in parameters for the volumes and surfaces, and split the surfaces into seperate lists
  tetVolP, wedgeVolP, portList, portP, qubitList, qubitP, impList, impP, voltList, voltP, pecList, \
    pecStartInds, abcList, outList, outP, singEdgeList, singEdgeP, singVertList, singVertP, scale \
    = readGeometryJSON(geomFile, volStartInds, volEleTypes, triList, surfStartInds, lineList, \
                          lineStartInds, nodesetList, nodesetStartInds, bc)
  nodeList[:,1:] *= scale
  for ii in range(portP["num"]):
    portP[f"{ii}"]["width"] *= scale
  for ii in range(qubitP["num"]):
    qubitP[f"{ii}"]["width"] *= scale
  for ii in range(outP["num"]):
    if "thickness" in outP[f"{ii}"]:
      outP[f"{ii}"]["thickness"] *= scale

  ## Renumber edges to eliminate PEC edges
  start_time = time.time() 
  edgeList, tetEdgeList, wedgeEdgeList, edgeLengths = reIndex(edgeList, tetEdgeList, wedgeEdgeList, nodeList, pecList, pecStartInds)

  ## Check if there are any nodes not connected to any tets (i.e.: on impedance surface volume boundary) and eliminate them
  if len(tetList) != 0:
    totTetNodes = np.unique(tetList)
    if len(totTetNodes) != totTetNodes[-1]+1:
      curInd = 0
      for ii in range(1,len(totTetNodes)):
        nodeShift = totTetNodes[ii] - totTetNodes[ii-1]
        if nodeShift != 1:
          nodeList[curInd+1:curInd+nodeShift, 0] = -1
        curInd += nodeShift
        nodeList[curInd,0] = ii

  ## Renumber nodes to mark nodes on PEC and outer boundary
  if len(pecList) != 0: # Don't renumber if there are no internal PEC and ABCs are used
    start_time = time.time()
    newIndex = renumberNodes(nodeList[:,0].astype(np.int32), pecList, pecStartInds)
    nodeList[:,0] = newIndex

  ## Form a list of faces and corresponding connectivity between different element types
  start_time = time.time()
  tetFaceList, wedgeFaceList, faceList = assembleFaceList(tetList, wedgeList)
  faceEdgeList = associateFacesToEdges(faceList, edgeList)
  tetFaceList, wedgeFaceList, faceEdgeList, faceList = renumberFaces(tetFaceList, wedgeFaceList, faceEdgeList, faceList)
  wedgeFaceList, quadFaceList = assembleQuadFaceList(wedgeList, wedgeFaceList, edgeList)
  wedgeFaceList[:,2:][wedgeFaceList[:,2:] != -1] += len(faceList) # Increase the numbering so it doesn't overlap with the triangles

  ## Collect all of the connectivity lists into a dictionary
  connLists = {'scale': scale, 'tetList': tetList, 'tetEdgeList': tetEdgeList, 'tetFaceList': tetFaceList, 'tetVolP': tetVolP, \
               'wedgeList': wedgeList, 'wedgeEdgeList': wedgeEdgeList, 'wedgeFaceList': wedgeFaceList, 'wedgeVolP': wedgeVolP, \
               'portList': portList, 'portP': portP, 'qubitList': qubitList, 'qubitP': qubitP, "impList": impList, "impP": impP, \
               'voltList': voltList, 'voltP': voltP, 'outList': outList, 'outP': outP, \
               'pecList': pecList, 'pecStartInds': pecStartInds, 'abcList': abcList, \
               'nodeList': nodeList, 'edgeList': edgeList, 'faceList': faceList, 'faceEdgeList': faceEdgeList, \
               'quadFaceList': quadFaceList, 'singEdgeList': singEdgeList}

  ## Loop through surface types and create connectivity lists for each
  for surf in ["port", "qubit", "imp", "volt", "out"]:
    start_time = time.time()
    if len(connLists[f"{surf}List"]) != 0:
      connLists[f"{surf}EdgeList"] = assembleSurfEdgeList(edgeList, connLists[f"{surf}List"])
      if np.any(connLists[f"{surf}EdgeList"] != -1): # If the surface is on PEC, just set all faces to -1
        connLists[f"{surf}FaceList"] = surfFaceNumbering(faceList, connLists[f"{surf}List"])
      else:
        connLists[f"{surf}FaceList"] = -np.ones((len(connLists[f"{surf}List"]),), dtype=np.int32)
    else:
      connLists[f"{surf}EdgeList"], connLists[f"{surf}FaceList"] = [[], []]

  ## Number each geometric entity
  N_n = int(np.max(nodeList[:,0])) + 1
  if len(pecList) != 0:
    N_pec = len(pecStartInds) - 2
    N_n += N_pec # 1 node for each PEC, -1 for GND
  else:
    N_pec = 0
  uniqueTriEdges = np.unique(np.append(np.unique(tetEdgeList), np.unique(wedgeEdgeList[:,:6])))
  N_e = len(uniqueTriEdges[uniqueTriEdges >= 0])

  # If tree-cotree is used, update the lists to remove tree edges
  if simType == "TCS": 
    connLists["tetEdgeList"], connLists["portEdgeList"], connLists["edgeList"], N_e = \
      minimum_spanning_tree(connLists["tetEdgeList"], connLists["portEdgeList"], connLists["edgeList"], nodeList, N_e, N_n, len(pecStartInds))
  
  if len(wedgeList) != 0: # Wedges are present
    uniqueLongEdges = np.unique(wedgeEdgeList[:,6:])
    N_w_e = len(uniqueLongEdges[uniqueLongEdges >= 0]) # Count the number of unique longitudinal wedge edges
  else:
    N_w_e = 0
  N_sing_e = len(singEdgeList) # Number of singular edges
  N_f = len(faceList)
  N_q_f = len(quadFaceList)

  N_tet = len(tetList)
  N_wedge = len(wedgeList)

  l_max = np.max(edgeLengths)
  l_min = np.min(edgeLengths)
  l_avg = np.average(edgeLengths)
  l_stdev = np.std(edgeLengths)
  meshStats = {"N_n": N_n, "N_e": N_e, "N_w_e": N_w_e, "N_f": N_f, "N_q_f": N_q_f, "N_tet": N_tet, "N_wedge": N_wedge, \
               "N_sing_e": N_sing_e, "N_pec": N_pec, "l_max": l_max, "l_min": l_min, "l_avg": l_avg, "l_stdev": l_stdev}

  connLists["faceTetList"] = assembleFaceEleList(tetFaceList, N_f)
  if len(connLists["outList"]) != 0:
    connLists["outTetList"] = assembleSurfEleList(connLists["outList"], tetList, wedgeList, len(tetList))

  subDLists = []


  singEdgeNodes = np.unique(singEdgeList) # The numbers of singular nodes
  if len(singEdgeList) != 0:
    ## Number singular edges and nodes and determine which elements are line / point tetrahedra
    if len(tetList) != 0:
      singTets, singTetCoeffs, singTetMap, tetSingEdgeList \
          = numberSingTets(tetList, N_tet, singEdgeList, singEdgeP, singVertList, singVertP)
    else:
      singTets, singTetCoeffs, singTetMap, tetSingEdgeList \
          = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), \
             -np.ones((N_tet,), dtype=np.int32), -np.ones((N_tet,1), dtype=np.int32)]

    if len(wedgeList) != 0:
      singWedges, singWedgeCoeffs, singWedgeMap, wedgeSingEdgeList \
        = numberSingWedges(wedgeList, N_wedge, singEdgeList, singEdgeP, singVertList, singVertP)
    else:
      singWedges, singWedgeCoeffs, singWedgeMap, wedgeSingEdgeList \
        = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), \
            -np.ones((N_wedge,), dtype=np.int32), np.zeros((N_wedge,1), dtype=np.int32)]
  else:
    singTets, singTetCoeffs, singTetMap, tetSingEdgeList \
          = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), \
             -np.ones((N_tet,), dtype=np.int32), -np.ones((N_tet,1), dtype=np.int32)]
    singWedges, singWedgeCoeffs, singWedgeMap, wedgeSingEdgeList \
        = [np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32), \
            -np.ones((N_wedge,), dtype=np.int32), np.zeros((N_wedge,1), dtype=np.int32)]
    
  singEleLists = {'singTets': singTets, 'singTetMap': singTetMap, 'singTetCoeffs': singTetCoeffs, 'tetSingEdgeList': tetSingEdgeList,
                  'singWedgeCoeffs': singWedgeCoeffs, 'singWedgeMap': singWedgeMap, 'singWedges': singWedges, 'wedgeSingEdgeList': wedgeSingEdgeList,
                  'singEdgeP': singEdgeP, 'singVertP': singVertP, 'singEdgeNodes': singEdgeNodes}

  return connLists, meshStats, subDLists, singEleLists

def assembleEdgeList(tetList, wedgeList):
  '''
  assembleEdgeList creates and numbers an edge to node connectivity list
  Inputs:
    tetList       - array of node indices forming each tetrahedron
    wedgeList     - array of node indices forming each wedge / prism element
  Outputs:
    edgeList      - array of nodes forming each unique edge
    tetEdgeList   - array of edge numbers forming each tetrahedron
    wedgeEdgeList - array of edge numbers forming each wedge
  '''
  edgeList = np.zeros((6*len(tetList)+9*len(wedgeList),4), dtype=np.int64) # List containing all edges - unsorted
  # edgeList structure:
  #   col 0: initialized with indicator number - later replaced with edge number
  #   col 1: node 1 index
  #   col 2: node 2 index
  #   col 3: element number for that edge - wedge numbers are multiplied by -1
  tetEdgeList = -1 * np.ones((len(tetList), 6), dtype=np.int32)
  wedgeEdgeList = -1 * np.ones((len(wedgeList), 9), dtype=np.int32)
  
  ind = 0 # Current index of edgeList
  eleInd = 0 # Current element being looked at
  edgeMask = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]) # Local node indexes of local edges
  for element in tetList:
    nodes_sorted = np.sort(element[0:4]).astype(np.int64) # Contains the sorted indices of the elements of the current node
    for edgeNum in range(6):
      edgeList[ind,1:3] = nodes_sorted[edgeMask[edgeNum]]
      # Calculate indicator number to be used for sorting
      edgeList[ind,0] = (nodes_sorted[edgeMask[edgeNum][0]]+1)*nodes_sorted[edgeMask[edgeNum][1]] 
      edgeList[ind,3] = eleInd
      ind += 1
    eleInd += 1

  if len(wedgeList) != 0:
    wedgeEdgeMask = np.array([[0,1], [0,2], [1,2], [3,4], [3,5], [4,5], [0,3], [1,4], [2,5]]) 
    wedgeInd = -1
    for wedge in wedgeList:
      for edgeNum in range(9):
        # edgeList[ind,1:3] = wedge[wedgeEdgeMask[edgeNum]]
        edgeList[ind,1:3] = np.sort(wedge[wedgeEdgeMask[edgeNum]])
        # Calculate indicator number to be used for sorting
        edgeList[ind,0] = (edgeList[ind,1]+1)*edgeList[ind,2]
        edgeList[ind,3] = wedgeInd
        ind += 1
      wedgeInd -= 1

        
  edgeList = edgeList[0:ind,:] # Remove 0 entries

  # Now determine the edge indices
  indOrder = np.argsort(edgeList[:,0]) # Sorted indicator numbers
  sortedEdgeList = edgeList[indOrder,:] # Sorted version of the edgeList
  prevInd = -1 # Indicator of the prior edge
  indNodes = [] # List of node indices associated with the current indicator
  uniqueEdgeInds = [] # Indices of unique edges
  globNum = 0 # Tracks the current global edge number
  ind = 0 # Current index for appending the list
  for edge in sortedEdgeList:
    curInd_edgeList = indOrder[ind]
    if edge[0] == prevInd: # check if edge has a duplicate index
       # the Row index of the current edge in edgeList
      numInGroup = np.where(np.all(edge[1:3] == indNodes, axis=1))[0] # check if the edge has already been seen
      if len(numInGroup) == 0: # If no, add it to the list, and create a new number for it
        indNodes.append(edge[1:3])
        edgeNum = globNum + len(indNodes) - 1
        edgeList[curInd_edgeList, 0] = edgeNum
        uniqueEdgeInds.append(curInd_edgeList)
      else: # If yes, select the corresponding number
        edgeNum = globNum + numInGroup[0]
        edgeList[curInd_edgeList,0] = edgeNum

      if edge[3] >= 0: # Tetrahedral element
        # Update tetEdgeList with the new edge number
        col = np.argmax(np.all(tetList[edge[3]][edgeMask] == edgeList[curInd_edgeList, 1:3], axis=1)) # Find the correct column to add to
        tetEdgeList[edge[3], col] = edgeNum
      elif edge[3] < 0: # Wedge element
        wedgeInd = -edge[3]-1
        col = np.argmax(np.all(np.sort(wedgeList[wedgeInd][wedgeEdgeMask]) == edgeList[curInd_edgeList, 1:3], axis=1)) # Find the correct column to add to
        wedgeEdgeList[wedgeInd, col] = edgeNum
      
    else:
      globNum += len(indNodes) # Update the global edge number
      indNodes = [edge[1:3]] # Reset the list of nodes for this indicator, starting with this edge

      # Update edgeList and record the indicator and row number
      edgeList[curInd_edgeList, 0] = globNum # Update the current edge number
      prevInd = edge[0] # Record the current indicator
      uniqueEdgeInds.append(curInd_edgeList)

      # Update tetEdgeList with the new edge number
      if edge[3] >= 0: # Tetrahedral element
        col = np.argmax(np.all(tetList[edge[3]][edgeMask] == edgeList[curInd_edgeList, 1:3], axis=1)) # Find the correct column to add to
        tetEdgeList[edge[3], col] = globNum
      elif edge[3] < 0: # Wedge element
        wedgeInd = -edge[3]-1
        col = np.argmax(np.all(np.sort(wedgeList[wedgeInd][wedgeEdgeMask]) == edgeList[curInd_edgeList, 1:3], axis=1)) # Find the correct column to add to
        wedgeEdgeList[wedgeInd, col] = globNum
    ind += 1

  uniqueEdgeList = edgeList[uniqueEdgeInds, 1:3].astype(np.int32)

  return uniqueEdgeList, tetEdgeList, wedgeEdgeList

def buildNodeAdjList(edgeList):
  """
  Creates an adjacency list which gives node-to-node connectivity

  Inputs:
    edgeList - Arary of the two nodes forming each edge

  Outputs:
    adj_list - Array of nodes contacting each node via an edge
  """
  numNodes = np.max(edgeList) + 1
  adj_list = [set() for _ in range(numNodes)]

  # Loop filling in the adjacency list
  for ii in range(len(edgeList)):
    adj_list[edgeList[ii,0]].add(edgeList[ii,1])
    adj_list[edgeList[ii,1]].add(edgeList[ii,0])

  return adj_list

# @jit(nopython=True)
def renumberNodes(nodeNums, pecList, pecStartInds):
  """
  renumberNodes updates the node numbers to mark any PEC nodes 
  (using negative numbers, with unique negatives for each unique PEC)
  
  Inputs:
    nodeNums     - List of the original node numbers
    pecList      - list of node numbers forming triangles along each PEC
    pecStartInds - List of row numbers in pecList for each unique PEC
  Outputs:
    newIndex     - vector of new node indices (which becomes the new first column of 
                   nodeList).
  """
  nodesOnBoundary = [] # List of nodes on each pec boundary
  for ii in range(len(pecStartInds)-1):
    nodesOnBoundary.append(np.unique(pecList[pecStartInds[ii]:pecStartInds[ii+1]]))
  index = np.zeros(len(nodeNums), dtype=np.int32) # For marking the PEC nodes

  for ii in range(len(nodeNums)):
    for jj in range(len(nodesOnBoundary)):
      if(nodeNums[ii] in nodesOnBoundary[jj]):
        index[ii] = int(-1*(jj+1)) # Use negative numbers to mark which PEC the nodes belong to
        break
      else:
        index[ii] = int(nodeNums[ii])

  newIndex = np.zeros(len(index), dtype=np.int32)
  newVal = 0
  for idx in range(len(index)):
    val = index[idx]
    if(val < 0):
      newIndex[idx] = val
    else:
      newIndex[idx] = newVal
      newVal += 1

  return newIndex

def reIndex(edgeList, tetEdgeList, wedgeEdgeList, nodeList, pecList, pecStartInds):
  '''
  reIndex adjusts the numbering of edges in tetEdgeList and removes edges from
  edgeList which lie on PEC. The PEC edges in tetEdgeList are marked with "-1."
  All other edges are renumbered to maintain continuity.
  Inputs:
    edgeList           - Nx2 array where each row lists the two nodes defining an edge
    tetEdgeList        - list of edge indices for each tetrahedral element
    wedgeEdgeList      - list of edge indices for each wedge / triangular prism element
    nodeList           - list of global node numbers and their coordinates
    pecList            - list of node numbers forming triangles along each PEC
    pecStartInds       - Indices where each new PEC begins
  Outputs:
    updatedEdgeList    - updated version of edgeList with PEC edges removed
    tetEdgeList        - updated version of tetEdgeList with PEC edges marked as "-1" and
                         the remaining nodes renumbered
    wedgeEdgeList      - updated version of wedgeEdgeList with PEC edges marked as "-1" and
                         the remaining nodes renumbered
    edgeLengths        - Array containing the lengths of all edges
  '''

  # Map edges on each PEC surface to a unique negative ID
  pecEdgeList = assembleSurfEdgeList(edgeList, pecList)
  edgeMap = np.zeros((len(edgeList),), dtype=np.int32) # Tracks which edges are connected to which PEC (negatives for PEC)
  for pecNum in range(len(pecStartInds)-1):
    indRange = np.arange(pecStartInds[pecNum], pecStartInds[pecNum+1])
    curPECEdges = np.unique(pecEdgeList[indRange])
    edgeMap[curPECEdges] = -(pecNum+1)

  updatedEdgeList = edgeList[edgeMap == 0] # Eliminate the PEC edges from the edge list
  
  # Renumber the non-PEC edges
  edgeLengths = np.zeros((len(updatedEdgeList),))
  curIndex = 0
  for ii in range(len(edgeMap)):
    if edgeMap[ii] == 0:
      # If the edge isn't on PEC, assign it a new index
      edgeMap[ii] = curIndex
      edgeLengths[curIndex] = np.linalg.norm(nodeList[updatedEdgeList[curIndex,0],1:] - nodeList[updatedEdgeList[curIndex,1],1:])
      curIndex += 1

  # Update tetEdgeList to use new indices for remaining edges
  for ii in range(tetEdgeList.shape[0]):
    for jj in range(tetEdgeList.shape[1]):
      oldIndex = tetEdgeList[ii, jj]
      if oldIndex >= 0:  # Only remap valid (non-boundary) edges
        tetEdgeList[ii, jj] = edgeMap[oldIndex]

  # Also update wedgeEdgeList
  for ii in range(wedgeEdgeList.shape[0]):
    for jj in range(wedgeEdgeList.shape[1]):
      oldIndex = wedgeEdgeList[ii, jj]
      if oldIndex >= 0:  # Only remap valid (non-boundary) edges
        wedgeEdgeList[ii, jj] = edgeMap[oldIndex]

  return updatedEdgeList, tetEdgeList, wedgeEdgeList, edgeLengths


def assembleSurfEdgeList(edgeList, surfList):
  '''
  assembleSurfEdgeList creates a list of edges along each surface
  Inputs:
    edgeList      - Nx2 array where each row lists the two nodes defining an edge
    surfList      - List of node numbers for triangles on each surface
  Outputs:
    surfEdgeList  - List of node numbers for triangles on each surface
  '''

  # Initialize the list to hold edge lists for each surface
  surfEdgeList = np.zeros((len(surfList),3), dtype=np.int32)

  # Sort the edge list by the first node to accelerate the search
  edgeSort = np.argsort(edgeList[:,0])
  nodeSortEdgeList = edgeList[edgeSort]
      
  # Iterate over each triangle in portList
  for ii, triangle in enumerate(surfList):
    # Extract the nodes and the port number
    n1, n2, n3 = triangle
        
    # Find the edges by pairing the nodes (make sure to consider unique edges)
    edges = [[n1, n2], [n1, n3], [n2, n3]]
            
    # Find the corresponding edge number in edgeList for each edge
    for jj, edge in enumerate(edges):
      # valid_edges = np.where(edgeList[:,0] == edge[0])[0]
      # edge_idx = valid_edges[np.where(edgeList[valid_edges,1] == edge[1])[0]]
      minRange = np.searchsorted(nodeSortEdgeList[:,0], edge[0])
      maxRange = np.searchsorted(nodeSortEdgeList[:,0], edge[0]+1)
      edge_idx = edgeSort[minRange + np.where(nodeSortEdgeList[minRange:maxRange,1] == edge[1])[0]]
      # edge_idx = np.where(np.all(edgeList == edge, axis=1))[0]
      if edge_idx.size > 0:
        surfEdgeList[ii,jj] = edge_idx[0]
      else:
        # If no match is found, something went wrong
        surfEdgeList[ii,jj] = -1  # Or handle as appropriate
    
  return surfEdgeList

def associateFacesToEdges(
    face_node_list: np.ndarray, 
    edge_node_list: np.ndarray,
):
    """
    Creates a face to edge connectivity list
    Inputs:
        face_node_list - array of 3 nodes for each face
        edge_node_list - array of 2 nodes for each edge
    Outputs:
        face_edge_list - array of 3 edges for each face
    """
    ## Generate Nodes -> Edge Associative Array (dict) ##
    node_edge_dict: dict[tuple[int, int], int] = dict()
    for eidx, nodes in enumerate(edge_node_list):
        nodes = tuple(nodes)
        if nodes[0] > nodes[1]: nodes = (nodes[1], nodes[0])
        node_edge_dict[nodes] = eidx
    
    ## Generate Face -> Edge Associative Array (ndarray) ##
    face_edge_list: np.ndarray = np.zeros(face_node_list.shape, dtype=np.int32)
    for fidx, face_nodes in enumerate(face_node_list):
        f_e_nodes = (
            (face_nodes[0], face_nodes[1]),
            (face_nodes[0], face_nodes[2]),
            (face_nodes[1], face_nodes[2]),
        )
        for nidx, e_nodes in enumerate(f_e_nodes):
            try:
                face_edge_list[fidx][nidx] = node_edge_dict[e_nodes]
            except KeyError:  # Edge along PEC, replace with -1
                face_edge_list[fidx][nidx] = -1
                
    return face_edge_list

def assembleFaceList(tetList, wedgeList):
    ''' 
    Creates and numbers face to node and face to element connectivity lists for 
    triangular faces in each element type
    Inputs:
      tetList        - array of tetrahedral elements and their corresponding nodes
      wedgeList      - array of wedge elements and their corresponding nodes
    Outputs:
      tetFaceList    - array of 4 faces for each tetrahedral element
      wedgeFaceList  - array of 2 triangular faces for each wedge element
      faceNodeList   - array of 3 nodes for each face
    '''

    N_tet = len(tetList)
    N_wedge = len(wedgeList)

    ## Number Faces ##
    tet_faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    wedge_faces = ((0, 1, 2), (3, 4, 5))
    _indicator = lambda x, y, z: x * y + x * z + y * z  # Less logical choice, procudes smaller numbers (~200x smaller for small mesh)
    raw_face_list = list()
    for eidx, tet in enumerate(tetList):
      for xidx, yidx, zidx in tet_faces:
        face = tuple((tet[xidx], tet[yidx], tet[zidx]))
        val = _indicator(np.int64(face[0]), np.int64(face[1]), np.int64(face[2]))
        heappush(raw_face_list, (val, face, eidx))

    
    for eidx, wedge in enumerate(wedgeList):
      for xidx, yidx, zidx in wedge_faces:
        face = tuple(np.sort([wedge[xidx], wedge[yidx], wedge[zidx]])) # Sort these in case they're out of order (can occur on the top face)
        val = _indicator(np.int64(face[0]), np.int64(face[1]), np.int64(face[2]))
        heappush(raw_face_list, (val, face, eidx+N_tet))
        
    compressed_face_list = list()
    while raw_face_list:
      indicator_list = [heappop(raw_face_list),]
      try: 
        while raw_face_list[0][0] == indicator_list[0][0]: indicator_list.append(heappop(raw_face_list))
      except IndexError: pass 
      
      # Collision Handling
      face_ele_dict = dict()
      for _, face, eidx in indicator_list:
        if face not in face_ele_dict:
            face_ele_dict[face] = list()
        face_ele_dict[face].append(eidx)
      
      for face, edixs in face_ele_dict.items():
        compressed_face_list.append((face, edixs))
    
    ## Construct face arrays ## 
    N_f = len(compressed_face_list)
    
    tetFaceList  = np.zeros((N_tet,4), dtype=np.int32)
    wedgeFaceList  = np.zeros((N_wedge,5), dtype=np.int32)
    faceNodeList = np.zeros((N_f,3), dtype=np.int32)
    
    ele_face_dict = dict()
    for fidx, (face, edixs) in enumerate(compressed_face_list):
      faceNodeList[fidx] = face 
      
      for eidx in edixs:
        if eidx not in ele_face_dict:
            ele_face_dict[eidx] = list()
        ele_face_dict[eidx].append(fidx)
            
    for eidx, fidxs in ele_face_dict.items():
      if eidx < N_tet: # 4 faces for tetrahedral elements
        tetFaceList[eidx] = fidxs
      else: # 2 triangular faces for wedge elements
        wedgeFaceList[eidx-N_tet, :2] = fidxs
        
    return tetFaceList, wedgeFaceList, faceNodeList

def renumberFaces(tetFaceList, wedgeFaceList, faceEdgeList, faceNodeList):
  """
  A function to renumber the triangular faces to eliminate faces lying on PEC

  Inputs:
      tetFaceList   - Provides tetrahedron to face connectivity
      wedgeFaceList - Provides wedge to face connectivity (only for triangular faces)
      faceEdgeList  - Provides face to edge connectivity
      faceNodeList  - Provides face to node connectivity
  Outputs:
      tetFaceList   - Faces lying on PEC are changed to -1, other faces are renumbered
      wedgeFaceList - Faces lying on PEC are changed to -1, other faces are renumbered
      faceEdgeList  - Rows for PEC faces are removed
      faceNodeList  - Rows for PEC faces are removed
  """
  ## Set up new face numbers
  N_f = len(faceEdgeList)
  newFaceInds = np.zeros((N_f,))
  cnt = 0
  for ii in range(N_f):
    if np.all(faceEdgeList[ii] == -1):
      newFaceInds[ii] = -1
    else:
      newFaceInds[ii] = cnt
      cnt += 1

  ## Renumber the element to face lists
  for ii in range(len(tetFaceList)):
    for jj in range(4):
      tetFaceList[ii,jj] = newFaceInds[tetFaceList[ii,jj]]
  for ii in range(len(wedgeFaceList)):
    for jj in range(2):
      wedgeFaceList[ii,jj] = newFaceInds[wedgeFaceList[ii,jj]]

  ## Remove PEC faces from the face edge and node lists
  nonPECFaces = newFaceInds != -1
  faceEdgeList = faceEdgeList[nonPECFaces]
  faceNodeList = faceNodeList[nonPECFaces]

  return tetFaceList, wedgeFaceList, faceEdgeList, faceNodeList

def assembleQuadFaceList(wedgeList, wedgeFaceList, edgeList):
  ''' 
  Creates and numbers face to node connectivity lists for quadrilaterals
  Inputs:
    wedgeList      - array of wedge elements and their corresponding nodes
    wedgeFaceList  - array containing wedge face indices, currently with only triangular faces
    edgeList       - array of nodes forming each edge
  Outputs:
    wedgeFaceList  - contains all 5 face indices for each wedge (2 tris, 3 quads)
    quadFaceList   - array of 4 nodes forming each quadrilateral face
  '''

  ## Number Faces ##
  wedge_faces = ((0, 1, 3, 4), (0, 2, 3, 5), (1, 2, 4, 5))
  _indicator = lambda w, x: (w*x)  # I'm using this indicator that only depends on the bottom nodes 
                                   # to avoid issues with the top nodes of a wedge being out of order.
  raw_face_list = list()
  for eidx, wedge in enumerate(wedgeList):
    for widx, xidx, yidx, zidx in wedge_faces:
      face = tuple((wedge[widx], wedge[xidx], wedge[yidx], wedge[zidx]))
      val = _indicator(np.int64(face[0]), np.int64(face[1]))
      heappush(raw_face_list, (val, face, eidx))
      
  compressed_face_list = list()
  while raw_face_list:
    indicator_list = [heappop(raw_face_list),]
    try: 
      while raw_face_list[0][0] == indicator_list[0][0]: indicator_list.append(heappop(raw_face_list))
    except IndexError: pass 
    
    # Collision Handling
    face_ele_dict = dict()
    for _, face, eidx in indicator_list:
      if face not in face_ele_dict:
          face_ele_dict[face] = list()
      face_ele_dict[face].append(eidx)
    
    for face, edixs in face_ele_dict.items():
      compressed_face_list.append((face, edixs))
  
  ## Construct face arrays ## 
  N_f = len(compressed_face_list)
  quadFaceList = np.zeros((N_f,4), dtype=np.int32)
  
  ele_face_dict = dict()
  for fidx, (face, edixs) in enumerate(compressed_face_list):
    quadFaceList[fidx] = face 
    
    for eidx in edixs:
      if eidx not in ele_face_dict:
          ele_face_dict[eidx] = list()
      ele_face_dict[eidx].append(fidx)
          
  for eidx, fidxs in ele_face_dict.items():
    wedgeFaceList[eidx, 2:] = fidxs

  node_edge_dict: dict[tuple[int, int], int] = dict()
  for eidx, nodes in enumerate(edgeList):
      nodes = tuple(nodes)
      if nodes[0] > nodes[1]: nodes = (nodes[1], nodes[0])
      node_edge_dict[nodes] = eidx

  ## Generate Face -> Edge Associative Array (ndarray) ##
  quadFaceEdgeList: np.ndarray = np.zeros(quadFaceList.shape, dtype=np.int32)
  for fidx, face_nodes in enumerate(quadFaceList):
    f_e_nodes = (
      (face_nodes[0], face_nodes[1]),
      (face_nodes[0], face_nodes[2]),
      (face_nodes[1], face_nodes[3]),
      (face_nodes[2], face_nodes[3])
    )
    for nidx, e_nodes in enumerate(f_e_nodes):
      try:
        quadFaceEdgeList[fidx][nidx] = node_edge_dict[e_nodes]
      except KeyError:  # Edge along PEC, replace with -1
        quadFaceEdgeList[fidx][nidx] = -1

  ## Renumber arrays to remove PEC entries ##
  newFaceInds = np.zeros((N_f,))
  cnt = 0
  for ii in range(N_f):
    if np.all(quadFaceEdgeList[ii] == -1):
      newFaceInds[ii] = -1 # If all edges lie on PEC, mark the face to be eliminated
    else:
      newFaceInds[ii] = cnt
      cnt += 1

  for ii in range(len(wedgeFaceList)):
    for jj in range(3):
      wedgeFaceList[ii,jj+2] = newFaceInds[wedgeFaceList[ii,jj+2]]

  # Remove PEC faces from the face node list
  nonPECFaces = newFaceInds != -1
  quadFaceList = quadFaceList[nonPECFaces]
        
  return wedgeFaceList, quadFaceList

def surfFaceNumbering(faceList, surfList):
    surfFaceList = np.zeros((len(surfList),), dtype=surfList.dtype)
    faceSort = np.argsort(faceList[:,0])
    faceListSorted = faceList[faceSort]
    for ii in range(len(surfList)):
      # surfFaceList[ii] = np.where(np.all(surfList[ii] == faceList, axis=1))[0][0]
      minRange = np.searchsorted(faceListSorted[:,0], surfList[ii,0])
      maxRange = np.searchsorted(faceListSorted[:,0], surfList[ii,0]+1)
      try:
        surfFaceList[ii] = faceSort[minRange + np.where(np.all(surfList[ii,1:] == faceListSorted[minRange:maxRange,1:], axis=1))[0][0]]
      except: # If the face isn't connected to a tet anywhere, set its index to -1
        surfFaceList[ii] = -1

    return surfFaceList

def assembleFaceEleList(eleFaceList, N_f):
  """
  Takes a list of faces forming each element (eleFaceList) and returns a set 
  of elements containing each face (faceEleList)

  Inputs:
    eleFaceList - List of traingular faces forming each element (tets or wedges/prisms)
    N_f         - Number of triangular faces in the geometry

  Outputs:
    faceEleList - List of elements containing each of the faces
  """

  faceEleList = -1*np.ones((N_f, 2), dtype=np.int32)
  for eleInd, faces in enumerate(eleFaceList):
    for face in faces:
      if face == -1:
        continue
      if faceEleList[face,0] == -1:
        faceEleList[face,0] = eleInd
      else:
        faceEleList[face,1] = eleInd

  return faceEleList

def assembleSurfEleList(surfList, tetList, wedgeList, N_t):
  """
  Obtains the set of 3D elements (tets or wedges/prisms) connected to faces on a surface

  Inputs:
    surfList  - Node numbers of each face on the surface
    tetList   - Node numbers of each tetrahedron in the mesh
    wedgeList - Node numbers of each wedge/prism in the mesh
    N_t       - Number of tetrahedrons in the mesh

  Outputs:
    surfEleList - List of elements touching each face
  """

  surfEleList = -np.ones((len(surfList), 2), dtype=np.int32) # Second column is -1 for the cases where there's only 1 element
  for ii, surfEle in enumerate(surfList):
    nodeTets = np.where(tetList == surfEle[0])[0]
    if len(nodeTets) > 0:
      edgeTets = nodeTets[np.where(tetList[nodeTets] == surfEle[1])[0]]
      if len(edgeTets) > 0:
        surfTets = edgeTets[np.where(tetList[edgeTets] == surfEle[2])[0]]
      else:
        surfTets = []
    else:
      surfTets = []
    
    nodeWedges = np.where(wedgeList == surfEle[0])[0]
    if len(nodeWedges) > 0:
      edgeWedges = nodeWedges[np.where(wedgeList[nodeWedges] == surfEle[1])[0]]
      if len(edgeWedges) > 0:
        surfWedges = edgeWedges[np.where(wedgeList[edgeWedges] == surfEle[2])[0]]
      else:
        surfWedges = []
    else:
      surfWedges = []
    

    if len(surfTets) + len(surfWedges) == 2: # Surface borders two element
      if len(surfTets) == 2: # Both are tets
        surfEleList[ii] = surfTets
      elif len(surfWedges) == 2: # Both are wedges
        surfEleList[ii] = surfWedges + N_t
      else: # Mixed
        surfEleList[ii, 0] = surfTets[0]
        surfEleList[ii, 1] = surfWedges[0] + N_t
    elif len(surfTets) + len(surfWedges) == 1: # Surface borders only one element - on PEC
      if len(surfTets) == 1:
        surfEleList[ii, 0] = surfTets[0]
      elif len(surfWedges) == 1:
        surfEleList[ii, 1] = surfWedges[0] + N_t

  return surfEleList

def assembleNodeTetList(tetList):
  """
  Forms a mapping between each node and the tetrahedrons that node acts as a vertex of

  Inputs:
    tetList     - list of nodes forming each tet

  Outputs:
    nodeTetList - list of tets containing each node
  """

  nodeTetList = [np.zeros((0,), dtype=np.int32) for _ in range(np.max(tetList) + 1)]
  for tetInd, tet in enumerate(tetList):
    for node in tet:
      nodeTetList[node] = np.append(nodeTetList[node], tetInd)

  return nodeTetList


def numberSingTets(tetList, N_tet, singEdgeList, singEdgeP, singVertList, singVertP):
  """
  Numbers the singular nodes and edges forming each tetrahedron

  Inputs:
    tetList         - Array containing the four nodes forming each tetrahedron in the mesh
    N_tet           - Total number of tetrahedra in the mesh
    singEdgeList    - Array containing the nodes forming each edge along a singular feature
    singEdgeP       - Dictionary containing information about the singular edges (coefficients and groupings)
    singVertList    - Array containing individual nodes along a singular feature (distinct from the edges above)
    singVertP       - Dictionary containing information about the singular nodes (coefficients and groupings)

  Outputs:
    singTets        - Vector containing tet numbers with singular features (nodes or edges)
    singTetCoeffs   - Coefficients associated with each node in each of the singular tetrahedrons
    singTetMap      - Mapping between tetrahedron numbers and the numbering in the above lists. 
                      Tetrahedrons with no singular nodes are marked with -1.
    tetSingEdgeList - List of singular edges and their coefficients in each tetrahedron
  """
  # Start by numbering the nodes
  singEdgeNodes = np.unique(singEdgeList)
  singEdgeStartInds = np.array([singEdgeP[f"{ii}"]["startInd"] for ii in range(singEdgeP["num"])])
  singEdgeCoeffs = np.array([singEdgeP[f"{ii}"]["coeff"] for ii in range(singEdgeP["num"])])
  singVertNodes = singVertList
  singVertStartInds = np.array([singVertP[f"{ii}"]["startInd"] for ii in range(singVertP["num"])])
  singVertCoeffs = np.array([singVertP[f"{ii}"]["coeff"] for ii in range(singVertP["num"])])

  ## Now mark which nodes and edges of each tetrahedron are singular
  singTets, singTetCoeffs = [[], []]
  tetSingEdgeList = -np.ones((N_tet,6), dtype=np.int32) # Singular edge numbers forming each tet
  edgeNums = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]], dtype=np.int32)
  numEdges = len(edgeNums)
  faceNums = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=np.int32)
  numFaces = len(faceNums)
  for eleInd in range(len(tetList)):
    # Determine which of the nodes are members of singular edges (and what the corresponding coefficients are)
    curSingCoeffs = np.zeros((len(tetList[eleInd],)), dtype=np.float64)
    for ii in range(len(tetList[eleInd])):
      singNodeInd = np.where(tetList[eleInd,ii] == singEdgeNodes)[0]
      if len(singNodeInd) != 0:
        # Check what line this is a part of to determine its coefficient
        singEdgeNum = 0
        while singEdgeNum < singEdgeP["num"] and singNodeInd >= singEdgeStartInds[singEdgeNum]:
          singEdgeNum += 1
        curSingCoeffs[ii] = singEdgeCoeffs[singEdgeNum-1]

    # Check if any of the nodes are also singular vertices (i.e.: they have a different coefficient)
    for ii in range(len(tetList[eleInd])):
      singNodeInd = np.where(tetList[eleInd,ii] == singVertNodes)[0]
      if len(singNodeInd) != 0:
        # Check what line this is a part of to determine its coefficient
        singNodesetNum = 0
        while singNodesetNum < singVertP["num"] and singNodeInd >= singVertStartInds[singNodesetNum]:
          singNodesetNum += 1
        curSingCoeffs[ii] = singVertCoeffs[singNodesetNum-1]

    numSingNodes = np.sum(curSingCoeffs != 0) # Number of singular nodes in the tet

    if numSingNodes != 0: # If the tet is singular, mark it as so
      singTets.append(eleInd)
      singTetCoeffs.append(curSingCoeffs)

      # Mark which edges are singular
      for ii, edge in enumerate(edgeNums):
        if curSingCoeffs[edge[0]] != 0 and curSingCoeffs[edge[1]] != 0:
          # If both nodes of the edge are singular, check if this is a singular edge and mark it as so
          singEdgeInd = np.where(np.all(tetList[eleInd][edge] == singEdgeList, axis=1))[0]
          if len(singEdgeInd) != 0:
            tetSingEdgeList[eleInd, ii] = singEdgeInd[0]

  singTets = np.array(singTets, dtype=np.int32)
  singTetCoeffs = np.array(singTetCoeffs, dtype=np.float64)

  singTetMap = -np.ones((len(tetList),), dtype=np.int32)
  singTetMap[singTets] = np.arange(len(singTets))

  return singTets, singTetCoeffs, singTetMap, tetSingEdgeList

def numberSingWedges(wedgeList, N_wedge, singEdgeList, singEdgeP, singVertList, singVertP):
  """
  Numbers the singular nodes and edges forming each wedge/prism element

  Inputs:
    wedgeList         - Array containing the four nodes forming each wedge in the mesh
    N_tet             - Total number of wedges in the mesh
    singEdgeList      - Array containing the nodes forming each edge along a singular feature
    singEdgeP         - Dictionary containing information about the singular edges (coefficients and groupings)
    singVertList      - Array containing individual nodes along a singular feature (distinct from the edges above)
    singVertP         - Dictionary containing information about the singular nodes (coefficients and groupings)

  Outputs:
    singWedges        - Vector containing wedge numbers with singular features (nodes or edges)
    singWedgeCoeffs   - Coefficients associated with each node in each of the singular wedges
    singWedgeMap      - Mapping between wedge numbers and the numbering in the above lists. 
                        Wedges with no singular nodes are marked with -1.
    wedgeSingEdgeList - List of singular edges and their coefficients in each wedge
  """
  # Start by numbering the nodes
  singEdgeNodes = np.unique(singEdgeList)
  singEdgeStartInds = np.array([singEdgeP[f"{ii}"]["startInd"] for ii in range(singEdgeP["num"])])
  singEdgeCoeffs = np.array([singEdgeP[f"{ii}"]["coeff"] for ii in range(singEdgeP["num"])])
  singVertNodes = singVertList
  singVertStartInds = np.array([singVertP[f"{ii}"]["startInd"] for ii in range(singVertP["num"])])
  singVertCoeffs = np.array([singVertP[f"{ii}"]["coeff"] for ii in range(singVertP["num"])])

  ## Now mark which nodes and edges of wedges are singular
  singWedges, singWedgeCoeffs = [[], []]
  wedgeSingEdgeList = -np.ones((N_wedge,9), dtype=np.int32) # Singular edge numbers forming each wedge
  edgeNums = np.array([[0, 1], [0, 2], [1, 2], [3, 4], [3, 5], [4, 5], [0, 3], [1, 4], [2, 5]], dtype=np.int32)
  # numEdges = len(edgeNums)
  # faceNums = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
  # numFaces = len(faceNums)
  # quadFaceNums = np.array([[0, 1, 3, 4], [0, 2, 3, 5], [1, 2, 4, 5]], dtype=np.int32)
  # numQuadFaces = len(quadFaceNums)
  for eleInd in range(len(wedgeList)):
    # Determine which of the nodes are members of singular edges (and what the corresponding coefficients are)
    singCoeffs = np.zeros((len(wedgeList[eleInd],)), dtype=np.float64)
    for ii in range(len(wedgeList[eleInd])):
      singNodeInd = np.where(wedgeList[eleInd,ii] == singEdgeNodes)[0]
      if len(singNodeInd) != 0:
        # Check what line this is a part of to determine its coefficient
        singEdgeNum = 0
        while singEdgeNum < singEdgeP["num"] and singNodeInd >= singEdgeStartInds[singEdgeNum]:
          singEdgeNum += 1
        singCoeffs[ii] = singEdgeCoeffs[singEdgeNum-1]

    # Check if any of the nodes are also singular vertices (i.e.: they have a different coefficient)
    for ii in range(len(wedgeList[eleInd])):
      singNodeInd = np.where(wedgeList[eleInd,ii] == singVertNodes)[0]
      if len(singNodeInd) != 0:
        # Check what line this is a part of to determine its coefficient
        singNodesetNum = 0
        while singNodesetNum < singVertP["num"] and singNodeInd >= singVertStartInds[singNodesetNum]:
          singNodesetNum += 1
        singCoeffs[ii] = singVertCoeffs[singNodesetNum-1]

    numSingNodes = np.sum(singCoeffs != 0) # Number of singular nodes in the wedge
    if numSingNodes != 0: # If the wedge is singular, mark it as so
      singWedges.append(eleInd)
      singWedgeCoeffs.append(singCoeffs)

      # Mark which edges are singular
      for edgeInd, edge in enumerate(edgeNums):
        if singCoeffs[edge[0]] != 0 and singCoeffs[edge[1]] != 0:
          # If both nodes of the edge are singular, check if this is a singular edge and mark it as so
          singEdgeInd = np.where(np.all(np.sort(wedgeList[eleInd][edge]) == singEdgeList, axis=1))[0] # Sort the nodes bc top nodes can be out of order
          if len(singEdgeInd) != 0:
            wedgeSingEdgeList[eleInd, edgeInd] = singEdgeInd[0]

  singWedges = np.array(singWedges, dtype=np.int32)
  singWedgeCoeffs = np.array(singWedgeCoeffs, dtype=np.float64)

  singWedgeMap = -np.ones((N_wedge,), dtype=np.int32)
  singWedgeMap[singWedges] = np.arange(len(singWedges))

  return singWedges, singWedgeCoeffs, singWedgeMap, wedgeSingEdgeList