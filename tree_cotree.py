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

import pickle
import numpy as np
import matplotlib.pyplot as plt



def minimum_spanning_tree(eleEdgeList, portEdgeList, edgeList, nodeList, N_e, N_n, numPEC):
    """
    A function to create an MST from a list of edges

    Inputs:
    - eleEdgeList  - List of edge numbers forming each tetrahedron
    - portEdgeList - List of edges forming each triangle in each port
    - edgeList     - List of nodes forming each edge
    - nodeList     - List containing coordinates of each node
    - N_e          - Number of edges
    - N_n          - Number of nodes
    - numPEC       - Number of PECs in the geometry

    Outputs:
    Updated versions of the above. Specifically:
    - eleEdgeList  - Updated to account for new edge numberings and put -1 for any edges not in the updated edgeList
    - edgeList     - Updated to only include edges in the MST
    - N_e          - Updated to reflect the edges in the MST

    Author: Niko Detwiler
    """
    # First step is to convert the edgeList to an adjacency list
    # This step is necessary for this algorithm to run in linear time
    adj_list = dict()

    for boundaryNum in range(numPEC): # Treat each PEC as a single node
        adj_list[-1*(boundaryNum+1)] = [set(), [], False]
    for nodeNum in range(np.max(edgeList)+1):
        adj_list[nodeNum] = [set(), [], False]
    # adj_list is a dictionary, with each key being the node number.
    # The first index of each value in adj_list is a set of all connected nodes (helps to build adj_list fast)
    # Second index is a list of lists, in this format [connected node, edge number]
    # Third index is if the node has been visited yet or not

    # Loop through the edges to determine which lie on a lumped port
    edgeFlags = np.zeros(len(edgeList), dtype=np.bool_)
    portEdgeInds = np.unique(portEdgeList)
    for ii in range(len(edgeList)):
        if ii in portEdgeInds:
            edgeFlags[ii] = True

    # Loop filling in the adjacency list
    for ii in range(len(edgeList)):
        if not edgeFlags[ii]: # Make sure the edge isn't on a lumped port or ABC
            if nodeList[edgeList[ii,0],0] >= 0: # Not on PEC - add to regular nodes
                adj_list[edgeList[ii,0]][0].add(edgeList[ii,1])
                adj_list[edgeList[ii,0]][1].append([edgeList[ii,1], ii])
            else: # On PEC - add to the PEC entries
                adj_list[int(nodeList[edgeList[ii,0],0])][0].add(edgeList[ii,1])
                adj_list[int(nodeList[edgeList[ii,0],0])][1].append([edgeList[ii,1], ii])

            if nodeList[edgeList[ii,1],0] >= 0: # Same as above, but for the second node
                adj_list[edgeList[ii,1]][0].add(edgeList[ii,0])
                adj_list[edgeList[ii,1]][1].append([edgeList[ii,0], ii])
            else:
                adj_list[int(nodeList[edgeList[ii,1],0])][0].add(edgeList[ii,0])
                adj_list[int(nodeList[edgeList[ii,1],0])][1].append([edgeList[ii,0], ii])

    # Set to determine which PEC regions have been reached
    pec_reached = set()
    # Boolean vector marking edges that are part of the MST
    tree_edges = np.zeros(N_e, dtype=np.bool_)
    
    # Determine the PEC with the largest number of connected nodes to ensure full connectivity
    maxBoundVal = len(adj_list[-1][1])
    startNode = -1
    for boundaryNum in range(1,numPEC):
        newBoundVal = len(adj_list[-1*(boundaryNum+1)][1])
        if newBoundVal > maxBoundVal:
            maxBoundVal = newBoundVal
            startNode = -1*(boundaryNum+1)

    # bfs_queue is the queue which will be doing most of the work of graph traversal.
    # Mostly equivalent to the way "Dirich" works in the pseudocode
    bfs_queue = [startNode]
    adj_list[startNode][2] = True
    pec_reached.add(startNode)
    # Main loop
    while len(bfs_queue) != 0:
        # Get first item in queue
        current_node = bfs_queue.pop()
        # Loop for each connected node
        for edge_ls in adj_list[current_node][1]:
            # If node isn't visited and not connected to an already visited PEC region
            if not adj_list[edge_ls[0]][2]:
                # If connected node is PEC region, then check if already reached
                locNum = nodeList[edge_ls[0],0]
                if locNum < 0:
                    if locNum in pec_reached: # Skip if already reached
                        continue
                    else:
                        pec_reached.add(locNum)
                
                # Mark the edge as a part of the tree
                tree_edges[edge_ls[1]] = True
                # Mark connected node as visited
                adj_list[edge_ls[0]][2] = True
                # Add connected node to back of queue (with special handling for PEC)
                if locNum >= 0: 
                    bfs_queue.insert(0, edge_ls[0])

    # Use the MST results to isolate the cotree edges (those not in the MST)
    newEdgeList = edgeList[tree_edges == False]

    # Create a renumbered list of edge numbers
    newIndex = np.zeros(N_e, dtype=np.int32)
    ind = 0
    for ii in range(N_e):
        if not tree_edges[ii]: # Edge is part of cotree - add number
            newIndex[ii] = ind
            ind += 1
        else: # Edge is not part of cotree - mark with -1
            newIndex[ii] = -1

    # Loop to remap edge numbers in eleEdgeList
    for ii in range(len(eleEdgeList)):
        for jj in range(6):
            if eleEdgeList[ii,jj] >= 0: # Don't renumber the PEC edges
                eleEdgeList[ii,jj] = newIndex[eleEdgeList[ii,jj]]
    # Same for portEdgeList
    for ii in range(len(portEdgeList)):
        for jj in range(3):
            if portEdgeList[ii,jj] >= 0: # Don't renumber the PEC edges
                portEdgeList[ii,jj] = newIndex[portEdgeList[ii,jj]]

    if len(newEdgeList) != N_e - N_n: # Make sure the number of edges is what's expected
        print(f"WARNING: The number of tree edges is less than the number of nodes.")

    return eleEdgeList, portEdgeList, newEdgeList, len(newEdgeList)

def tree_visualization(edgeList, nodeList):
    """
    Function to visualize the edges and nodes of a mesh
    Input: edgeList, nodeList
    Output: None, but an image is created which can be saved

    Author: Niko Detwiler
    """
    #Dictionary to allow for constant-time node accesses
    unquadraticfier = dict()
    for ii in range(len(nodeList)):
        unquadraticfier[int(nodeList[ii][0])] = ii
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    connected_nodes = set()
    for ii in edgeList:
        #Remove if statements to see PEC nodes
        if int(ii[0]) >= 0:
            connected_nodes.add(int(ii[0]))
        if int(ii[1]) >= 0:
            connected_nodes.add(int(ii[1]))

    for ii in range(int(len(edgeList) / 1)):
        try:
            #Plotting edges: not very fast, but gets the job done
            ax.plot([nodeList[unquadraticfier[int(edgeList[ii][0])]][1],
                     nodeList[unquadraticfier[int(edgeList[ii][1])]][1]],
                    [nodeList[unquadraticfier[int(edgeList[ii][0])]][2], nodeList[unquadraticfier[int(edgeList[ii][1])]][2]],
                    [nodeList[unquadraticfier[int(edgeList[ii][0])]][3], nodeList[unquadraticfier[int(edgeList[ii][1])]][3]],
                    'k', alpha = 0.3)
        except KeyError:
            continue

    inbounds = [x for x in nodeList if int(x[0]) in connected_nodes]
    #Plotting nodes
    ax.scatter([x[1] for x in inbounds], [x[2] for x in inbounds], [x[3] for x in inbounds], 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    #plt.savefig('tree_cotree_img.jpg', dpi=350)
    return
