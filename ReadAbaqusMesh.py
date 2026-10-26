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

import re
import numpy as np

def readAbq(fName):
    '''
    This function reads an Abaqus (.inp) file and returns connectivity information about the mesh
        Inputs:
        fName - filepath for .inp file

        Outputs:
        nodes         - node numbers and coordinates of each node
        tets          - element number and node indices for each tetrahedral element
        wedges        - element numbers and node indices for "wedge" (triangular prism) elements
        volStartInds  - indices marking the start of each new block (for marking material parameters)
        volEleTypes   - element type associated with each start index in 'volStartInds'
        tris          - node indices for triangles on each surface
        surfStartInds - indices marking the start of each new block (for marking surface types)
    '''

    file = open(fName, encoding='utf-8')
    nodes = []                                          # safe initializations
    tets = []
    wedges = []
    tris = []
    quads = []
    lines = []
    nodesets = []
    volStartInds = []
    volEleTypes = []
    surfStartInds = []
    surfEleTypes = []
    lineStartInds = []
    nodesetStartInds = []
    cmprsChk = 1                                        # verify node numbers are compressed

    while 1:                                            # move to NODES
        if (file.readline()).find("*NODE") != -1:
            break
            
    while 1:                                            # read out NODES
        currLine = file.readline()
        if currLine.find("*") != -1:                    # terminate if end reached
            break
        else:                                           # construct [nodes]
            currLine = currLine.split(",")
            if cmprsChk != int(currLine[0]):
                raise Exception("Node numbers are not compressed. code:Abq3")
            nodes.append([float(currLine[0])-1,         # global node index
                          float(currLine[1]),           # coordinates
                          float(currLine[2]),
                          float(currLine[3])])
            cmprsChk += 1

    while 1:                                            # move to ELEMENTS
        if currLine.find("E L E M E N T S") != -1:
            currLine = file.readline()
            break
        currLine = file.readline()
    
    while currLine.find("**") == -1 and len(currLine):  # read out ELEMENTS
        if currLine.find("ELEMENT,") != -1:             # element property line catch
            eleType = currLine.split(",")               # Get the type of the current element
            eleType = eleType[1].strip()
            match eleType:                              # change if cubit decides to export a different element
                case "TYPE=C3D6": # Prism / wedge elements (6 sided)
                    eleType = "WEDGE"
                    volStartInds.append(len(wedges))
                    volEleTypes.append("Wedge")
                case "TYPE=C3D4": # Tetrahedral elements
                    eleType = "TET"
                    volStartInds.append(len(tets))
                    volEleTypes.append("Tet")
                case "TYPE=STRI3": # Triangular elements
                    eleType = "TRI"
                    surfStartInds.append(len(tris))
                case "TYPE=BAR2" | "TYPE=B21" | "TYPE=B31": # Line elements
                    eleType = "LINE"
                    lineStartInds.append(len(lines))
                case _:
                    raise Exception("Unrecognized element type. code:Abq1")
        else:
            currLine = currLine.split(",")
            if eleType == "TRI":            
                tris.append([int(currLine[1]),
                             int(currLine[2]),
                             int(currLine[3])])
            elif eleType == "LINE":
                lines.append([int(currLine[1]),
                              int(currLine[2])])
                
            elif eleType == "TET":
                tets.append([int(currLine[1]),      # node indices
                             int(currLine[2]),
                             int(currLine[3]),
                             int(currLine[4])])
                    
            elif eleType == "WEDGE":
                wedges.append([int(currLine[1]),      # node indices
                               int(currLine[2]),
                               int(currLine[3]),
                               int(currLine[4]),
                               int(currLine[5]),
                               int(currLine[6])])

        currLine = file.readline()

    while 1:                                            # move to NODESETS (if present)
        if currLine.find("N O D E S E T S") != -1:
            currLine = file.readline()
            nodesetFlag = 1
            break
        elif len(currLine) == 0:
            nodesetFlag = 0
            break # If the end of the file is reached without finding nodesets, stop reading
        currLine = file.readline()

    if nodesetFlag:
        while currLine.find("**") == -1 and len(currLine):  # read out NODESETS
            if currLine.find("NSET,") != -1:             # element property line catch
                nodesetStartInds.append(len(nodesets))
            else:
                currLine = currLine.split(",")
                for nodeNum in currLine:
                    if nodeNum != '\n':
                        nodesets.append(int(nodeNum))

            currLine = file.readline()

    
    #bndry.sort(key=itemgetter(2))                       # sort from 2D to 3D bndry
    file.close()
    return [np.array(nodes), np.array(tets), np.array(wedges), np.array(volStartInds), np.array(volEleTypes), \
            np.array(tris), np.array(quads), surfStartInds, surfEleTypes, np.array(lines), lineStartInds, np.array(nodesets), nodesetStartInds]
