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
from basis_funcs import basis3Dscalar_numba as b3sn
from basis_funcs import basis_vector_numba as bvn
from numba import jit
import random
import time

# quadratic solving
@jit
def basis3DSquad_numba(nodes, points):
    '''
    3D 2nd Order Scalar bases

    Arguments:
    nodes: np.array
    points: np.array

    Returns:
    fvals: np.array (10, len(points))
    gvs: np.array (10, len(points), 3)
    '''

    numPts = len(points)
    fvals = np.zeros((numPts, 10))
    gvs = np.zeros((numPts, 10, 3))

    lvalsn = np.zeros((numPts, 4))
    gvals = np.zeros((4, 3))
    for i in range(4):
        lvaln, gvaln = b3sn(nodes, i, points)

        for j in range(numPts):
            lvalsn[j][i] = lvaln[j]

        gvals[i] = gvaln

    for ind, _ in enumerate(points):
        fval = np.zeros(10)
        gv = np.zeros((10, 3))
        lvals = lvalsn[ind]

        # evaluate N1, N2, N3, N4
        for i in range(4):
            fval[i] = lvals[i] * (2 * lvals[i] - 1)
            x = (4 * lvals[i] - 1) * gvals[i][0]
            y = (4 * lvals[i] - 1) * gvals[i][1]
            z = (4 * lvals[i] - 1) * gvals[i][2]
            gv[i] = np.array([x, y, z])

        # evaluate N5, N6, N7
        for i in range(3):
            fval[4 + i] = 4 * lvals[0] * lvals[i + 1]
            x = 4 * lvals[0] * gvals[i + 1][0] + 4 * lvals[i+1] * gvals[0][0]
            y = 4 * lvals[0] * gvals[i + 1][1] + 4 * lvals[i+1] * gvals[0][1]
            z = 4 * lvals[0] * gvals[i + 1][2] + 4 * lvals[i+1] * gvals[0][2]
            gv[4 + i] = np.array([x, y, z])


        # evaluate N8, N9, N10
        fval[7] = 4 * lvals[1] * lvals[2]
        x = 4 * lvals[1] * gvals[2][0] + 4 * lvals[2] * gvals[1][0]
        y = 4 * lvals[1] * gvals[2][1] + 4 * lvals[2] * gvals[1][1]
        z = 4 * lvals[1] * gvals[2][2] + 4 * lvals[2] * gvals[1][2]
        gv[7] = np.array([x, y, z])

        fval[8] = 4 * lvals[1] * lvals[3]
        x = 4 * lvals[1] * gvals[3][0] + 4 * lvals[3] * gvals[1][0]
        y = 4 * lvals[1] * gvals[3][1] + 4 * lvals[3] * gvals[1][1]
        z = 4 * lvals[1] * gvals[3][2] + 4 * lvals[3] * gvals[1][2]
        gv[8] = np.array([x, y, z])

        fval[9] = 4 * lvals[2] * lvals[3]
        x = 4 * lvals[2] * gvals[3][0] + 4 * lvals[3] * gvals[2][0]
        y = 4 * lvals[2] * gvals[3][1] + 4 * lvals[3] * gvals[2][1]
        z = 4 * lvals[2] * gvals[3][2] + 4 * lvals[3] * gvals[2][2]
        gv[9] = np.array([x, y, z])

        fvals[ind] = fval
        gvs[ind] = gv

    fvals = np.transpose(fvals)
    gvs = np.transpose(gvs, (1, 0, 2))

    return fvals, gvs

# Old function, not being used
def basis3DSquad(nodes, points):

    [n1, n2, n3, n4] = nodes
    numPts = len(points)
    fvals = []
    gvalsx = []
    gvalsy = []
    gvalsz = []
    gvs = []
    aMat = np.array([
        [1,     1,     1,     1    ],
        [n1[0], n2[0], n3[0], n4[0]],
        [n1[1], n2[1], n3[1], n4[1]],
        [n1[2], n2[2], n3[2], n4[2]]
    ])

    lvalsn = np.zeros((numPts, 4))
    gvals = np.zeros((4, 3))
    for i in range(4):
        lvaln, gvaln = b3sn(nodes, i, points);
        # print("Shape:", lvaln.shape, gvaln.shape);
        for j in range(numPts):
            lvalsn[j][i] = lvaln[j]
        gvals[i] = gvaln
    
    l_index = 0

    for pt in points:
        fval = []
        gvx = []
        gvy = []
        gvz = []
        gv = []
        lvals = lvalsn[l_index]
        l_index += 1

        # evaluate N1, N2, N3, N4
        for i in range(4):
            fval.append(lvals[i] * (2 * lvals[i] - 1))
            gvx.append((4 * lvals[i] - 1) * gvals[i][0])
            gvy.append((4 * lvals[i] - 1) * gvals[i][1])
            gvz.append((4 * lvals[i] - 1) * gvals[i][2])
            gv.append([gvx[-1], gvy[-1], gvz[-1]])

        # evaluate N5, N6, N7
        for i in range(3):
            fval.append(4 * lvals[0] * lvals[i + 1])
            gvx.append(4 * lvals[0] * gvals[i + 1][0] + 4 * lvals[i+1] * gvals[0][0])
            gvy.append(4 * lvals[0] * gvals[i + 1][1] + 4 * lvals[i+1] * gvals[0][1])
            gvz.append(4 * lvals[0] * gvals[i + 1][2] + 4 * lvals[i+1] * gvals[0][2])
            gv.append([gvx[-1], gvy[-1], gvz[-1]])


        # evaluate N8, N9, N10
        fval.append(4 * lvals[1] * lvals[2])
        gvx.append(4 * lvals[1] * gvals[2][0] + 4 * lvals[2] * gvals[1][0])
        gvy.append(4 * lvals[1] * gvals[2][1] + 4 * lvals[2] * gvals[1][1])
        gvz.append(4 * lvals[1] * gvals[2][2] + 4 * lvals[2] * gvals[1][2])
        gv.append([gvx[-1], gvy[-1], gvz[-1]])

        fval.append(4 * lvals[1] * lvals[3])
        gvx.append(4 * lvals[1] * gvals[3][0] + 4 * lvals[3] * gvals[1][0])
        gvy.append(4 * lvals[1] * gvals[3][1] + 4 * lvals[3] * gvals[1][1])
        gvz.append(4 * lvals[1] * gvals[3][2] + 4 * lvals[3] * gvals[1][2])
        gv.append([gvx[-1], gvy[-1], gvz[-1]])

        fval.append(4 * lvals[2] * lvals[3])
        gvx.append(4 * lvals[2] * gvals[3][0] + 4 * lvals[3] * gvals[2][0])
        gvy.append(4 * lvals[2] * gvals[3][1] + 4 * lvals[3] * gvals[2][1])
        gvz.append(4 * lvals[2] * gvals[3][2] + 4 * lvals[3] * gvals[2][2])
        gv.append([gvx[-1], gvy[-1], gvz[-1]])

        fvals.append(fval)
        gvalsx.append(gvx)
        gvalsy.append(gvy)
        gvalsz.append(gvz)
        gvs.append(gv)
        # print("gvx:", gvx)
        # print("gvy:", gvy)
        # print("gvz:", gvz)
        # print("gv:", gv)
        # print(" ")

    # fvals = np.transpose(np.array(fvals))
    # print(np.array(fvals))
    # print(np.transpose(np.array(fvals)))
    # print(" ")
    # print(np.array(gvs))
    # print(" ")
    # print(np.transpose(np.array(gvs), (1, 0, 2)))
    # print(" ")
    fvals = np.transpose(np.array(fvals))
    gvs = np.transpose(np.array(gvs), (1, 0, 2))

    return fvals, gvs

def basis3DVquad(nodes, points, getDCurl=False):
    '''
    3D 2nd Order Vector bases

    Inputs:
        nodes    - Coordinates of the 4 nodes of the tetrahedron
        points   - Coordinates of the integration points
        getDCurl - Indicates whether to also return the second curl of the function

    Outputs:
        fvals    - Values of each basis function at each integration point
        cvals    - Curl of each basis function at each integration point
        dcvals   - Curl of the curl of each basis function at each integration point
    '''

    numPts = len(points)
    fvals = np.zeros((numPts, 20, 3), dtype=np.float64) # Function evaluation at each point
    cvals = np.zeros((numPts, 20, 3), dtype=np.float64) # Curl at each point
    if getDCurl:
        dcvals = np.zeros((numPts, 20, 3), dtype=np.float64) # Curl of curl at each point
    else:
        dcvals = np.zeros((0,0,0), dtype=np.float64)

    # precompute all the values to minimize calls to b2v
    W1vals, W1curl = bvn(nodes, 0, 1, points)
    W2vals, W2curl = bvn(nodes, 0, 2, points)
    W3vals, W3curl = bvn(nodes, 0, 3, points)
    W4vals, W4curl = bvn(nodes, 1, 2, points)
    W5vals, W5curl = bvn(nodes, 1, 3, points)
    W6vals, W6curl = bvn(nodes, 2, 3, points)
    W1curl = W1curl[0,:] # The curl of the first order function is constant, so only one position is needed
    W2curl = W2curl[0,:] 
    W3curl = W3curl[0,:] 
    W4curl = W4curl[0,:] 
    W5curl = W5curl[0,:] 
    W6curl = W6curl[0,:] 

    L1vals, L1grad = b3sn(nodes, 0, points)
    L2vals, L2grad = b3sn(nodes, 1, points)
    L3vals, L3grad = b3sn(nodes, 2, points)
    L4vals, L4grad = b3sn(nodes, 3, points)

    for ii in range(numPts):  
        ## Evaluate edge bases
        fvals[ii,0,:]  = W1vals[ii] * (3*L1vals[ii] - 1) #2100
        fvals[ii,1,:]  = W1vals[ii] * (3*L2vals[ii] - 1) #1200
        fvals[ii,2,:]  = W2vals[ii] * (3*L1vals[ii] - 1) #2010
        fvals[ii,3,:]  = W2vals[ii] * (3*L3vals[ii] - 1) #1020 
        fvals[ii,4,:]  = W3vals[ii] * (3*L1vals[ii] - 1) #2001
        fvals[ii,5,:]  = W3vals[ii] * (3*L4vals[ii] - 1) #1002
        fvals[ii,6,:]  = W4vals[ii] * (3*L2vals[ii] - 1) #0210
        fvals[ii,7,:]  = W4vals[ii] * (3*L3vals[ii] - 1) #0120
        fvals[ii,8,:]  = W5vals[ii] * (3*L2vals[ii] - 1) #0201
        fvals[ii,9,:]  = W5vals[ii] * (3*L4vals[ii] - 1) #0102
        fvals[ii,10,:] = W6vals[ii] * (3*L3vals[ii] - 1) #0021
        fvals[ii,11,:] = W6vals[ii] * (3*L4vals[ii] - 1) #0012

        ##  Evaluate edge curls
        cvals[ii,0,:] = W1curl * (3*L1vals[ii] - 1) + 3*np.cross(L1grad, W1vals[ii])
        cvals[ii,1,:] = W1curl * (3*L2vals[ii] - 1) + 3*np.cross(L2grad, W1vals[ii])
        cvals[ii,2,:] = W2curl * (3*L1vals[ii] - 1) + 3*np.cross(L1grad, W2vals[ii])
        cvals[ii,3,:] = W2curl * (3*L3vals[ii] - 1) + 3*np.cross(L3grad, W2vals[ii])
        cvals[ii,4,:] = W3curl * (3*L1vals[ii] - 1) + 3*np.cross(L1grad, W3vals[ii])
        cvals[ii,5,:] = W3curl * (3*L4vals[ii] - 1) + 3*np.cross(L4grad, W3vals[ii])
        cvals[ii,6,:] = W4curl * (3*L2vals[ii] - 1) + 3*np.cross(L2grad, W4vals[ii])
        cvals[ii,7,:] = W4curl * (3*L3vals[ii] - 1) + 3*np.cross(L3grad, W4vals[ii])
        cvals[ii,8,:] = W5curl * (3*L2vals[ii] - 1) + 3*np.cross(L2grad, W5vals[ii])
        cvals[ii,9,:] = W5curl * (3*L4vals[ii] - 1) + 3*np.cross(L4grad, W5vals[ii])
        cvals[ii,10,:] = W6curl * (3*L3vals[ii] - 1) + 3*np.cross(L3grad, W6vals[ii])
        cvals[ii,11,:] = W6curl * (3*L4vals[ii] - 1) + 3*np.cross(L4grad, W6vals[ii])

        if getDCurl:
            dcvals[ii,0,:] = 3*np.cross(L1grad, W1curl)
            dcvals[ii,1,:] = 3*np.cross(L2grad, W1curl)
            dcvals[ii,2,:] = 3*np.cross(L1grad, W2curl)
            dcvals[ii,3,:] = 3*np.cross(L3grad, W2curl)
            dcvals[ii,4,:] = 3*np.cross(L1grad, W3curl)
            dcvals[ii,5,:] = 3*np.cross(L4grad, W3curl)
            dcvals[ii,6,:] = 3*np.cross(L2grad, W4curl)
            dcvals[ii,7,:] = 3*np.cross(L3grad, W4curl)
            dcvals[ii,8,:] = 3*np.cross(L2grad, W5curl)
            dcvals[ii,9,:] = 3*np.cross(L4grad, W5curl)
            dcvals[ii,10,:] = 3*np.cross(L3grad, W6curl)
            dcvals[ii,11,:] = 3*np.cross(L4grad, W6curl)


        ## evaluate interior bases, one of each group of three can be discarded (arbitrarily picked the second basis)
        #1110 face
        fvals[ii,12,:] = 4.5*W1vals[ii] * L3vals[ii] 
        #                4.5*W2vals[ii] * L2vals[ii] 
        fvals[ii,13,:] = 4.5*W4vals[ii] * L1vals[ii]

        #1101 face
        fvals[ii,14,:] = 4.5*W1vals[ii] * L4vals[ii]
        #                4.5*W3vals[ii] * L2vals[ii]
        fvals[ii,15,:] = 4.5*W5vals[ii] * L1vals[ii]

        #1011 face
        fvals[ii,16,:] = 4.5*W2vals[ii] * L4vals[ii]
        #                4.5*W3vals[ii] * L3vals[ii]
        fvals[ii,17,:] = 4.5*W6vals[ii] * L1vals[ii]

        #0111
        fvals[ii,18,:] = 4.5*W4vals[ii] * L4vals[ii]
        #                4.5*W5vals[ii] * L3vals[ii]
        fvals[ii,19,:] = 4.5*W6vals[ii] * L2vals[ii]

        ## Evaluate face curls
        cvals[ii,12,:] = 4.5*(L3vals[ii]*W1curl + np.cross(L3grad, W1vals[ii]))
        #                4.5*(L2vals[ii]*W2curl + np.cross(L2grad, W2vals[ii]))
        cvals[ii,13,:] = 4.5*(L1vals[ii]*W4curl + np.cross(L1grad, W4vals[ii]))

        cvals[ii,14,:] = 4.5*(L4vals[ii]*W1curl + np.cross(L4grad, W1vals[ii]))
        #                4.5*(L2vals[ii]*W3curl + np.cross(L2grad, W3vals[ii]))
        cvals[ii,15,:] = 4.5*(L1vals[ii]*W5curl + np.cross(L1grad, W5vals[ii]))

        cvals[ii,16,:] = 4.5*(L4vals[ii]*W2curl + np.cross(L4grad, W2vals[ii]))
        #                4.5*(L3vals[ii]*W3curl + np.cross(L3grad, W3vals[ii]))
        cvals[ii,17,:] = 4.5*(L1vals[ii]*W6curl + np.cross(L1grad, W6vals[ii]))

        cvals[ii,18,:] = 4.5*(L4vals[ii]*W4curl + np.cross(L4grad, W4vals[ii]))
        #                4.5*(L3vals[ii]*W5curl + np.cross(L3grad, W5vals[ii]))
        cvals[ii,19,:] = 4.5*(L2vals[ii]*W6curl + np.cross(L2grad, W6vals[ii]))

        if getDCurl:
            dcvals[ii,12,:] = 4.5*np.cross(L3grad, W1curl)
            #                 4.5np.cross(L2grad, W2curl)
            dcvals[ii,13,:] = 4.5*np.cross(L1grad, W4curl)

            dcvals[ii,14,:] = 4.5*np.cross(L4grad, W1curl)
            #                 4.5*np.cross(L2grad, W3vals)
            dcvals[ii,15,:] = 4.5*np.cross(L1grad, W5curl)

            dcvals[ii,16,:] = 4.5*np.cross(L4grad, W2curl)
            #                 4.5*np.cross(L3grad, W3vals)
            dcvals[ii,17,:] = 4.5*np.cross(L1grad, W6curl)

            dcvals[ii,18,:] = 4.5*np.cross(L4grad, W4curl)
            #                 4.5*np.cross(L3grad, W5curl)
            dcvals[ii,19,:] = 4.5*np.cross(L2grad, W6curl)

    fvals = np.transpose(np.array(fvals), (1, 0, 2))
    cvals = np.transpose(np.array(cvals), (1, 0, 2))
    if getDCurl:
        dcvals = np.transpose(np.array(dcvals), (1, 0, 2))
        
    return fvals, cvals, dcvals
    

if (__name__ == "__main__"):

    n1 = [0,0,1]
    n2 = [0,1,0]
    n3 = [0,0,0]
    n4 = [1,0,0]

    arr = np.array([n1, n2, n3, n4])

    points = [
        [0,0,1],
        [0,1,0],
        [0,0,0],
        [1,0,0],
        [0, 0.5, 0.5],
        [0, 0, 0.5],
        [0.5, 0, 0.5],
        [0, 0.5, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0],

    ]
    # s = time.time()
    # fvals, gvals = basis3DSquad(arr, np.array(points))
    # print(fvals)
    # print(gvals)

    # for i in range(10000):
    #     arr += random.randrange(1, 10)
    #     fvals, gvals = basis3DSquad(arr, np.array(points))
    # print("Pre numba time =", time.time() - s)
    # s = time.time()
    # arr = np.array([n1, n2, n3, n4])
    # fvals, gvals = basis3DSquad_numba(arr, np.array(points))

    # for i in range(10000):
    #     arr += random.randrange(1, 10)
    #     fvals, gvals = basis3DSquad_numba(arr, np.array(points))
    # print("Post numba time =", time.time() - s)

    edgeBasis, curlBasis = basis3DVquad(arr, points)
