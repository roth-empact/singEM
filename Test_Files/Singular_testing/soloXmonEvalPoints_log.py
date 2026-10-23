import numpy as np

# Creates a file containing a set of points to evaluate the fields at for the soloXmon mesh

# singEdgePos = np.array([2.960000, 2.107164, 0.015000]) # Curve 101 (vertical curve)
# singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.000175]) # Horizontal curve
singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.030000]) # Horizontal curve
# singEdgePos = np.array([6.371420e-01, -4.646402e-01,  1.000000e-04]) # Y axis curve on top of IBM surface participation geometry
# singEdgePos = np.array([6.371420e-01, -4.646402e-01,  0.000000e-04]) # Y axis curve on bottom of IBM surface participation geometry
# cornerPos = np.array([2.960000000000001, 2.1071637061436004, 0.03])
# cornerPos = np.array([0.6371420000000002, -0.23387099999999994, 0.0001]) # Corner on top of IBM surface participation geometry
# cornerPos = np.array([0.6371420000000002, -0.23387099999999994, 0.0000]) # Corner on bottom of IBM surface participation geometry

lineStart = 1e-7
lineEnd = 1e-2
numPts = 1001
points1D = 10**np.linspace(np.log10(lineStart), np.log10(lineEnd), numPts)

# # Along the singular edge (for a curve oriented along z)
# totPoints = singEdgePos + lineDist*np.linspace(np.array([0,0,-1]), np.array([0,0,1]), 1000) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)

# # Curve oriented along z axis
# deg0Line = singEdgePos + np.array([-points1D, np.zeros((numPts,)), np.zeros((numPts,))]).T
# deg45Line = singEdgePos + np.array([-points1D/np.sqrt(2), -points1D/np.sqrt(2), np.zeros((numPts,))]).T
# deg90Line = singEdgePos + np.array([np.zeros((numPts,)), -points1D, np.zeros((numPts,))]).T
# deg135Line = singEdgePos + np.array([points1D/np.sqrt(2), -points1D/np.sqrt(2), np.zeros((numPts,))]).T
# deg180Line = singEdgePos + np.array([points1D, np.zeros((numPts,)), np.zeros((numPts,))]).T
# deg225Line = singEdgePos + np.array([points1D/np.sqrt(2), points1D/np.sqrt(2), np.zeros((numPts,))]).T
# deg270Line = singEdgePos + np.array([np.zeros((numPts,)), points1D, np.zeros((numPts,))]).T

# Curve oriented along x axis
deg0Line = singEdgePos + np.array([np.zeros((numPts,)), np.zeros((numPts,)), -points1D]).T
deg45Line = singEdgePos + np.array([np.zeros((numPts,)), -points1D/np.sqrt(2), -points1D/np.sqrt(2)]).T
deg90Line = singEdgePos + np.array([np.zeros((numPts,)), -points1D, np.zeros((numPts,))]).T
deg135Line = singEdgePos + np.array([np.zeros((numPts,)), -points1D/np.sqrt(2), points1D/np.sqrt(2)]).T
deg180Line = singEdgePos + np.array([np.zeros((numPts,)), np.zeros((numPts,)), points1D]).T
deg225Line = singEdgePos + np.array([np.zeros((numPts,)), points1D/np.sqrt(2), points1D/np.sqrt(2)]).T
deg270Line = singEdgePos + np.array([np.zeros((numPts,)), points1D, np.zeros((numPts,))]).T

totPoints = np.vstack([deg0Line, deg45Line, deg90Line, deg135Line, deg180Line, deg225Line, deg270Line])

np.save("soloXmonEvalPoints_thickMetal_xCurve_log.npy", totPoints)
# np.save("IBMSurfPartEvalPoints_vertLine.npy", totPoints)