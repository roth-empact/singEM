import numpy as np

# Creates a file containing a set of points to evaluate the fields at for the soloXmon mesh

# singEdgePos = np.array([2.960000, 2.107164, 0.015000]) # Curve 101
# singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.000175]) # Horizontal curve
# singEdgePos = np.array([2.88625000000000, 2.1071637061436, 0.030000]) # Horizontal curve
# singEdgePos = np.array([6.371420e-01, -4.646402e-01,  1.000000e-04]) # Y axis curve on top of IBM surface participation geometry
singEdgePos = np.array([6.371420e-01, -4.646402e-01,  0.000000e-04]) # Y axis curve on bottom of IBM surface participation geometry
# cornerPos = np.array([2.960000000000001, 2.1071637061436004, 0.03])
# cornerPos = np.array([0.6371420000000002, -0.23387099999999994, 0.0001]) # Corner on top of IBM surface participation geometry
# cornerPos = np.array([0.6371420000000002, -0.23387099999999994, 0.0000]) # Corner on bottom of IBM surface participation geometry

lineDist = 1e-2

# # Along the singular edge (for a curve oriented along z)
# totPoints = singEdgePos + lineDist*np.linspace(np.array([0,0,-1]), np.array([0,0,1]), 1000) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)

# # Curve oriented along z axis
# deg0Line = np.linspace(singEdgePos-np.array([lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg45Line = np.linspace(singEdgePos-np.array([lineDist, lineDist, 0])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg90Line = np.linspace(singEdgePos-np.array([0, lineDist, 0]), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg135Line = np.linspace(singEdgePos-np.array([-lineDist, lineDist, 0])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg180Line = np.linspace(singEdgePos-np.array([-lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg225Line = np.linspace(singEdgePos-np.array([-lineDist, -lineDist, 0])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)
# deg270Line = np.linspace(singEdgePos-np.array([0, -lineDist, 0]), singEdgePos, 1001)[:-1] + np.array([1e-8, -1e-8, 0], dtype=np.float64)

# # Curve oriented along x axis
# deg0Line = np.linspace(singEdgePos-np.array([0, 0, lineDist]), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg45Line = np.linspace(singEdgePos-np.array([0, lineDist, lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg90Line = np.linspace(singEdgePos-np.array([0, lineDist, 0]), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg135Line = np.linspace(singEdgePos-np.array([0, lineDist, -lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg180Line = np.linspace(singEdgePos-np.array([0, 0, -lineDist]), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg225Line = np.linspace(singEdgePos-np.array([0, -lineDist, -lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)
# deg270Line = np.linspace(singEdgePos-np.array([0, -lineDist, 0]), singEdgePos, 1001)[:-1] + np.array([0, -1e-8, 1e-8], dtype=np.float64)

# # Curve oriented along y axis on top of PEC
# deg0Line = np.linspace(singEdgePos-np.array([0, 0, lineDist]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg45Line = np.linspace(singEdgePos-np.array([lineDist, 0, lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg90Line = np.linspace(singEdgePos-np.array([lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg135Line = np.linspace(singEdgePos-np.array([lineDist, 0, -lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg180Line = np.linspace(singEdgePos-np.array([0, 0, -lineDist]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg225Line = np.linspace(singEdgePos-np.array([-lineDist, 0, -lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)
# deg270Line = np.linspace(singEdgePos-np.array([-lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, 1e-8], dtype=np.float64)

# Curve oriented along y axis on bottom of PEC
deg0Line = np.linspace(singEdgePos-np.array([0, 0, -lineDist]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg45Line = np.linspace(singEdgePos-np.array([lineDist, 0, -lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg90Line = np.linspace(singEdgePos-np.array([lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg135Line = np.linspace(singEdgePos-np.array([lineDist, 0, lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg180Line = np.linspace(singEdgePos-np.array([0, 0, -lineDist]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg225Line = np.linspace(singEdgePos-np.array([-lineDist, 0, lineDist])/np.sqrt(2), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)
deg270Line = np.linspace(singEdgePos-np.array([-lineDist, 0, 0]), singEdgePos, 1001)[:-1] + np.array([-1e-8, 0, -1e-8], dtype=np.float64)

# # Bottom right corner
# xLine = cornerPos + lineDist/2*np.linspace(np.array([-1,0,0]), np.array([1,0,0]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# yLine = cornerPos + lineDist/2*np.linspace(np.array([0,-1,0]), np.array([0,1,0]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# zLine = cornerPos + lineDist/2*np.linspace(np.array([0,0,-1]), np.array([0,0,1]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# xyLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,-1,0]), np.array([1,1,0]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# mxyLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,1,0]), np.array([1,-1,0]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# xzLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,0,-1]), np.array([1,0,1]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# yzLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([0,-1,-1]), np.array([0,1,1]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)
# xyzLine = cornerPos + lineDist/(2*np.sqrt(3))*np.linspace(np.array([-1,-1,-1]), np.array([1,1,1]), 1001) + np.array([1e-8, -1e-8, 1e-8], dtype=np.float64)

# # Top left corner
# xLine = cornerPos + lineDist/2*np.linspace(np.array([-1,0,0]), np.array([1,0,0]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# yLine = cornerPos + lineDist/2*np.linspace(np.array([0,-1,0]), np.array([0,1,0]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# zLine = cornerPos + lineDist/2*np.linspace(np.array([0,0,-1]), np.array([0,0,1]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# xyLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,-1,0]), np.array([1,1,0]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# mxyLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,1,0]), np.array([1,-1,0]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# xzLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([-1,0,-1]), np.array([1,0,1]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# yzLine = cornerPos + lineDist/(2*np.sqrt(2))*np.linspace(np.array([0,-1,-1]), np.array([0,1,1]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)
# xyzLine = cornerPos + lineDist/(2*np.sqrt(3))*np.linspace(np.array([-1,-1,-1]), np.array([1,1,1]), 1001) + np.array([-1e-8, 1e-8, -1e-8], dtype=np.float64)

totPoints = np.vstack([deg0Line, deg45Line, deg90Line, deg135Line, deg180Line, deg225Line, deg270Line])
# totPoints = np.vstack([deg45Line, deg90Line, deg135Line, deg180Line, deg225Line])
# totPoints = np.vstack([xLine, yLine, zLine, xyLine, mxyLine, xzLine, yzLine, xyzLine])

# # Vertical line
# totPoints = np.linspace(singEdgePos-np.array([0, 0, 2*lineDist]), singEdgePos+np.array([0, 0, lineDist]), 3001) + np.array([-1e-8, 0, 0], dtype=np.float64)

# np.save("soloXmonEvalPoints_thickMetal_xCurve.npy", totPoints)
np.save("IBMSurfPartEvalPoints_vertLine.npy", totPoints)