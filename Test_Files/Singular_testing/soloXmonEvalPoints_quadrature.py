import numpy as np
import sys
sys.path.append("C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM")
import gauss_quadrature_tetra as q


# Creates a file containing a set of points to evaluate the fields at for the soloXmon mesh
# This version of the code creates points based on quadrature rules to evaluate a surface integral along a single triangle

# Corner triangle in soloXmon_tets_thickMetal_singLines:
# # Triangle is tri 19798 in Cubit, nodes are 9670, 10087, 10553
# nodeCoords = np.array([[2.9600000000000001, 2.1071637061436004, 0.0300000000000000],
#                        [2.9600000000000001, 2.1121637061436003, 0.0300000000000000],
#                        [2.9550833333333344, 2.1071637061436004, 0.0300000000000000]]) \
#              + np.array([0.0, 0.0, 1e-8])

# Line triangle in soloXmon_tets_thickMetal_measSurf
# Triangle is tri 20772 in Cubit, nodes are 9690, 9691, 22753 (on pc)
tri1Coords = np.array([[2.888750000000001, 2.1071637061436004, 0.0300000000000000],
                       [2.883750000000001, 2.1071637061436004, 0.0300000000000000],
                       [2.888750000000001, 2.1121637061436003, 0.0300000000000000]])

# Second triangle in soloXmon_tets_thickMetal_measSurf
# Triangle is tri 20775 in Cubit, nodes are 9709, 22860, 22861 (on laptop)
tri2Coords = np.array([[2.883750000000001, 2.1071637061436004, 0.0300000000000000],
                       [2.888750000000001, 2.1121637061436003, 0.0300000000000000],
                       [2.883750000000001, 2.1121637061436003, 0.0300000000000000]])

tri1Points = q.gauss_quadrature_trisimp_init(9, *tri1Coords) # 2D quadrature in the XY direction
tri2Points = q.gauss_quadrature_trisimp_init(9, *tri2Coords)

zPoints = q.gauss_quadrature_1D_init(11, 0.03, 0.03+5e-6) # 1D quadrature in the z direction

totPoints = np.zeros((0,3), dtype=np.float64)
for triPoints in [tri1Points, tri2Points]:
  for ii in range(len(zPoints)):
    curTriPoints = np.copy(triPoints)
    curTriPoints[:,2] = zPoints[ii]
    totPoints = np.append(totPoints, curTriPoints, axis=0)
# totPoints = q.gauss_quadrature_trisimp_init(9, *nodeCoords)


np.save("soloXmonEvalPoints_thickMetal_volPoints_line.npy", totPoints)