#!python
import numpy as np
import sys
_DEFAULT_CUBIT_PATH = "C:/Program Files/Coreform Cubit 2025.12/bin"
#_DEFAULT_CUBIT_PATH = "/depot/rothte/apps/Coreform-Cubit-2025.3/bin"
sys.path.append(f'{_DEFAULT_CUBIT_PATH}')
import cubit # type: ignore
cubit.init(['cubit', '-nojournal'])
cubit.cmd('reset')

# Create a unit tet
coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int32)
for ii in range(len(coords)):
  cubit.cmd(f"create node location {coords[ii][0]} {coords[ii][1]} {coords[ii][2]}")
cubit.cmd("create tet node 1 2 3 4")

# Split it into a set of smaller tets
numsplit = 4
cubit.cmd(f"refine tet 1 NumSplit {numsplit}")

# Export the mesh
# cubit.cmd(f"export abaqus \"C:/Users/selkin/OneDrive - purdue.edu/Documents/Research/Code/CQEM-FEM/3D_Potential_FEM/Test_Files/Singular_testing/tetSplit_numSplit={numsplit}.inp\" mesh_only overwrite everything")
# cubit.cmd("export abaqus \"/scratch/negishi/selkin/3D-FEM/Meshes/surfPart/IBMSurfPart_fineEdgeRegion.inp\" mesh_only overwrite everything")










